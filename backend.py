#!/usr/bin/env python3
"""
backend.py  –  fast(er) Pinecone recall  ➜  re-score  ➜  diversify  ➜  filter

Key upgrades
────────────
• one *batched* fetch for all seed vectors
• ThreadPool for per-seed neighborhood queries
• vectors cached in-process (LRU) to avoid re-download
• **Maximal Marginal Relevance (MMR)** for result diversification
"""
import asyncio
import functools
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np
from cachetools import LRUCache, cached
from pinecone import PineconeException, PineconeAsyncio

from pinecone_client import INDEX, DIM, PINECONE_API_KEY, PINECONE_INDEX_NAME

# ───────────────────────────────────────────────────────────────────────────────
# constants / knobs
# ───────────────────────────────────────────────────────────────────────────────
RAW_K = 100  # neighborhood size per seed
ALPHA = 1.0  # + weight for liked similarity
BETA = 0.8  # – weight for dislike similarity
MMR_LAMBDA = 0.70  # trade-off relevance / diversity
WORKERS = min(32, (os.cpu_count() or 8) * 2)

ARTIFACTS = Path(__file__).parent / "artifacts"
METADATA_PKL = ARTIFACTS / "enriched_movies.pkl"
TOK_PATH = ARTIFACTS / "tokens.pkl"
try:
    TOKENS: Dict[str, set[str]] = pickle.load(open(TOK_PATH, "rb"))
except FileNotFoundError:
    logging.warning("tokens.pkl missing – falling back to on-the-fly tokenisation")
    TOKENS = {}

# ───────────────────────────────────────────────────────────────────────────────
# logging
# ───────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("backend")
if not log.handlers:
    logging.basicConfig(
        format="%(levelname)s:%(name)s:%(message)s",
        level=logging.INFO,
        force=True,
    )


def _fix_lists(rec: dict) -> dict:
    for key in ("genres", "countries", "languages"):
        v = rec.get(key, [])
        if not isinstance(v, list):
            rec[key] = []
    return rec


_VEC_CACHE: LRUCache[str, np.ndarray] = LRUCache(maxsize=250_000)


@cached(_VEC_CACHE)
def _fetch_vec_cached(imdb: str) -> np.ndarray:
    """Fetch **once** – further calls served from RAM."""
    try:
        v = INDEX.fetch(ids=[imdb]).vectors[imdb].values
        return np.asarray(v, dtype=np.float32)
    except PineconeException as e:
        log.warning("failed to fetch %s – %s", imdb, e)
        return np.zeros(DIM, dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                       PARALLEL NEIGHBOURHOOD RECALL                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def _query_seed(
    seed_id: str, disliked: set[str]
) -> list[tuple[str, float, list, dict]]:
    res = INDEX.query(
        id=seed_id,
        top_k=RAW_K,
        include_values=True,
        include_metadata=True,
    )
    out = []
    for m in res.matches:
        if m.id == seed_id or m.id in disliked:
            continue
        out.append((m.id, m.score, m.values, m.metadata))
    return out


async def _recall_parallel_async(
    liked: list[str],
    disliked: list[str],
) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, list[float]]]:
    seeds = liked + disliked
    if not seeds:
        return {}, {}, defaultdict(list)

    cand_meta: dict[str, dict] = {}
    cand_vec:  dict[str, np.ndarray] = {}
    sim_likes: dict[str, list[float]] = defaultdict(list)
    dneg = set(disliked)

    # open one async client & index, ensure both sessions get closed
    async with PineconeAsyncio(api_key=PINECONE_API_KEY) as pc:
        # describe_index gives you the correct endpoint URL
        desc = await pc.describe_index(PINECONE_INDEX_NAME)
        async with pc.IndexAsyncio(host=desc.host) as idx:
            # throttle so we don't spin up N sessions at once
            sem = asyncio.Semaphore(WORKERS)

            async def _sem_query(seed_id: str):
                async with sem:
                    return await idx.query(
                        id=seed_id,
                        top_k=RAW_K,
                        include_values=True,
                        include_metadata=True,
                    )

            # fire off at most WORKERS concurrent queries
            responses = await asyncio.gather(*(_sem_query(s) for s in seeds))

    # parse responses
    for seed_id, resp in zip(seeds, responses):
        is_like = seed_id in liked
        for m in resp.matches:
            if m.id == seed_id or m.id in dneg:
                continue
            cid = m.id
            cand_vec.setdefault(cid, np.asarray(m.values, dtype=np.float32))
            cand_meta.setdefault(cid, m.metadata or {})
            if is_like:
                sim_likes[cid].append(m.score)

    return cand_meta, cand_vec, sim_likes


def _recall_parallel(
    liked: list[str],
    disliked: list[str],
) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, list[float]]]:
    """Sync wrapper around the async recall function."""
    return asyncio.run(_recall_parallel_async(liked, disliked))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    DIVERSITY: MMR HELPER FUNCTIONS                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def _tokens(m: Dict[str, Any]) -> Set[str]:
    """Collect genre + decade tokens for a movie."""
    toks: Set[str] = set()

    genres = m.get("genres", [])
    if isinstance(genres, (list, tuple)):
        for g in genres:
            toks.add(str(g).lower())

    year = m.get("year")
    if isinstance(year, (int, float, str)):
        try:
            y = int(year)
            toks.add(str(y)[:3])  # decade token
        except Exception:
            pass
    return toks


def _sim_meta(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient between two token sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def _mmr_ranking(
    scores: Dict[str, float],
    k: int,
    lam: float,
) -> List[str]:
    """
    Maximal Marginal Relevance: select k items balancing relevance (scores)
    against diversity (meta-token overlap), with trade-off λ.
    """
    selected: List[str] = []
    candidates = set(scores)
    # precompute tokens
    toks = {cid: TOKENS.get(cid, set()) for cid in candidates}

    while candidates and len(selected) < k:
        best_c, best_val = None, None
        for cid in candidates:
            rel = scores[cid]
            div = max((_sim_meta(toks[cid], toks[s]) for s in selected), default=0.0)
            mmr_score = lam * rel - (1 - lam) * div
            if best_val is None or mmr_score > best_val:
                best_c, best_val = cid, mmr_score
        if best_c is None:
            break
        selected.append(best_c)
        candidates.remove(best_c)

    return selected


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                PUBLIC API                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def _freeze(v):
    if isinstance(v, list):
        return tuple(v)
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    return v

@functools.lru_cache(maxsize=2048)
def _cached_recommend(likes_key, dislikes_key, filt_key):
    return _recommend_inner(list(likes_key), list(dislikes_key), **dict(filt_key))


def recommend(liked_ids: List[str], disliked_ids: List[str], **filters):
    if not liked_ids and not disliked_ids:
        return []

    filt_key = tuple(sorted((k, _freeze(v)) for k, v in filters.items()))
    return _cached_recommend(tuple(sorted(liked_ids)),
                             tuple(sorted(disliked_ids)),
                             filt_key)


def _recommend_inner(
    liked_ids: List[str],
    disliked_ids: List[str],
    *,
    top_k: int = 15,
    # numeric filters
    min_year: int = None,
    min_rating: float = None,
    min_norm: float = None,
    min_votes: int = None,
    # include / exclude
    include_genres: List[str] = None,
    exclude_genres: List[str] = None,
    include_countries: List[str] = None,
    exclude_countries: List[str] = None,
    include_languages: List[str] = None,
    exclude_languages: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid scorer:
      • like-similarity           ← adaptive α
      • dislike-dissimilarity     ← adaptive β
      • popularity / norm_rating  ← small prior (POP_W)
    Works even with only likes, only dislikes, or none.
    """
    # ── 0) adaptive weights & prior -------------------------------------------
    n_like, n_dis = len(liked_ids), len(disliked_ids)
    alpha = ALPHA / max(1.0, np.log1p(n_like))  # log schedule
    beta = 0.0 if n_dis == 0 else BETA / np.sqrt(n_dis)
    pop_w = 0.05 if (n_like or n_dis) else 1.0  # full pop if no feedback

    # ── 1) neighbourhood recall -----------------------------------------------
    cand_meta, cand_vec, sim_likes = _recall_parallel(liked_ids, disliked_ids)
    if disliked_ids:  # pre-fetch for cache
        _ = INDEX.fetch(ids=disliked_ids)

    cand_ids = set(cand_vec) - set(liked_ids) - set(disliked_ids)
    if not cand_ids:
        return []

    # ── 2) scoring -------------------------------------------------------------
    dis_vecs = [_fetch_vec_cached(i) for i in disliked_ids]
    scores: Dict[str, float] = {}
    for cid in cand_ids:
        vec = cand_vec[cid]
        pos = float(np.mean(sim_likes[cid])) if sim_likes[cid] else 0.0
        if dis_vecs:
            dv = np.stack(dis_vecs)  # d × dim
            neg = float(
                (dv @ vec).mean()
                / (np.linalg.norm(dv, axis=1).mean() * np.linalg.norm(vec))
            )
        else:
            neg = 0.0
        base = cand_meta[cid].get("norm_rating", 0.0)
        scores[cid] = alpha * pos - beta * neg + pop_w * base

    # ── 3) hard filters --------------------------------------------------------
    def _keep(mid: str) -> bool:
        m = cand_meta.get(mid, {})
        if min_year and (m.get("year") or 0) < min_year:
            return False
        if min_rating and (m.get("rating") or 0) < min_rating:
            return False
        if min_norm and (m.get("norm_rating") or 0) < min_norm:
            return False
        if min_votes and (m.get("votes") or 0) < min_votes:
            return False
        if include_genres and not set(m.get("genres", [])).intersection(include_genres):
            return False
        if include_countries and not set(m.get("countries", [])).intersection(
            include_countries
        ):
            return False
        if include_languages and not set(m.get("languages", [])).intersection(
            include_languages
        ):
            return False
        if exclude_genres and set(m.get("genres", [])).intersection(exclude_genres):
            return False
        if exclude_countries and set(m.get("countries", [])).intersection(
            exclude_countries
        ):
            return False
        if exclude_languages and set(m.get("languages", [])).intersection(
            exclude_languages
        ):
            return False
        return True

    filtered = [
        mid
        for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if _keep(mid)
    ]
    if not filtered:
        return []

    # ── 4) diversity via MMR ---------------------------------------------------
    top_diverse = _mmr_ranking(
        {mid: scores[mid] for mid in filtered}, top_k, MMR_LAMBDA
    )

    # ── 5) output --------------------------------------------------------------
    out: List[Dict[str, Any]] = []
    for mid in top_diverse:
        m = cand_meta.get(mid, {})
        out.append(
            {
                "title": m.get("title", "N/A"),
                "year": m.get("year", ""),
                "rating": m.get("rating", ""),
                "norm_rating": m.get("norm_rating", ""),
                "score": round(scores[mid], 4),
                "url": f"https://www.imdb.com/title/{mid}/",
            }
        )
    return out


# make example with input
# Love Death and Robots
# Reservation Dogs
# Invincible
# Foundation
# American Crime Story
# Dr. Brain
# The Expanse
# The Boys
# Outer Range
# Atlanta
# Better Call Saul
# Barry
# Tehran
# Severance
# Rick and Morty
# Solar Opposites
# Harley Quinn
# The Resort
# Pantheon

def main():
    # Example usage
    liked_ids = ["tt1234567", "tt2345678"]
    disliked_ids = ["tt3456789"]
    filters = {
        "min_year": 2000,
        "min_rating": 7.0,
        "include_genres": ["Action", "Drama"],
        "exclude_genres": ["Horror"],
    }
    recommendations = recommend(liked_ids, disliked_ids, **filters)
    for rec in recommendations:
        print(rec)