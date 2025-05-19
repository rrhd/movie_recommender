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

SCRIPT_DIR_BE = Path(__file__).parent


RAW_K = 100
ALPHA = 1.0
BETA = 0.8
MMR_LAMBDA = 0.70
MMR_EMBEDDING_DIVERSITY_WEIGHT = 0.5  # 0 for pure token, 1 for pure embedding, 0.5 for equal mix
WORKERS = min(32, (os.cpu_count() or 8) * 2)

# For Centroid Scoring
GAMMA_CENTROID_LIKE = 0.15      # Weight for similarity to liked items' centroid
DELTA_CENTROID_DISLIKE = 0.10   # Weight for *dissimilarity* from disliked items' centroid

# For Pseudo Label Spreading/Propagation
PROPAGATION_ITERATIONS = 2      # Number of propagation iterations (0 to disable)
PROPAGATION_K_NEIGHBORS = 7    # Max neighbors for each candidate during propagation
PROPAGATION_ALPHA_MIX = 0.20    # Mixing factor for propagated score vs. original score in an iteration
PROPAGATION_HOP2_DAMPING = 0.6  # Damping factor for 2-hop influence

ARTIFACTS = Path(__file__).parent / "artifacts"
METADATA_PKL = ARTIFACTS / "enriched_movies.pkl"
TOK_PATH_RT = SCRIPT_DIR_BE / "tokens.pkl"
try:
    TOKENS: Dict[str, set[str]] = pickle.load(open(TOK_PATH_RT, "rb"))
    logging.info(f"Successfully loaded tokens from {TOK_PATH_RT.name}")
except FileNotFoundError:
    logging.warning(
        f"{TOK_PATH_RT.name} missing – MMR diversity will use on-the-fly tokenisation or be affected."
    )
    TOKENS = {}
except Exception as e:
    logging.error(
        f"Error loading {TOK_PATH_RT.name}: {e}. Falling back to empty tokens."
    )
    TOKENS = {}


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


def _pseudo_label_propagation(
        current_scores: Dict[str, float],
        candidate_vectors: Dict[str, np.ndarray],
        iterations: int,
        k_neighbors: int,
        alpha_mix: float,
        hop2_damping: float,
        logger: logging.Logger,
) -> Dict[str, float]:
    if not candidate_vectors or not current_scores or iterations == 0 or k_neighbors == 0:
        logger.info("Skipping pseudo-label propagation (no iterations, neighbors, data, or scores).")
        return current_scores

    logger.info(
        f"Starting pseudo-label propagation: {iterations} iterations, "
        f"k_neighbors={k_neighbors}, alpha_mix={alpha_mix}, hop2_damping={hop2_damping}."
    )

    propagated_scores = current_scores.copy()

    # Consider only candidates that are in current_scores and have vectors
    valid_ids_for_propagation = [
        cid for cid in current_scores.keys() if cid in candidate_vectors
    ]

    if len(valid_ids_for_propagation) < 2:
        logger.info("Not enough valid candidates for propagation.")
        return propagated_scores

    # Precompute local neighborhoods (top-K similar candidates for each candidate)
    # WARNING: This is O(M^2 * D) for M candidates if brute-force.
    # For larger M, an ANN index (FAISS, Annoy) on candidate_vectors would be much faster.
    # Assuming M (number of candidates after recall) is manageable (e.g., a few hundreds).
    local_neighborhoods: dict[str, list[tuple[str, float]]] = defaultdict(list)
    logger.info(f"Building local neighborhoods for {len(valid_ids_for_propagation)} candidates...")

    for i, cid1 in enumerate(valid_ids_for_propagation):
        vec1 = candidate_vectors[cid1]
        neighbor_sims = []
        for j, cid2 in enumerate(valid_ids_for_propagation):
            if i == j:  # Don't compare to self
                continue
            vec2 = candidate_vectors[cid2]
            sim = _cos(vec1, vec2)
            # Threshold to keep only reasonably similar items and positive similarity
            if sim > 0.05:
                neighbor_sims.append((cid2, sim))

        neighbor_sims.sort(key=lambda x: x[1], reverse=True)
        local_neighborhoods[cid1] = neighbor_sims[:k_neighbors]

    logger.info("Local neighborhoods built. Starting propagation iterations...")
    for iteration in range(iterations):
        scores_at_iter_start = propagated_scores.copy()
        updates_count = 0
        for cid in valid_ids_for_propagation:
            original_score_cid = scores_at_iter_start.get(cid, 0.0)  # Use .get for safety

            # 1-hop influence
            influence_1hop = 0.0
            weight_sum_1hop = 0.0
            if cid in local_neighborhoods:
                for neighbor_id, sim1 in local_neighborhoods[cid]:
                    neighbor_score = scores_at_iter_start.get(neighbor_id)
                    if neighbor_score is not None:
                        influence_1hop += neighbor_score * sim1
                        weight_sum_1hop += sim1

            # 2-hop influence
            influence_2hop = 0.0
            weight_sum_2hop = 0.0
            # Keep track of 2-hop neighbors already processed to avoid duplicates from different paths
            # and to exclude self and direct 1-hop neighbors.
            processed_2hop_neighbors = {cid}  # Initialize with self
            if cid in local_neighborhoods:
                for n_id, _ in local_neighborhoods[cid]:  # Add 1-hop neighbors
                    processed_2hop_neighbors.add(n_id)

                for neighbor_id, sim1 in local_neighborhoods[cid]:
                    if neighbor_id in local_neighborhoods:  # Check if 1-hop neighbor also has precomputed neighbors
                        for second_hop_id, sim2 in local_neighborhoods[neighbor_id]:
                            if second_hop_id in processed_2hop_neighbors:
                                continue

                            second_hop_score = scores_at_iter_start.get(second_hop_id)
                            if second_hop_score is not None:
                                path_similarity = sim1 * sim2 * hop2_damping
                                influence_2hop += second_hop_score * path_similarity
                                weight_sum_2hop += path_similarity
                                processed_2hop_neighbors.add(second_hop_id)

            total_influence = influence_1hop + influence_2hop
            total_weight_sum = weight_sum_1hop + weight_sum_2hop

            if total_weight_sum > 1e-9:  # Avoid division by zero
                propagated_value = total_influence / total_weight_sum
                propagated_scores[cid] = (1 - alpha_mix) * original_score_cid + \
                                         alpha_mix * propagated_value
                updates_count += 1
            # else: score remains unchanged if no influencing neighbors
        logger.debug(
            f"Propagation iteration {iteration + 1}/{iterations} completed. Scores updated for {updates_count} candidates.")

    logger.info("Pseudo-label propagation finished.")
    return propagated_scores

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
    cand_vec: dict[str, np.ndarray] = {}
    sim_likes: dict[str, list[float]] = defaultdict(list)
    dneg = set(disliked)

    async with PineconeAsyncio(api_key=PINECONE_API_KEY) as pc:
        desc = await pc.describe_index(PINECONE_INDEX_NAME)
        async with pc.IndexAsyncio(host=desc.host) as idx:
            sem = asyncio.Semaphore(WORKERS)

            async def _sem_query(seed_id: str):
                async with sem:
                    return await idx.query(
                        id=seed_id,
                        top_k=RAW_K,
                        include_values=True,
                        include_metadata=True,
                    )

            responses = await asyncio.gather(*(_sem_query(s) for s in seeds))

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
    recs = _cached_recommend(
            tuple(sorted(liked_ids)), tuple(sorted(disliked_ids)), filt_key)
    # exclude any that the user already likes
    liked_set = set(liked_ids)
    recs = [
            r
     for r in recs
     if r["url"].split("/title/")[-1].split("/")[0] not in liked_set
    ]
    return recs


def _recommend_inner(
    liked_ids: List[str],
    disliked_ids: List[str],
    *,
    top_k: int = 15,
    min_year: int = None,
    min_rating: float = None,
    min_norm: float = None,
    min_votes: int = None,
    include_genres: List[str] = None,
    exclude_genres: List[str] = None,
    include_countries: List[str] = None,
    exclude_countries: List[str] = None,
    include_languages: List[str] = None,
    exclude_languages: List[str] = None,
) -> List[Dict[str, Any]]:
    log.info(f"Recommendation request: {len(liked_ids)} likes, {len(disliked_ids)} dislikes, filters: {{...}}")

    n_like, n_dis = len(liked_ids), len(disliked_ids)
    alpha = ALPHA / max(1.0, np.log1p(n_like))  # Adjusted like weight
    beta = 0.0 if n_dis == 0 else BETA / np.sqrt(n_dis) # Adjusted dislike weight
    pop_w = 0.05 if (n_like or n_dis) else 1.0

    cand_meta, cand_vec, sim_likes = _recall_parallel(liked_ids, disliked_ids)

    # This fetch for disliked_ids might be redundant if _recall_parallel already gets vectors,
    # but _fetch_vec_cached ensures they are in _VEC_CACHE for centroid calculation.
    # If _recall_parallel always returns vectors for dislikes in cand_vec, this can be skipped.
    # For safety or if _recall_parallel behavior changes, keep it.
    if disliked_ids:
        # Prime cache for disliked vectors if not already handled by recall's cand_vec
        for d_id in disliked_ids: _fetch_vec_cached(d_id)


    # Ensure cand_ids are only those present in cand_vec (which _recall_parallel should guarantee)
    cand_ids_from_recall = set(cand_vec.keys())
    cand_ids = cand_ids_from_recall - set(liked_ids) - set(disliked_ids)

    if not cand_ids:
        log.info("No candidates after recall or initial filtering of seeds.")
        return []
    log.info(f"Initial {len(cand_ids)} candidates from recall.")

    # --- 1. Initial Scoring (including Centroid) ---
    log.info("Starting initial scoring with centroid components...")
    liked_centroid_vec: np.ndarray | None = None
    if liked_ids:
        liked_item_vecs = [v for i in liked_ids if (v := _fetch_vec_cached(i)) is not None and np.any(v)] # Filter out zero vectors
        if liked_item_vecs:
            liked_centroid_vec = np.mean(np.stack(liked_item_vecs), axis=0)

    disliked_centroid_vec: np.ndarray | None = None
    if disliked_ids:
        disliked_item_vecs = [v for i in disliked_ids if (v := _fetch_vec_cached(i)) is not None and np.any(v)] # Filter out zero vectors
        if disliked_item_vecs:
            disliked_centroid_vec = np.mean(np.stack(disliked_item_vecs), axis=0)

    scores: Dict[str, float] = {}
    for cid in cand_ids:
        if cid not in cand_vec: continue # Should not happen if cand_ids is from cand_vec.keys()
        vec = cand_vec[cid]

        pos_sim = float(np.mean(sim_likes[cid])) if sim_likes.get(cid) else 0.0

        neg_sim = 0.0
        # For dislike similarity, use _fetch_vec_cached for disliked item vectors
        # This part was calculating dot product with disliked item vectors directly.
        # Ensure dis_vecs are up-to-date (they are, from _fetch_vec_cached calls above or from centroid part)
        dis_item_vectors_for_neg_sim = [_fetch_vec_cached(i) for i in disliked_ids if np.any(_fetch_vec_cached(i))]
        if dis_item_vectors_for_neg_sim:
            # Original calculation of 'neg'
            dv_stack = np.stack(dis_item_vectors_for_neg_sim)
            dot_products = dv_stack @ vec
            norms_dv = np.linalg.norm(dv_stack, axis=1)
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 0 and np.mean(norms_dv) > 0 : # Check for zero norms
                 # Average of cosine similarities to disliked items
                neg_sim = float(np.mean(dot_products / (norms_dv * norm_vec + 1e-9))) # add epsilon for safety
            else:
                neg_sim = 0.0


        base_popularity = cand_meta.get(cid, {}).get("norm_rating", 0.0)

        current_score = alpha * pos_sim - beta * neg_sim + pop_w * base_popularity

        # Centroid Scoring part
        if liked_centroid_vec is not None:
            sim_to_liked_centroid = _cos(vec, liked_centroid_vec)
            current_score += GAMMA_CENTROID_LIKE * sim_to_liked_centroid

        if disliked_centroid_vec is not None:
            sim_to_disliked_centroid = _cos(vec, disliked_centroid_vec)
            current_score -= DELTA_CENTROID_DISLIKE * sim_to_disliked_centroid # Penalize similarity

        scores[cid] = current_score
    log.info(f"Initial scoring completed for {len(scores)} candidates.")

    # --- 2. Pseudo Label Propagation ---
    if PROPAGATION_ITERATIONS > 0 and PROPAGATION_K_NEIGHBORS > 0 and len(scores) > 1:
        # Pass scores of actual candidates and their vectors
        # cand_vec contains vectors for all recalled items, scores is for cand_ids
        candidate_vectors_for_propagation = {
            cid: cand_vec[cid] for cid in scores.keys() if cid in cand_vec
        }
        if candidate_vectors_for_propagation:
            scores = _pseudo_label_propagation(
                scores,
                candidate_vectors_for_propagation,
                PROPAGATION_ITERATIONS,
                PROPAGATION_K_NEIGHBORS,
                PROPAGATION_ALPHA_MIX,
                PROPAGATION_HOP2_DAMPING,
                log # Pass the logger
            )
            log.info("Pseudo-label propagation applied.")
        else:
            log.info("Skipped propagation as no candidate vectors were available for scored items.")
    else:
        log.info("Skipping pseudo-label propagation based on settings or candidate count.")


    # --- 3. Filtering ---
    def _keep(mid: str) -> bool:
        # (Your existing _keep function logic remains unchanged)
        m = cand_meta.get(mid, {}) # Ensure cand_meta has info for mid
        if not m and mid in scores: # If metadata missing for a scored candidate, decide how to handle
            log.warning(f"Metadata missing for candidate {mid}, will likely be filtered out by criteria.")
            # return False # Option: filter out if essential metadata is missing

        if min_year and (m.get("year") or 0) < min_year: return False
        if min_rating and (m.get("rating") or 0) < min_rating: return False
        if min_norm and (m.get("norm_rating") or 0) < min_norm: return False
        if min_votes and (m.get("votes") or 0) < min_votes: return False

        # Handle cases where m.get might return non-list if metadata is malformed
        meta_genres = m.get("genres", [])
        if not isinstance(meta_genres, list): meta_genres = []
        meta_countries = m.get("countries", [])
        if not isinstance(meta_countries, list): meta_countries = []
        meta_languages = m.get("languages", [])
        if not isinstance(meta_languages, list): meta_languages = []

        if include_genres and not set(meta_genres).intersection(include_genres): return False
        if include_countries and not set(meta_countries).intersection(include_countries): return False
        if include_languages and not set(meta_languages).intersection(include_languages): return False
        if exclude_genres and set(meta_genres).intersection(exclude_genres): return False
        if exclude_countries and set(meta_countries).intersection(exclude_countries): return False
        if exclude_languages and set(meta_languages).intersection(exclude_languages): return False
        return True

    # Filter candidates based on criteria. Apply to current cand_ids that have scores.
    # We need scores for filtered items for MMR.
    ids_passing_filters = [cid for cid in scores.keys() if _keep(cid)]

    if not ids_passing_filters:
        log.info("No candidates remaining after filtering.")
        return []
    log.info(f"{len(ids_passing_filters)} candidates remaining after filtering.")

    scores_for_mmr = {mid: scores[mid] for mid in ids_passing_filters}

    # --- 4. MMR with Combined Diversity (Inner Function) ---
    # This inner function can access `cand_vec` and global `TOKENS` and `MMR_EMBEDDING_DIVERSITY_WEIGHT`
    def _mmr_ranking_inner(
        scores_subset: Dict[str, float], # Scores for filtered items
        k_mmr: int,
        lambda_mmr: float,
    ) -> List[str]:
        selected_ids: List[str] = []
        candidate_pool_ids = set(scores_subset.keys())

        # Pre-fetch tokens for relevant candidates
        # Global TOKENS is used. If a token set is empty, _sim_meta handles it.
        item_tokens = {cid: TOKENS.get(cid, set()) for cid in candidate_pool_ids}

        # We need cand_vec (from outer scope) for embedding diversity
        nonlocal cand_vec # Explicitly declare usage of cand_vec from outer scope

        log.debug(f"MMR starting with {len(candidate_pool_ids)} candidates for top {k_mmr}.")

        while candidate_pool_ids and len(selected_ids) < k_mmr:
            best_candidate_id, best_mmr_score = None, -float('inf')

            for cid in candidate_pool_ids:
                relevance_score = scores_subset[cid]

                token_diversity_penalty = 0.0
                embedding_diversity_penalty = 0.0

                if selected_ids: # Calculate diversity only if items have been selected
                    # Token-based diversity penalty (max similarity to already selected)
                    if item_tokens.get(cid): # Only if current item has tokens
                        token_diversity_penalty = max(
                            (_sim_meta(item_tokens[cid], item_tokens[s_id])
                             for s_id in selected_ids if item_tokens.get(s_id)), # and selected item has tokens
                            default=0.0
                        )

                    # Embedding-based diversity penalty (max similarity to already selected)
                    if cid in cand_vec: # Current candidate must have a vector
                        embedding_diversity_penalty = max(
                            (
                                _cos(cand_vec[cid], cand_vec[s_id])
                                for s_id in selected_ids
                                if s_id in cand_vec # Selected item must have a vector
                            ),
                            default=0.0,
                        )

                # Combine diversity penalties
                # MMR_EMBEDDING_DIVERSITY_WEIGHT (0 to 1): 0 for pure token, 1 for pure embedding
                combined_diversity_penalty = (
                    MMR_EMBEDDING_DIVERSITY_WEIGHT * embedding_diversity_penalty +
                    (1 - MMR_EMBEDDING_DIVERSITY_WEIGHT) * token_diversity_penalty
                )

                # MMR formula: lambda * Relevance - (1 - lambda) * Max_Similarity_to_Selected
                current_mmr_value = lambda_mmr * relevance_score - (1 - lambda_mmr) * combined_diversity_penalty

                if current_mmr_value > best_mmr_score:
                    best_candidate_id, best_mmr_score = cid, current_mmr_value

            if best_candidate_id is None:
                log.debug("MMR: No best candidate found in this iteration, stopping.")
                break
            selected_ids.append(best_candidate_id)
            candidate_pool_ids.remove(best_candidate_id)
            log.debug(f"MMR selected: {best_candidate_id} (score: {best_mmr_score:.4f}). {len(selected_ids)}/{k_mmr} selected.")

        return selected_ids

    log.info("Starting MMR ranking...")
    top_diverse_ids = _mmr_ranking_inner(
        scores_for_mmr,
        top_k, # The final K recommendations to return
        MMR_LAMBDA
    )
    log.info(f"MMR ranking finished, {len(top_diverse_ids)} diverse recommendations.")

    # --- 5. Prepare Output ---
    output_recs: List[Dict[str, Any]] = []
    for mid in top_diverse_ids:
        m_meta = cand_meta.get(mid, {}) # Metadata for the recommended item
        output_recs.append(
            {
                "title": m_meta.get("title", "N/A"),
                "year": m_meta.get("year", ""),
                "rating": m_meta.get("rating", ""),
                "norm_rating": m_meta.get("norm_rating", ""),
                "score": round(scores.get(mid, 0.0), 4), # Use final score from 'scores' dict
                "url": f"https://www.imdb.com/title/{mid}/",
            }
        )
    return output_recs


def main():
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
