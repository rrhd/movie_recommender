#!/usr/bin/env python3
"""
push_svd_vectors_to_pinecone.py
--------------------------------
Upload the dense SVD vectors produced by build_feature_vector.py
into a Pinecone index.

The script …

1. reads secrets from .streamlit/secrets.toml
2. verifies / creates the index (dimension + metric must match)
3. streams vectors in batches of `BATCH_SZ`
4. attaches the metadata fields the UI / service expect so that
   later filters can be executed directly inside Pinecone.
"""

from __future__ import annotations

import sys
import time
import pickle
import sqlite3
import toml
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.stats as st
from pinecone import Pinecone, ServerlessSpec, PineconeException

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

VEC_PATH = ARTIFACTS_DIR / "final_feature_vectors.npy"
IDMAP_PATH = ARTIFACTS_DIR / "final_feature_id_map.pkl"
META_PATH = ARTIFACTS_DIR / "enriched_movies.pkl"  # optional
ZSCORE_DB_PATH = ARTIFACTS_DIR / "movie_database.db"  # optional

SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"
BATCH_SZ = 100  # ≈ 0.8 MB per batch
SPEC = ServerlessSpec(cloud="aws", region="us-east-1")
METRIC = "cosine"  # must match training
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_secrets(path: Path) -> dict[str, Any]:
    if not path.is_file():
        sys.exit(f"[ERR] secrets.toml not found: {path.resolve()}")
    secrets = toml.load(path)
    for k in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
        if k not in secrets:
            sys.exit(f"[ERR] '{k}' missing in {path.name}")
    return secrets


def ensure_index(pc: Pinecone, name: str, dim: int) -> None:
    """Create the index if necessary and verify configuration."""
    existing = pc.list_indexes().names()
    if name not in existing:
        print(f"[SETUP] creating Pinecone index '{name}' …")
        pc.create_index(name=name, dimension=dim, metric=METRIC, spec=SPEC)
        # wait until ready (≤ 2 min)
        for waited in range(0, 125, 5):
            if pc.describe_index(name).status["ready"]:
                print("[SETUP] index ready.")
                break
            print(f"  … waiting {waited}s")
            time.sleep(5)
        else:
            sys.exit("[ERR] index creation timeout.")
    else:
        desc = pc.describe_index(name)
        if desc.dimension != dim or desc.metric.lower() != METRIC.lower():
            sys.exit(
                f"[ERR] existing index cfg mismatch: dim {desc.dimension} vs {dim}, "
                f"metric {desc.metric} vs {METRIC} – delete it in console first."
            )
        print(f"[OK] using existing index '{name}'.")


def chunked(rng: range, size: int) -> Iterable[range]:
    start = rng.start
    while start < rng.stop:
        yield range(start, min(start + size, rng.stop))
        start += size


def scale_z(z: float | int | None) -> float | None:
    """Convert z-score to 0-10 scale (Φ(z) · 10)."""
    if z is None:
        return None
    return round(float(st.norm.cdf(float(z))) * 10, 2)


# ── helpers local to the upload loop ───────────────────────────────────────
def safe_int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except Exception:
        return None


def safe_float(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except Exception:
        return None


def safe_list(v):
    """
    Convert to a list; tolerate None / NaN / strings, etc.
    - None or NaN  -> []
    - list         -> list
    - "A,B"        -> ["A", "B"]
    - everything else -> []
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


# ───────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── secrets ────────────────────────────────────────────────────────────
    sec = load_secrets(SECRETS_FILE)
    api_key = sec["PINECONE_API_KEY"]
    index_name = sec["PINECONE_INDEX_NAME"]

    # ── vectors + id-map ───────────────────────────────────────────────────
    print(f"[LOAD] vectors → {VEC_PATH}")
    vecs: np.ndarray = np.load(VEC_PATH)
    dim = vecs.shape[1]

    print(f"[LOAD] id-map  → {IDMAP_PATH}")
    id_map: dict[int, str] = pickle.load(IDMAP_PATH.open("rb"))

    if vecs.shape[0] != len(id_map):
        sys.exit("[ERR] vector count and id-map length differ!")

    # ── metadata frame / dict ──────────────────────────────────────────────
    meta: dict[str, dict] = {}

    if META_PATH.exists():
        print(f"[LOAD] metadata → {META_PATH}")
        raw = pickle.load(META_PATH.open("rb"))

        # dataframe
        if isinstance(raw, pd.DataFrame):
            if "imdb_id" in raw.columns:
                df = raw.copy()
            else:
                print("[WARN] DataFrame missing 'imdb_id' – skipping metadata.")
                df = pd.DataFrame()

        # list[dict]
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
            if "imdb_id" not in df.columns:
                print("[WARN] list<dict> lacks 'imdb_id' – skipping metadata.")
                df = pd.DataFrame()
        else:
            print(f"[WARN] unsupported metadata type {type(raw)} – skipping.")
            df = pd.DataFrame()

        if not df.empty:
            want = [
                c
                for c in (
                    "title",
                    "year",
                    "rating",
                    "votes",
                    "genres",
                    "countries",
                    "languages",
                )
                if c in df.columns
            ]
            # numeric cleanup
            for col in ("year", "rating", "votes"):
                if col in want:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            meta = df[["imdb_id", *want]].set_index("imdb_id").to_dict("index")

    # ── z-scores (optional) ────────────────────────────────────────────────
    if ZSCORE_DB_PATH.exists():
        try:
            con = sqlite3.connect(f"file:{ZSCORE_DB_PATH.resolve()}?mode=ro", uri=True)
            q = """
                SELECT nr.imdb_id, nr.norm_rating
                FROM NormalisedRatings nr
                JOIN Communities c USING(community_id)
                WHERE nr.norm_type='zscore'
                  AND c.community_type LIKE '_feature_cluster_%'
            """
            rows = con.execute(q).fetchall()
            print(f"[LOAD] z-scores → {len(rows)} rows")
            for imdb_id, z in rows:
                meta.setdefault(imdb_id, {})["norm_rating"] = scale_z(z)
            con.close()
        except Exception as e:
            print(f"[WARN] could not load z-scores: {e}")

    print(f"[OK] metadata dict entries: {len(meta)}")

    # ── Pinecone setup ─────────────────────────────────────────────────────
    print("[INIT] Pinecone client …")
    pc = Pinecone(api_key=api_key)
    ensure_index(pc, index_name, dim)
    index = pc.Index(index_name)

    # ── stream upload ──────────────────────────────────────────────────────
    total = vecs.shape[0]
    print(f"[UPSERT] {total} vectors ({dim}-d) → index '{index_name}'")
    uploaded = 0

    for rng in chunked(range(total), BATCH_SZ):
        payload = []
        for i in rng:
            imdb_id = id_map[i]  # human-friendly vector id
            vec = vecs[i].tolist()
            m_raw = meta.get(imdb_id, {})

            m = {
                "title": m_raw.get("title") or None,
                "year": safe_int(m_raw.get("year")),
                "rating": safe_float(m_raw.get("rating")),
                "norm_rating": safe_float(m_raw.get("norm_rating")),
                "votes": safe_int(m_raw.get("votes")),
                "genres": safe_list(m_raw.get("genres")),
                "countries": safe_list(m_raw.get("countries")),
                "languages": safe_list(m_raw.get("languages")),
            }

            # prune empty fields
            m = {k: v for k, v in m.items() if v not in (None, [], "", {})}

            payload.append((imdb_id, vec, m))

        try:
            index.upsert(payload)
        except PineconeException as e:
            sys.exit(f"\n[ERR] upsert failed: {e}")

        uploaded += len(payload)
        print(f"  … {uploaded}/{total}", end="\r", flush=True)

    print(f"\n[DONE] all vectors uploaded ✔ ({total})")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
