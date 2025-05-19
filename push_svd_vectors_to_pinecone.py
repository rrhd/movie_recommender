"""
push_svd_vectors_to_pinecone.py
--------------------------------
Upload the dense SVD vectors produced by build_feature_vector.py
into a Pinecone index.

The script …

1. reads secrets from .streamlit/secrets.toml
2. verifies / creates the index (dimension + metric must match)
3. fetches existing IDs in batches to avoid re-uploading (uses Pinecone's fetch method [A])
4. streams NEW vectors in batches of `BATCH_SZ` (uses Pinecone's upsert method [B])
5. attaches the metadata fields the UI / service expect so that
   later filters can be executed directly inside Pinecone.

Citations (based on Pinecone documentation and client behavior):
[A] Pinecone `Index.fetch()` method: This operation retrieves records by their ID from an index.
    It is used here to check which of the candidate IDs already exist in the Pinecone index.
    If an ID provided in the `ids` list to `fetch()` is not found in the index, it will be
    absent from the `vectors` dictionary in the response.
    Reference: Official Pinecone Documentation (e.g., "Fetch records", "Fetch data"). [3, 4]

[B] Pinecone `Index.upsert()` method: This operation writes vectors into a namespace.
    If a new value (vector and/or metadata) is upserted for an existing vector ID,
    it will overwrite the previous value. If the ID does not exist, it creates a new vector.
    In this script, we use it specifically for IDs determined to be new after the `fetch` check
    to minimize write operations.
    Reference: Official Pinecone Documentation (e.g., "Upsert vectors", "Upsert data"). [1, 2, 7]
"""

from __future__ import annotations

import sys
import time
import pickle
import sqlite3
import toml
from pathlib import Path
from typing import Any, Iterable, List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from pinecone import Pinecone, ServerlessSpec, PineconeException


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

VEC_PATH = ARTIFACTS_DIR / "final_feature_vectors.npy"
IDMAP_PATH = ARTIFACTS_DIR / "final_feature_id_map.pkl"
META_PATH = ARTIFACTS_DIR / "enriched_movies.pkl"
ZSCORE_DB_PATH = ARTIFACTS_DIR / "movie_database.db"

SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"
BATCH_SZ = 100 # Keep batch size reasonable for both fetch and upsert
SPEC = ServerlessSpec(cloud="aws", region="us-east-1")
METRIC = "cosine"


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

        for waited in range(0, 125, 5):
            try:
                status = pc.describe_index(name).status
                if status and status.get("ready", False):
                    print("[SETUP] index ready.")
                    break
            except PineconeException as e:
                print(f"  … waiting for index, error describing: {e}")
            print(f"  … waiting {waited+5}s for index to be ready")
            time.sleep(5)
        else:
            sys.exit("[ERR] index creation timeout or error checking status.")
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
    if z is None:
        return None
    return round(float(st.norm.cdf(float(z))) * 10, 2)


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
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def main() -> None:
    sec = load_secrets(SECRETS_FILE)
    api_key = sec["PINECONE_API_KEY"]
    index_name = sec["PINECONE_INDEX_NAME"]

    print(f"[LOAD] vectors → {VEC_PATH}")
    vecs: np.ndarray = np.load(VEC_PATH)
    dim = vecs.shape[1]

    print(f"[LOAD] id-map  → {IDMAP_PATH}")
    id_map: dict[int, str] = pickle.load(IDMAP_PATH.open("rb"))

    if vecs.shape[0] != len(id_map):
        sys.exit("[ERR] vector count and id-map length differ!")

    meta: dict[str, dict] = {}

    if META_PATH.exists():
        print(f"[LOAD] metadata → {META_PATH}")
        raw = pickle.load(META_PATH.open("rb"))

        if isinstance(raw, pd.DataFrame):
            if "imdb_id" in raw.columns:
                df = raw.copy()
            else:
                print("[WARN] DataFrame missing 'imdb_id' – skipping metadata.")
                df = pd.DataFrame()

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

            for col in ("year", "rating", "votes"):
                if col in want:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            meta = df[["imdb_id", *want]].set_index("imdb_id").to_dict("index")

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

    print("[INIT] Pinecone client …")
    pc = Pinecone(api_key=api_key)
    ensure_index(pc, index_name, dim)
    index = pc.Index(index_name)

    total_items_to_process = vecs.shape[0]
    print(f"[UPSERT CHECK] {total_items_to_process} total local items ({dim}-d) for index '{index_name}'")

    actually_uploaded_count = 0
    processed_count = 0

    for rng in chunked(range(total_items_to_process), BATCH_SZ):
        batch_ids_to_check: List[str] = []
        potential_payload_map: Dict[str, Tuple[str, List[float], Dict[str, Any]]] = {}

        for i in rng:
            imdb_id = str(id_map[i])
            vec_values = vecs[i].tolist()
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

            m = {k: v for k, v in m.items() if v not in (None, [], "", {})}

            batch_ids_to_check.append(imdb_id)
            potential_payload_map[imdb_id] = (imdb_id, vec_values, m)

        existing_ids_in_pinecone = set()
        if batch_ids_to_check:
            try:
                fetch_response = index.fetch(ids=batch_ids_to_check)
                if fetch_response and fetch_response.vectors:
                    existing_ids_in_pinecone.update(fetch_response.vectors.keys())
            except PineconeException as e:
                print(f"\n[WARN] Fetching IDs failed for a batch: {e}. Will attempt to upsert all in this batch as a fallback.")
                existing_ids_in_pinecone.clear()
            except Exception as e:
                print(f"\n[WARN] Non-Pinecone error during fetch: {e}. Will attempt to upsert all in this batch.")
                existing_ids_in_pinecone.clear()

        payload_to_upsert = []
        for imdb_id_to_eval in batch_ids_to_check: # Renamed for clarity
            if imdb_id_to_eval not in existing_ids_in_pinecone:
                payload_to_upsert.append(potential_payload_map[imdb_id_to_eval])

        if payload_to_upsert:
            try:
                # Pinecone: Upsert only new vectors.
                # See Pinecone documentation for Index.upsert() [B]. [1, 2, 7]
                index.upsert(payload_to_upsert)
                actually_uploaded_count += len(payload_to_upsert)
            except PineconeException as e:
                print(f"\n[ERR] upsert failed for a batch of {len(payload_to_upsert)} items: {e}")
                print(f"Problematic batch IDs (first 5): {[item[0] for item in payload_to_upsert[:5]]}")
                sys.exit(f"\n[ERR] Critical upsert failure after filtering. Exiting.")
            except Exception as e:
                print(f"\n[ERR] Non-Pinecone error during upsert: {e}")
                sys.exit(f"\n[ERR] Critical non-Pinecone upsert failure. Exiting.")

        processed_count += len(batch_ids_to_check)
        print(f"  … processed: {processed_count}/{total_items_to_process}, new items uploaded: {actually_uploaded_count}", end="\r", flush=True)

    print(f"\n[DONE] All {total_items_to_process} local items processed. {actually_uploaded_count} new items uploaded. ✔")

if __name__ == "__main__":
    main()