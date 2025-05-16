#!/usr/bin/env python3
"""
title_lookup.py
---------------
Fast fuzzy *title → IMDb-ID* via cheap heuristics + char-TFIDF fallback.

▸ Build (offline): python title_lookup.py --build
▸ Runtime:         from title_lookup import fuzzy_match_one, match_many
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ────────────────────────────────────────────────────────────── #
# Script directory for runtime files
SCRIPT_DIR_TL = Path(__file__).parent

# --- Paths for RUNTIME files (co-located with this script) ---
LOOKUP_JSON_RT = SCRIPT_DIR_TL / "title_lookup.json"
VECTORIZER_JOBLIB_RT = SCRIPT_DIR_TL / "title_vectorizer.joblib"
TFIDF_NPZ_RT = SCRIPT_DIR_TL / "title_tfidf.npz"
ORIG_TITLES_PKL_RT = SCRIPT_DIR_TL / "title_lookup_orig.pkl"
YEAR_MAP_PKL_RT = SCRIPT_DIR_TL / "title_year_map.pkl"

# --- Paths for BUILD process (source from artifacts, lightweight cache in script_dir) ---
BUILD_ARTIFACTS_DIR = SCRIPT_DIR_TL / "artifacts" # Assuming artifacts is relative to title_lookup.py for build
META_PKL_BUILD_SOURCE = BUILD_ARTIFACTS_DIR / "enriched_movies.pkl"
LIGHTWEIGHT_INPUT_SUFFIX = "_light_lookup.pkl" # For caching during build

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("title_lookup")


def build_lookup(
    pkl_path: Path = META_PKL_BUILD_SOURCE, # Source for build
    # Output paths default to runtime paths in SCRIPT_DIR_TL
    out_json: Path = LOOKUP_JSON_RT,
    out_vectorizer: Path = VECTORIZER_JOBLIB_RT,
    out_tfidf: Path = TFIDF_NPZ_RT,
    out_orig_titles: Path = ORIG_TITLES_PKL_RT,
    out_year_map: Path = YEAR_MAP_PKL_RT,
) -> None:
    """Offline: build title lookup + TF-IDF index + original titles map.
    Outputs are saved to the script directory for runtime use.
    """
    script_dir = SCRIPT_DIR_TL # Outputs and lightweight cache go here
    lightweight_input_filename = f"{pkl_path.stem}{LIGHTWEIGHT_INPUT_SUFFIX}"
    lightweight_input_pkl_path = script_dir / lightweight_input_filename
    df = None

    if lightweight_input_pkl_path.exists():
        if pkl_path.exists(): # pkl_path is the source like META_PKL_BUILD_SOURCE
            if lightweight_input_pkl_path.stat().st_mtime >= pkl_path.stat().st_mtime:
                logger.info("Found up-to-date lightweight input for build: %s. Loading it.", lightweight_input_pkl_path.name)
                try:
                    df = pd.read_pickle(lightweight_input_pkl_path)
                except Exception as e:
                    logger.warning("Failed to load %s: %s. Will try to rebuild from %s.", lightweight_input_pkl_path.name, e, pkl_path.name)
                    df = None
            else:
                logger.info("Lightweight input %s is older than %s. Will rebuild.", lightweight_input_pkl_path.name, pkl_path.name)
        else:
            logger.error("Source metadata pickle %s not found. Cannot build, even if %s exists.", pkl_path.name, lightweight_input_pkl_path.name)
            sys.exit(f"[ERR] Source metadata pickle not found: {pkl_path}")

    if df is None:
        if not pkl_path.exists():
            sys.exit(f"[ERR] Source metadata pickle not found: {pkl_path}. Cannot build.")
        logger.info("Loading full data from %s to create/update lightweight version for build.", pkl_path.name)
        raw_full = pickle.load(open(pkl_path, "rb"))
        df_full = pd.DataFrame(raw_full) if isinstance(raw_full, list) else raw_full
        required_cols = ["imdb_id", "title", "year"]
        missing_cols = [col for col in required_cols if col not in df_full.columns]
        if missing_cols:
            sys.exit(f"[ERR] Missing required columns {missing_cols} in {pkl_path.name}")
        df_for_lightweight = df_full[required_cols].copy()
        try:
            lightweight_input_pkl_path.parent.mkdir(exist_ok=True, parents=True) # script_dir
            df_for_lightweight.to_pickle(lightweight_input_pkl_path)
            logger.info("SAVED lightweight input data for build cache → %s", lightweight_input_pkl_path.name)
        except Exception as e:
            logger.error("Failed to save lightweight input data to %s: %s", lightweight_input_pkl_path.name, e)
        df = df_for_lightweight

    df = df.dropna(subset=["imdb_id", "title"])
    df["title_lc"] = df["title"].str.lower()

    ids = df["imdb_id"].astype(str).tolist()
    titles_lc = df["title_lc"].tolist()
    mapping = dict(zip(ids, titles_lc))
    out_json.parent.mkdir(exist_ok=True, parents=True) # Ensure script_dir exists
    out_json.write_text(json.dumps(mapping, ensure_ascii=False))
    logger.info("WROTE %d id→title_lc → %s", len(mapping), out_json.name)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform(titles_lc)
    joblib.dump(vec, out_vectorizer)
    sp.save_npz(out_tfidf, X)
    logger.info("SAVED vectorizer → %s, matrix → %s", out_vectorizer.name, out_tfidf.name)

    orig_map = dict(zip(ids, df["title"].astype(str).tolist()))
    pickle.dump(orig_map, open(out_orig_titles, "wb"))
    logger.info("SAVED original-titles map → %s", out_orig_titles.name)

    year_series = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    year_map = dict(zip(ids, year_series.tolist()))
    pickle.dump(year_map, open(out_year_map, "wb"))
    logger.info("SAVED year map → %s", out_year_map.name)

# Global variables for loaded data
_ID2LC: dict[str, str] = {}
_IDS: list[str] = []
_VEC = _X = _ORIG_MAP = _YEAR_MAP = None

def _ensure_loaded() -> None:
    global _ID2LC, _IDS, _VEC, _X, _ORIG_MAP, _YEAR_MAP
    if _ID2LC:  # already done
        return
    # Load from runtime paths in SCRIPT_DIR_TL
    if not LOOKUP_JSON_RT.exists():
        sys.exit(f"[ERR] Lookup JSON {LOOKUP_JSON_RT.name} not found in script directory. Please run --build.")
    _ID2LC = json.loads(LOOKUP_JSON_RT.read_text())
    _IDS = list(_ID2LC.keys())
    _VEC = joblib.load(VECTORIZER_JOBLIB_RT)
    _X = sp.load_npz(TFIDF_NPZ_RT).tocsr()
    _ORIG_MAP = pickle.load(open(ORIG_TITLES_PKL_RT, "rb"))
    _YEAR_MAP = pickle.load(open(YEAR_MAP_PKL_RT, "rb"))

# ... (rest of fuzzy_match_one, match_many, _strip_articles functions remain the same) ...

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build / use title lookup")
    p.add_argument("--build", action="store_true", help="(re)generate everything")
    p.add_argument("--pkl", type=Path, default=META_PKL_BUILD_SOURCE, help="Path to source enriched_movies.pkl for build.")
    # Output paths are now handled by defaults in build_lookup
    p.add_argument("titles", nargs="*", help="titles to test (omit for --build)")
    args = p.parse_args()

    # SCRIPT_DIR_TL.mkdir(parents=True, exist_ok=True) # Ensure script dir exists for build outputs

    if args.build:
        build_lookup(args.pkl) # Uses new defaults for output paths
        sys.exit(0)
    if not args.titles:
        p.print_help()
        sys.exit(1)

    _ensure_loaded() # Ensure data is loaded for matching if not building