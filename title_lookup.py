#!/usr/bin/env python3
"""
title_lookup.py
---------------
Fast fuzzy *title → IMDb-ID* via cheap heuristics + char-TFIDF fallback.

▸ Build (offline): python title_lookup.py --build
▸ Runtime:         from title_lookup import fuzzy_match_one, match_many
"""

import argparse, json, logging, pickle, re, sys
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import scipy.sparse as sp
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

# ────────────────────────────────────────────────────────────── #
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
META_PKL = ARTIFACTS_DIR / "enriched_movies.pkl"
LOOKUP_JSON = ARTIFACTS_DIR / "title_lookup.json"
VECTORIZER_JOBLIB = ARTIFACTS_DIR / "title_vectorizer.joblib"
TFIDF_NPZ = ARTIFACTS_DIR / "title_tfidf.npz"
ORIG_TITLES_PKL = ARTIFACTS_DIR / "title_lookup_orig.pkl"
YEAR_MAP_PKL = ARTIFACTS_DIR / "title_year_map.pkl"

# Suffix for the lightweight input file
LIGHTWEIGHT_INPUT_SUFFIX = "_light_lookup.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("title_lookup")


def build_lookup(pkl_path: Path = META_PKL, out_json: Path = LOOKUP_JSON) -> None:
    """Offline: build title lookup + TF-IDF index + original titles map."""

    script_dir = Path(__file__).parent
    lightweight_input_filename = f"{pkl_path.stem}{LIGHTWEIGHT_INPUT_SUFFIX}"
    lightweight_input_pkl_path = script_dir / lightweight_input_filename
    df = None

    # Try to load lightweight version first if it's valid and up-to-date
    if lightweight_input_pkl_path.exists():
        if pkl_path.exists():
            if lightweight_input_pkl_path.stat().st_mtime >= pkl_path.stat().st_mtime:
                logger.info("Found up-to-date lightweight input: %s. Loading it.", lightweight_input_pkl_path.name)
                try:
                    df = pd.read_pickle(lightweight_input_pkl_path)
                except Exception as e:
                    logger.warning("Failed to load %s: %s. Will try to rebuild from %s.", lightweight_input_pkl_path.name, e, pkl_path.name)
                    df = None # Ensure df is None so it gets rebuilt
            else:
                logger.info("Lightweight input %s is older than %s. Will rebuild.", lightweight_input_pkl_path.name, pkl_path.name)
        else:
            # Original pkl_path (source) doesn't exist. build_lookup needs the source.
            logger.error("Source metadata pickle %s not found. Cannot build, even if %s exists.", pkl_path.name, lightweight_input_pkl_path.name)
            sys.exit(f"[ERR] Source metadata pickle not found: {pkl_path}")

    if df is None: # Need to load from pkl_path and create/update lightweight version
        if not pkl_path.exists():
            sys.exit(f"[ERR] Source metadata pickle not found: {pkl_path}. Cannot build.")

        logger.info("Loading full data from %s to create/update lightweight version.", pkl_path.name)
        raw_full = pickle.load(open(pkl_path, "rb"))
        df_full = pd.DataFrame(raw_full) if isinstance(raw_full, list) else raw_full

        required_cols = ["imdb_id", "title", "year"]
        missing_cols = [col for col in required_cols if col not in df_full.columns]
        if missing_cols:
            sys.exit(f"[ERR] Missing required columns {missing_cols} in {pkl_path.name}")

        df_for_lightweight = df_full[required_cols].copy()

        try:
            lightweight_input_pkl_path.parent.mkdir(exist_ok=True, parents=True)
            df_for_lightweight.to_pickle(lightweight_input_pkl_path)
            logger.info("SAVED lightweight input data → %s", lightweight_input_pkl_path.name)
        except Exception as e:
            logger.error("Failed to save lightweight input data to %s: %s", lightweight_input_pkl_path.name, e)
            # Proceed with df_for_lightweight in memory if saving failed, but log error
        df = df_for_lightweight

    # Proceed with df (loaded from lightweight or freshly processed)
    df = df.dropna(subset=["imdb_id", "title"])
    df["title_lc"] = df["title"].str.lower()

    # 1) JSON: imdb_id → lowercase title
    ids = df["imdb_id"].astype(str).tolist()
    titles_lc = df["title_lc"].tolist()
    mapping = dict(zip(ids, titles_lc))
    out_json.parent.mkdir(exist_ok=True, parents=True)
    out_json.write_text(json.dumps(mapping, ensure_ascii=False))
    logger.info("WROTE %d id→title_lc → %s", len(mapping), out_json.name)

    # 2) TF-IDF char-ngrams
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform(titles_lc) # titles_lc is from the (potentially lightweight) df
    joblib.dump(vec, VECTORIZER_JOBLIB)
    sp.save_npz(TFIDF_NPZ, X)
    logger.info(
        "SAVED vectorizer → %s, matrix → %s", VECTORIZER_JOBLIB.name, TFIDF_NPZ.name
    )

    # 3) original titles map for pretty display
    orig_map = dict(zip(ids, df["title"].astype(str).tolist()))
    pickle.dump(orig_map, open(ORIG_TITLES_PKL, "wb"))
    logger.info("SAVED original-titles map → %s", ORIG_TITLES_PKL.name)

    year_series = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    year_map = dict(zip(ids, year_series.tolist()))
    pickle.dump(year_map, open(YEAR_MAP_PKL, "wb"))
    logger.info("SAVED year map → %s", YEAR_MAP_PKL.name)


# ────────────────────────────────────────────────────────────── #
_ID2LC: dict[str, str] = {}
_IDS: list[str] = []
_VEC = _X = _ORIG_MAP = _YEAR_MAP = None


def _ensure_loaded() -> None:
    global _ID2LC, _IDS, _VEC, _X, _ORIG_MAP, _YEAR_MAP
    if _ID2LC:  # already done
        return
    if not LOOKUP_JSON.exists():
        sys.exit(f"[ERR] Lookup JSON {LOOKUP_JSON.name} not found. Please run --build.")
    _ID2LC = json.loads(LOOKUP_JSON.read_text())
    _IDS = list(_ID2LC.keys())
    _VEC = joblib.load(VECTORIZER_JOBLIB)
    _X = sp.load_npz(TFIDF_NPZ).tocsr()
    _ORIG_MAP = pickle.load(open(ORIG_TITLES_PKL, "rb"))
    _YEAR_MAP = pickle.load(open(YEAR_MAP_PKL, "rb"))


def _strip_articles(s: str) -> str:
    for a in ("the ", "a ", "an "):
        if s.startswith(a):
            return s[len(a) :]
    return s


def fuzzy_match_one(
    q: str, threshold: float = 0.1
) -> Tuple[str | None, str | None, float]:
    """
    1) Heuristic exact/bare-title + year match
    2) Bare-prefix match
    3) TF-IDF fallback
    """
    _ensure_loaded()
    raw = q.strip()
    ql = raw.lower().strip()
    if not ql:
        return None, None, 0.0

    # extract a year if present
    year = None
    m = re.search(r"\b(19|20)\d{2}\b", ql)
    if m:
        year = int(m.group())
        ql = ql[: m.start()] + ql[m.end() :]  # drop the year
        ql = ql.strip().strip("():-")

    bare = _strip_articles(ql)

    # --- 1) exact bare-title match ---
    exact = [i for i, t in enumerate(_ID2LC.values()) if _strip_articles(t) == bare]
    if exact and year is not None:
        # prefer ones matching year
        exact_year = [i for i in exact if _YEAR_MAP.get(_IDS[i]) == year]
        if exact_year:
            exact = exact_year
    if exact:
        i = exact[0]
        # Using .get for _ORIG_MAP for consistency with other parts, though it should exist
        return _IDS[i], _ORIG_MAP.get(_IDS[i], _ID2LC[_IDS[i]].title()), 1.0

    if len(bare) > 7:
        pattern = re.compile(rf"^{re.escape(bare)}\b")
        pref = [
            i
            for i, t in enumerate(_ID2LC.values()) # t is _ID2LC[_IDS[i]]
            if pattern.match(_strip_articles(t))
        ]
        if pref:
            # pick the one with highest fuzzy score instead of shortest
            scores = {
                i: fuzz.WRatio(bare, _strip_articles(_ID2LC[_IDS[i]]))
                for i in pref
            }
            best = max(scores, key=scores.get)
            # Using .get for _ORIG_MAP for consistency
            return _IDS[best], _ORIG_MAP.get(_IDS[best], _ID2LC[_IDS[best]].title()), scores[best] / 100.0

    # --- 2) bare-prefix match (e.g. "terminator" → "the terminator") ---
    pref = [
        i
        for i, t in enumerate(_ID2LC.values())
        if _strip_articles(t).startswith(bare + " ")
    ]
    if pref and year is not None:
        pref_year = [i for i in pref if _YEAR_MAP.get(_IDS[i]) == year]
        if pref_year:
            pref = pref_year
    if pref:
        # pick the shortest match (likely correct)
        i = min(pref, key=lambda i: len(_ID2LC[_IDS[i]]))
        return _IDS[i], _ORIG_MAP.get(_IDS[i], _ID2LC[_IDS[i]].title()), 0.9

    # --- 3) TF-IDF fallback ---
    if _VEC is None or _X is None: # Should not happen if _ensure_loaded worked
        logger.warning("TF-IDF vectorizer or matrix not loaded, TF-IDF fallback unavailable.")
        return None, None, 0.0

    qv = _VEC.transform([ql])  # 1×F
    sims = (_X @ qv.T).toarray().ravel()
    idxs = np.where(sims >= threshold)[0]
    if idxs.size == 0:
        return None, None, 0.0

    # take top sim
    i = idxs[np.argmax(sims[idxs])]
    return _IDS[i], _ORIG_MAP.get(_IDS[i], _ID2LC[_IDS[i]].title()), float(sims[i])


def match_many(queries: Iterable[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    ok, miss = [], []
    for q in queries:
        _id, pretty, _ = fuzzy_match_one(q)
        if _id:
            ok.append((_id, pretty))
        else:
            miss.append(q)
    return ok, miss


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build / use title lookup")
    p.add_argument("--build", action="store_true", help="(re)generate everything")
    p.add_argument("--pkl", type=Path, default=META_PKL, help="Path to enriched_movies.pkl or similar.")
    p.add_argument("--out", type=Path, default=LOOKUP_JSON, help="Path to output lookup.json.")
    p.add_argument("titles", nargs="*", help="titles to test (omit for --build)")
    args = p.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure ARTIFACTS_DIR exists for storing outputs

    if args.build:
        build_lookup(args.pkl, args.out)
        sys.exit(0)
    if not args.titles:
        p.print_help()
        sys.exit(1)

    _ensure_loaded() # Ensure data is loaded for matching if not building
    found, missing = match_many(args.titles)
    print("\nMatched:")
    for _id, ttl in found:
        print(f"  {_id} ← {ttl}")
    if missing:
        print("\nNot found:")
        for t in missing:
            print("  ", t)