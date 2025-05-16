#!/usr/bin/env python3
"""
streamlit_app.py  –  front-end for the Pinecone movie recommender.

╭────────────────────────────  Flow for users ────────────────────────────╮
│ 1. Paste movie titles (one per line) into 👍 Liked or 👎 Disliked.       │
│ 2. Click **🔍 Preview matches** – check what the app recognised.        │
│ 3. Adjust filters in the sidebar if you like.                           │
│ 4. Hit **✨ Recommend** – enjoy clickable results.                       │
╰──────────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from backend import recommend
from utils import resolve_lines

# ──────────────────────────────────────────────────────────────────────────
# Setup logger for this module
logger = logging.getLogger(__name__)
# silence duplicate "loaded X titles" once Streamlit re-executes the script
logging.getLogger("title_lookup").propagate = False


# ───────── paths & cached metadata ───────────────────────────────────────
ART_DIR = Path(__file__).parent / "artifacts"
META_PKL = ART_DIR / "enriched_movies.pkl" # Source of truth for app metadata
# Define path for the lightweight, pre-processed data for the app
script_dir = Path(__file__).parent
lightweight_app_data_filename = f"{META_PKL.stem}_light_app.pkl"
LIGHTWEIGHT_APP_DATA_PKL = script_dir / lightweight_app_data_filename


st.set_page_config(
    page_title="Movie recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading metadata …")
def load_metadata() -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    df = None
    # Try to load lightweight pre-processed data first
    if LIGHTWEIGHT_APP_DATA_PKL.exists():
        if META_PKL.exists(): # Check against the original source file
            if LIGHTWEIGHT_APP_DATA_PKL.stat().st_mtime >= META_PKL.stat().st_mtime:
                logger.info("Found up-to-date lightweight app data: %s. Loading it.", LIGHTWEIGHT_APP_DATA_PKL.name)
                try:
                    df = pd.read_pickle(LIGHTWEIGHT_APP_DATA_PKL)
                except Exception as e:
                    logger.warning("Failed to load %s: %s. Will try to re-process from %s.", LIGHTWEIGHT_APP_DATA_PKL.name, e, META_PKL.name)
                    df = None # Ensure df is None so it gets re-processed
            else:
                logger.info("Lightweight app data %s is older than %s. Will re-process.", LIGHTWEIGHT_APP_DATA_PKL.name, META_PKL.name)
        else: # META_PKL (original) doesn't exist, but lightweight app data does
            logger.warning("Original data %s not found. Using existing lightweight app data %s.", META_PKL.name, LIGHTWEIGHT_APP_DATA_PKL.name)
            try:
                df = pd.read_pickle(LIGHTWEIGHT_APP_DATA_PKL)
            except Exception as e:
                logger.error("Failed to load existing lightweight app data %s (original %s missing): %s.", LIGHTWEIGHT_APP_DATA_PKL.name, META_PKL.name, e)
                st.error(f"Critical error: Cannot load metadata. {META_PKL.name} missing and failed to load {LIGHTWEIGHT_APP_DATA_PKL.name}")
                return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {} # Return empty structures

    if df is None: # Need to load from META_PKL and create/update lightweight version
        if not META_PKL.exists():
            logger.error(f"Critical error: {META_PKL.name} not found and no usable lightweight version available.")
            st.error(f"Critical error: {META_PKL.name} not found. Please ensure data is available.")
            return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {} # Return empty structures

        logger.info("Loading and processing full data from %s", META_PKL.name)
        try:
            raw = pickle.load(open(META_PKL, "rb"))
        except Exception as e:
            logger.error(f"Failed to load {META_PKL.name}: {e}")
            st.error(f"Failed to load primary data file: {META_PKL.name}. Please check the file.")
            return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {}

        df_full = pd.DataFrame(raw) if isinstance(raw, list) else raw

        # Define columns needed for the app
        base_required_cols = ["imdb_id", "title"] # Absolutely essential
        desired_cols_for_meta = ["year", "rating"] # For the meta dict
        list_cols = ["genres", "countries", "languages"] # For filters and display

        # Check for absolutely essential columns
        missing_essential_cols = [col for col in base_required_cols if col not in df_full.columns]
        if missing_essential_cols:
             logger.error(f"Essential columns {missing_essential_cols} missing in {META_PKL.name}")
             st.error(f"Data integrity issue: Essential columns {missing_essential_cols} missing in {META_PKL.name}")
             return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {}

        # Start with essential columns
        cols_to_select = base_required_cols[:]
        # Add other desired columns if they exist
        for col in desired_cols_for_meta + list_cols:
            if col in df_full.columns and col not in cols_to_select:
                cols_to_select.append(col)

        df = df_full[cols_to_select].copy()
        df = df.dropna(subset=["imdb_id", "title"]) # Critical dropna

        # Process list-like columns
        for col in list_cols:
            if col not in df.columns: # If column wasn't in df_full initially
                df[col] = [[] for _ in range(len(df))]
            else:
                # Ensure items are lists, convert scalars/NaNs appropriately
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else ([x] if not pd.isna(x) else []))

        # Save the processed df for next time
        try:
            LIGHTWEIGHT_APP_DATA_PKL.parent.mkdir(exist_ok=True, parents=True)
            df.to_pickle(LIGHTWEIGHT_APP_DATA_PKL)
            logger.info("SAVED processed data for app → %s", LIGHTWEIGHT_APP_DATA_PKL.name)
        except Exception as e:
            logger.error("Failed to save lightweight app data to %s: %s", LIGHTWEIGHT_APP_DATA_PKL.name, e)
            # Continue with in-memory df even if saving fails

    # ---- Create 'meta' dictionary from the DataFrame 'df' ----
    if df.empty:
        logger.warning("DataFrame is empty after loading/processing. Metadata dictionary will be empty.")
        return df, {}

    # Ensure 'imdb_id' is a column before setting as index for 'meta'
    # df might already have imdb_id as index if loaded from an older lightweight version.
    df_for_meta = df.copy()
    if df_for_meta.index.name == 'imdb_id':
        df_for_meta = df_for_meta.reset_index()

    if 'imdb_id' not in df_for_meta.columns:
        logger.error("'imdb_id' column missing from DataFrame. Cannot create metadata dictionary.")
        st.error("Critical data integrity issue: 'imdb_id' missing for metadata creation.")
        return df, {}

    # Columns to include in the 'meta' dictionary values
    meta_dict_val_cols = ["title", "year", "rating"]
    # Select only those columns that actually exist in df_for_meta for the 'meta' dictionary
    final_meta_cols_present = [col for col in meta_dict_val_cols if col in df_for_meta.columns]

    meta = (
        df_for_meta[["imdb_id"] + final_meta_cols_present]
        .set_index("imdb_id")
        .to_dict(orient="index")
    )

    for col in ("genres", "countries", "languages"):
        if col not in df.columns: # Should have been created if missing
            df[col] = [[] for _ in range(len(df))]
        else: # Ensure it's list of lists and not list of scalars mixed with NaNs
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

    return df, meta


@st.cache_resource(show_spinner="Loading title lookup …")
def _title_lookup():
    from title_lookup import match_many, fuzzy_match_one
    return match_many, fuzzy_match_one


match_many, fuzzy_match_one = _title_lookup()

DF_META, META_BY_ID = load_metadata()

# Dynamically generate ALL_GENRES, etc., from the loaded DF_META
ALL_GENRES = sorted({g for lst in DF_META.get("genres", pd.Series([[]]*(len(DF_META) if not DF_META.empty else 0))) for g in lst if isinstance(g, str)})
ALL_COUNTRIES = sorted({c for lst in DF_META.get("countries", pd.Series([[]]*(len(DF_META) if not DF_META.empty else 0))) for c in lst if isinstance(c, str)})
ALL_LANGUAGES = sorted({l for lst in DF_META.get("languages", pd.Series([[]]*(len(DF_META) if not DF_META.empty else 0))) for l in lst if isinstance(l, str)})
# save all genres, countries, languages to a file
with open(script_dir / "all_genres.txt", "w") as f:
    for g in ALL_GENRES: f.write(f"{g}\n")
with open(script_dir / "all_countries.txt", "w") as f:
    for c in ALL_COUNTRIES: f.write(f"{c}\n")
with open(script_dir / "all_languages.txt", "w") as f:
    for l in ALL_LANGUAGES: f.write(f"{l}\n")

# ───────── sidebar – filters ─────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    c1, c2 = st.columns(2)
    with c1:
        min_year = st.number_input("Min year", 0, 2035, 2020, 1)
        min_votes = st.number_input("Min votes", 0, None, 500, 100)
    with c2:
        min_imdb = st.slider("Min IMDb rating", 0.0, 10.0, 5.0, 0.1)
        min_norm = st.slider(
            "Min norm rating (0-10)",
            0.0,
            10.0,
            5.0,
            0.1,
            help="Cluster-relative Z-score rescaled to 0-10",
        )

    st.markdown("#### Include / exclude")
    ic_gen = st.multiselect("Genres (any)", ALL_GENRES)
    ex_gen = st.multiselect("Exclude genres", ALL_GENRES)
    ic_cty = st.multiselect("Countries (any)", ALL_COUNTRIES)
    ex_cty = st.multiselect("Exclude countries", ALL_COUNTRIES)
    ic_lang = st.multiselect("Languages (any)", ALL_LANGUAGES)
    ex_lang = st.multiselect("Exclude languages", ALL_LANGUAGES)

    top_k = st.slider("Results", 5, 50, 15)

# ───────── main layout ───────────────────────────────────────────────────

st.title("🎬 Movie recommender")

with st.expander("💡 How to use (click to expand)", expanded=True):
    st.markdown(
        """
1. **Paste** movie titles (one per line) into the *Liked* and/or *Disliked* boxes.<br>
2. Press **🔍 Preview matches** – you’ll see how each line was interpreted.<br> (optional)
3. Adjust **filters** in the left sidebar (year, rating, genres …).<br>
4. Hit **✨ Recommend** to get personalised suggestions (titles are clickable).
        """,
        unsafe_allow_html=True,
    )

c_l, c_r = st.columns(2)
with c_l:
    likes_txt = st.text_area("👍 Liked titles — one per line")
with c_r:
    dislikes_txt = st.text_area("👎 Disliked titles — one per line")

b_prev, b_rec = st.columns(2)
btn_preview = b_prev.button("🔍 Preview matches")
btn_recommend = b_rec.button("✨ Recommend")


# ───────── helper – resolve lines once (cached per run) ──────────────────


# ═══════════════════════════════════  1) preview  ════════════════════════════
if btn_preview:
    likes_raw = [t.strip() for t in likes_txt.splitlines() if t.strip()]
    dislikes_raw = [t.strip() for t in dislikes_txt.splitlines() if t.strip()]
    with st.spinner("🔍 Matching your titles..."):
        ok_like, miss_like = resolve_lines(likes_raw)
        ok_dis, miss_dis = resolve_lines(dislikes_raw)

        if ok_like:
            st.subheader("👍 We’ll search as Likes")
            st.dataframe(
                [
                    {
                        "Input": i,
                        "Matched title": f"{t} ({META_BY_ID.get(id_, {}).get('year', '')})",
                    }
                    for i, id_, t in ok_like
                ],
                use_container_width=True,
                hide_index=True,
            )
        if ok_dis:
            st.subheader("👎 We’ll avoid as Dislikes")
            st.dataframe(
                [
                    {
                        "Input": i,
                        "Matched title": f"{t} ({META_BY_ID.get(id_, {}).get('year', '')})",
                    }
                    for i, id_, t in ok_dis
                ],
                use_container_width=True,
                hide_index=True,
            )

        if miss_like or miss_dis:
            st.subheader("⚠ Not recognised")
            st.warning("\n".join(miss_like + miss_dis))

# ═══════════════════════════════════  2) recommend  ═══════════════════════════
if btn_recommend:
    likes_raw = [t.strip() for t in likes_txt.splitlines() if t.strip()]
    dislikes_raw = [t.strip() for t in dislikes_txt.splitlines() if t.strip()]
    with st.spinner("✨ Resolving titles and preparing recommendations..."):
        ok_like, miss_like = resolve_lines(likes_raw)
        ok_dis, miss_dis = resolve_lines(dislikes_raw)

        liked_ids = [imdb for _raw, imdb, _ in ok_like]
        disliked_ids = [imdb for _raw, imdb, _ in ok_dis]

        if not liked_ids and not disliked_ids:
            st.error("Need at least one resolvable like or dislike.")

        if miss_like or miss_dis:
            st.info("Unresolved titles: " + ", ".join(miss_like + miss_dis))

        with st.spinner("Querying Pinecone & computing recommendations …"):
            recs = recommend(
                liked_ids=liked_ids,
                disliked_ids=disliked_ids,
                top_k=top_k,
                min_year=min_year or None,
                min_rating=min_imdb or None,
                min_norm=min_norm or None,
                min_votes=min_votes or None,
                include_genres=ic_gen or None,
                exclude_genres=ex_gen or None,
                include_countries=ic_cty or None,
                exclude_countries=ex_cty or None,
                include_languages=ic_lang or None,
                exclude_languages=ex_lang or None,
            )

    st.subheader(f"Top {len(recs)} recommendations")

    if not recs:
        st.warning("No titles matched the current filters.")
    else:
        # turn title into a clickable link column
        df = pd.DataFrame(recs)
        # rename `norm_rating` to Normalized rating
        df.rename(columns={"norm_rating": "normalized rating"}, inplace=True)
        df["Movie"] = df.apply( # title with year
            lambda r: f'<a href="{r["url"]}" target="_blank">{r["title"]} ({int(r["year"])})</a>',
            axis=1,
        )
        show = df[["Movie", "rating", "normalized rating", "score"]]
        st.write(
            show.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
