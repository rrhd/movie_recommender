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

# Setup logger
logger = logging.getLogger(__name__)
logging.getLogger("title_lookup").propagate = False # If title_lookup is a module you use

# Define paths - these are now the primary data sources for the app
SCRIPT_DIR = Path(__file__).parent
APP_DATA_PKL = SCRIPT_DIR / "app_data.pkl"
# APP_TOKENS_PKL = SCRIPT_DIR / "app_tokens.pkl" # Load this if streamlit_app directly needs tokens
                                               # Otherwise, backend.py might load it.

st.set_page_config(
    page_title="Movie recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Loading movie data …")
def load_metadata() -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Loads pre-processed movie data and metadata dictionary."""
    if not APP_DATA_PKL.exists():
        err_msg = (f"Critical error: Pre-processed data file '{APP_DATA_PKL.name}' not found "
                   f"in the script directory ({SCRIPT_DIR}). "
                   "Please run the `prepare_app_data.py` script first.")
        logger.error(err_msg)
        st.error(err_msg)
        # Return empty structures to prevent app from crashing further down,
        # but indicate a fatal error.
        return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {}

    logger.info(f"Loading pre-processed app data from {APP_DATA_PKL.name}")
    try:
        df = pd.read_pickle(APP_DATA_PKL)
    except Exception as e:
        logger.error(f"Failed to load {APP_DATA_PKL.name}: {e}")
        st.error(f"Failed to load application data: {APP_DATA_PKL.name}. Check the file integrity.")
        return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {}

    if df.empty:
        logger.warning("Loaded app data DataFrame is empty.")
        # Decide if this is an error or just an empty state
        # st.warning("No movie data loaded.")
        # For now, let's assume it's possible to have an empty but valid file.
        # Fallback to empty structures.
        return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {}


    # Ensure 'imdb_id' is suitable for use as an index and for META_BY_ID
    # The prepare_app_data.py should have ensured imdb_id exists and is clean.
    if 'imdb_id' not in df.columns:
        logger.error("Critical: 'imdb_id' column missing in pre-processed app_data.pkl.")
        st.error("Data integrity issue: 'imdb_id' missing. Cannot proceed.")
        return pd.DataFrame(columns=['imdb_id']).set_index('imdb_id'), {}

    # Create META_BY_ID dictionary
    # Columns to include in the 'meta' dictionary values (should match what's in APP_DATA_PKL)
    meta_dict_val_cols = ["title", "year", "rating", "url"] # Add/remove based on app_data.pkl
    final_meta_cols_present = [col for col in meta_dict_val_cols if col in df.columns]

    # Ensure imdb_id is not already the index for this operation, or reset it.
    df_for_meta = df.copy()
    if df_for_meta.index.name == 'imdb_id':
        df_for_meta = df_for_meta.reset_index()

    meta = (
        df_for_meta[["imdb_id"] + final_meta_cols_present]
        .set_index("imdb_id")
        .to_dict(orient="index")
    )

    # The DataFrame 'df' should already have processed list columns (genres, countries, languages)
    # from app_data.pkl. No slow .apply() needed here.
    # We just need to ensure the DataFrame is indexed by 'imdb_id' for consistency if DF_META expects it.
    if df.index.name != 'imdb_id':
        df = df.set_index('imdb_id', drop=False) # drop=False keeps imdb_id as a column too if needed

    # Ensure expected list columns exist, even if empty, for downstream consistency
    for list_col_name in ["genres", "countries", "languages"]:
        if list_col_name not in df.columns:
            logger.warning(f"Expected list column '{list_col_name}' not found in app_data.pkl. Adding as empty lists.")
            df[list_col_name] = [[] for _ in range(len(df))]
        else:
            # Ensure they are lists (should be from prepare_app_data.py)
             df[list_col_name] = df[list_col_name].apply(lambda x: x if isinstance(x, list) else [])


    logger.info(f"Successfully loaded and prepared metadata. DF_META shape: {df.shape}, META_BY_ID keys: {len(meta)}")
    return df, meta # df is now DF_META


@st.cache_resource(show_spinner="Loading title lookup …")
def _title_lookup():
    from title_lookup import match_many, fuzzy_match_one
    return match_many, fuzzy_match_one

# --- Main app logic starts ---
DF_META, META_BY_ID = load_metadata()

# If load_metadata returned empty df due to critical error, stop gracefully.
if DF_META.empty and not META_BY_ID :
    st.error("Application cannot start due to missing or failed data loading. Please check logs.")
    st.stop()


match_many, fuzzy_match_one = _title_lookup()


# Dynamically generate ALL_GENRES, etc., from the loaded DF_META
# DF_META.get("genres", pd.Series([[]]*len(DF_META))) handles if "genres" column is missing
# Ensure the Series contains lists
genres_series = DF_META.get("genres", pd.Series([[] for _ in range(len(DF_META))]))
countries_series = DF_META.get("countries", pd.Series([[] for _ in range(len(DF_META))]))
languages_series = DF_META.get("languages", pd.Series([[] for _ in range(len(DF_META))]))

ALL_GENRES = sorted({g for lst in genres_series if isinstance(lst, list) for g in lst if isinstance(g, str)})
ALL_COUNTRIES = sorted({c for lst in countries_series if isinstance(lst, list) for c in lst if isinstance(c, str)})
ALL_LANGUAGES = sorted({l for lst in languages_series if isinstance(lst, list) for l in lst if isinstance(l, str)})

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
