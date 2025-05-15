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
# silence duplicate "loaded X titles" once Streamlit re-executes the script
logging.getLogger("title_lookup").propagate = False

# ───────── paths & cached metadata ───────────────────────────────────────
ART_DIR = Path(__file__).parent / "artifacts"
META_PKL = ART_DIR / "enriched_movies.pkl"

st.set_page_config(
    page_title="Movie recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading metadata …")
def load_metadata() -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    raw = pickle.load(open(META_PKL, "rb"))
    df = pd.DataFrame(raw) if isinstance(raw, list) else raw
    df = df.dropna(subset=["imdb_id", "title"])

    meta = (
        df[["imdb_id", "title", "year", "rating"]]
        .set_index("imdb_id")
        .to_dict(orient="index")
    )

    for col in ("genres", "countries", "languages"):
        if col not in df.columns:
            df[col] = [[]] * len(df)
        else:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    return df, meta


@st.cache_resource(show_spinner="Loading title lookup …")
def _title_lookup():
    from title_lookup import match_many, fuzzy_match_one

    return match_many, fuzzy_match_one


match_many, fuzzy_match_one = _title_lookup()

DF_META, META_BY_ID = load_metadata()  # unchanged
ALL_GENRES = sorted({g for lst in DF_META["genres"] for g in lst})
ALL_COUNTRIES = sorted({c for lst in DF_META["countries"] for c in lst})
ALL_LANGUAGES = sorted({l for lst in DF_META["languages"] for l in lst})

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

    ok_like, miss_like = resolve_lines(likes_raw)
    ok_dis, miss_dis = resolve_lines(dislikes_raw)

    liked_ids = [imdb for _raw, imdb, _ in ok_like]
    disliked_ids = [imdb for _raw, imdb, _ in ok_dis]

    if not liked_ids and not disliked_ids:
        st.error("Need at least one resolvable like or dislike.")
        st.stop()

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
