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
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from backend import recommend
from utils import resolve_lines


logger = logging.getLogger(__name__)
logging.getLogger("title_lookup").propagate = False


SCRIPT_DIR = Path(__file__).parent
APP_DATA_PKL = SCRIPT_DIR / "app_data.pkl"


st.set_page_config(
    page_title="Movie recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "preview_output" not in st.session_state:
    st.session_state.preview_output = None
if "recommendation_output" not in st.session_state:
    st.session_state.recommendation_output = None
if "last_submitted_likes_preview" not in st.session_state:
    st.session_state.last_submitted_likes_preview = ""
if "last_submitted_dislikes_preview" not in st.session_state:
    st.session_state.last_submitted_dislikes_preview = ""
if "last_submitted_likes_recommend" not in st.session_state:
    st.session_state.last_submitted_likes_recommend = ""
if "last_submitted_dislikes_recommend" not in st.session_state:
    st.session_state.last_submitted_dislikes_recommend = ""
if "likes_txt_content" not in st.session_state:
    st.session_state.likes_txt_content = ""
if "dislikes_txt_content" not in st.session_state:
    st.session_state.dislikes_txt_content = ""


@st.cache_resource(show_spinner="Loading movie data …")
def load_metadata() -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Loads pre-processed movie data and metadata dictionary."""
    if not APP_DATA_PKL.exists():
        err_msg = (
            f"Critical error: Pre-processed data file '{APP_DATA_PKL.name}' not found "
            f"in the script directory ({SCRIPT_DIR}). "
            "Please run the `prepare_app_data.py` script first."
        )
        logger.error(err_msg)
        st.error(err_msg)

        return pd.DataFrame(columns=["imdb_id"]).set_index("imdb_id"), {}

    logger.info(f"Loading pre-processed app data from {APP_DATA_PKL.name}")
    try:
        df = pd.read_pickle(APP_DATA_PKL)
    except Exception as e:
        logger.error(f"Failed to load {APP_DATA_PKL.name}: {e}")
        st.error(
            f"Failed to load application data: {APP_DATA_PKL.name}. Check the file integrity."
        )
        return pd.DataFrame(columns=["imdb_id"]).set_index("imdb_id"), {}

    if df.empty:
        logger.warning("Loaded app data DataFrame is empty.")

        return pd.DataFrame(columns=["imdb_id"]).set_index("imdb_id"), {}

    if "imdb_id" not in df.columns:
        logger.error(
            "Critical: 'imdb_id' column missing in pre-processed app_data.pkl."
        )
        st.error("Data integrity issue: 'imdb_id' missing. Cannot proceed.")
        return pd.DataFrame(columns=["imdb_id"]).set_index("imdb_id"), {}

    meta_dict_val_cols = ["title", "year", "rating", "url"]
    final_meta_cols_present = [col for col in meta_dict_val_cols if col in df.columns]

    df_for_meta = df.copy()
    if df_for_meta.index.name == "imdb_id":
        df_for_meta = df_for_meta.reset_index()

    meta = (
        df_for_meta[["imdb_id"] + final_meta_cols_present]
        .set_index("imdb_id")
        .to_dict(orient="index")
    )

    if df.index.name != "imdb_id":
        df = df.set_index("imdb_id", drop=False)

    for list_col_name in ["genres", "countries", "languages"]:
        if list_col_name not in df.columns:
            logger.warning(
                f"Expected list column '{list_col_name}' not found in app_data.pkl. Adding as empty lists."
            )
            df[list_col_name] = [[] for _ in range(len(df))]
        else:
            df[list_col_name] = df[list_col_name].apply(
                lambda x: x if isinstance(x, list) else []
            )

    logger.info(
        f"Successfully loaded and prepared metadata. DF_META shape: {df.shape}, META_BY_ID keys: {len(meta)}"
    )
    return df, meta


@st.cache_resource(show_spinner="Loading title lookup …")
def _title_lookup():
    from title_lookup import match_many, fuzzy_match_one

    return match_many, fuzzy_match_one


DF_META, META_BY_ID = load_metadata()


if DF_META.empty and not META_BY_ID:
    st.error(
        "Application cannot start due to missing or failed data loading. Please check logs."
    )
    st.stop()


match_many, fuzzy_match_one = _title_lookup()


genres_series = DF_META.get("genres", pd.Series([[] for _ in range(len(DF_META))]))
countries_series = DF_META.get(
    "countries", pd.Series([[] for _ in range(len(DF_META))])
)
languages_series = DF_META.get(
    "languages", pd.Series([[] for _ in range(len(DF_META))])
)

ALL_GENRES = sorted(
    {
        g
        for lst in genres_series
        if isinstance(lst, list)
        for g in lst
        if isinstance(g, str)
    }
)
ALL_COUNTRIES = sorted(
    {
        c
        for lst in countries_series
        if isinstance(lst, list)
        for c in lst
        if isinstance(c, str)
    }
)
ALL_LANGUAGES = sorted(
    {
        l
        for lst in languages_series
        if isinstance(lst, list)
        for l in lst
        if isinstance(l, str)
    }
)

if (
    st.session_state.recommendation_output
    and st.session_state.recommendation_output.get("recs")
):
    processed_row_action = False

    for rec_movie_action in st.session_state.recommendation_output["recs"]:
        movie_id_from_url = (
            rec_movie_action.get("url", "").split("/title/")[-1].split("/")[0]
        )
        if not movie_id_from_url.startswith("tt"):
            continue

        movie_title_action = rec_movie_action.get("title", "N/A")
        movie_year_action_val = rec_movie_action.get("year")
        movie_year_action_str = (
            str(int(movie_year_action_val))
            if pd.notna(movie_year_action_val)
            and str(movie_year_action_val).strip() != ""
            else "N/A"
        )
        title_to_add = f"{movie_title_action} ({movie_year_action_str})"

        like_button_key = f"add_like_{movie_id_from_url}"
        dislike_button_key = f"add_dislike_{movie_id_from_url}"

        if st.session_state.get(like_button_key):
            current_likes_list = [
                line.strip()
                for line in st.session_state.likes_txt_content.splitlines()
                if line.strip()
            ]
            if title_to_add not in current_likes_list:
                current_likes_list.append(title_to_add)
                st.session_state.likes_txt_content = "\n".join(current_likes_list)
            del st.session_state[like_button_key]
            processed_row_action = True

        if st.session_state.get(dislike_button_key):
            current_dislikes_list = [
                line.strip()
                for line in st.session_state.dislikes_txt_content.splitlines()
                if line.strip()
            ]
            if title_to_add not in current_dislikes_list:
                current_dislikes_list.append(title_to_add)
                st.session_state.dislikes_txt_content = "\n".join(current_dislikes_list)
            del st.session_state[dislike_button_key]
            processed_row_action = True

    if processed_row_action:
        st.rerun()

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


st.title("🎬 Movie recommender")

with st.expander("💡 How to use (click to expand)", expanded=True):
    st.markdown(
        """
1.  **Enter Titles:** Paste or type movie titles (one per line) into the 👍 **Liked** and/or 👎 **Disliked** boxes.
2.  **Preview (Optional):** Click **🔍 Preview matches**.
    *   The text boxes will update to show how your input was recognized.
    *   A preview of matched titles will appear below.
3.  **Get Recommendations:** Click **✨ Recommend**.
    *   The text boxes will update with the recognized titles used for this recommendation set.
    *   Personalized, clickable movie suggestions will appear below.
4.  **Refine & Iterate:**
    *   Use the 👍 or 👎 buttons next to any recommended movie to quickly add it to your Liked/Disliked lists. The text boxes will update automatically.
    *   Adjust **filters** in the left sidebar (year, rating, genres, etc.) at any time.
    *   Click **✨ Recommend** again with your updated lists and filters to get new suggestions.
        """,
        unsafe_allow_html=True,
    )

c_l, c_r = st.columns(2)
with c_l:
    likes_txt_val_from_ss = st.session_state.get("likes_txt_content", "")
    likes_txt = st.text_area(
        "👍 Liked titles — one per line",
        value=likes_txt_val_from_ss,
        key="likes_text_area_widget",
    )
    if likes_txt != likes_txt_val_from_ss:
        st.session_state.likes_txt_content = likes_txt


with c_r:
    dislikes_txt_val_from_ss = st.session_state.get("dislikes_txt_content", "")
    dislikes_txt = st.text_area(
        "👎 Disliked titles — one per line",
        value=dislikes_txt_val_from_ss,
        key="dislikes_text_area_widget",
    )
    if dislikes_txt != dislikes_txt_val_from_ss:
        st.session_state.dislikes_txt_content = dislikes_txt


b_prev, b_rec = st.columns(2)
btn_preview = b_prev.button("🔍 Preview matches")
btn_recommend = b_rec.button("✨ Recommend")

if btn_preview:
    st.session_state.recommendation_output = None

    current_likes_input_str = likes_txt
    current_dislikes_input_str = dislikes_txt

    likes_raw_list = [
        t.strip() for t in current_likes_input_str.splitlines() if t.strip()
    ]
    dislikes_raw_list = [
        t.strip() for t in current_dislikes_input_str.splitlines() if t.strip()
    ]

    if not likes_raw_list and not dislikes_raw_list:
        st.session_state.preview_output = {"error": "Please enter titles to preview."}

    else:
        with st.spinner("🔍 Matching your titles..."):
            ok_like, miss_like = resolve_lines(likes_raw_list)
            ok_dis, miss_dis = resolve_lines(dislikes_raw_list)

        st.session_state.preview_output = {
            "ok_like": ok_like,
            "miss_like": miss_like,
            "ok_dis": ok_dis,
            "miss_dis": miss_dis,
            "error": None,
        }

        matched_likes_display_list = [
            f"{title.strip()} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})"
            for _, id_, title in ok_like
        ]
        st.session_state.likes_txt_content = "\n".join(matched_likes_display_list)

        matched_dislikes_display_list = [
            f"{title.strip()} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})"
            for _, id_, title in ok_dis
        ]
        st.session_state.dislikes_txt_content = "\n".join(matched_dislikes_display_list)

    st.session_state.last_submitted_likes_preview = current_likes_input_str
    st.session_state.last_submitted_dislikes_preview = current_dislikes_input_str
    st.rerun()

if btn_recommend:
    st.session_state.preview_output = None

    current_likes_input_str = likes_txt
    current_dislikes_input_str = dislikes_txt

    likes_raw_list = [
        t.strip() for t in current_likes_input_str.splitlines() if t.strip()
    ]
    dislikes_raw_list = [
        t.strip() for t in current_dislikes_input_str.splitlines() if t.strip()
    ]

    if not likes_raw_list and not dislikes_raw_list:
        st.session_state.recommendation_output = {
            "error": "Need at least one like or dislike to generate recommendations."
        }

    else:
        with st.spinner("✨ Resolving titles and preparing recommendations..."):
            ok_like, miss_like = resolve_lines(likes_raw_list)
            ok_dis, miss_dis = resolve_lines(dislikes_raw_list)
            liked_ids = [imdb for _raw, imdb, _ in ok_like]
            disliked_ids = [imdb for _raw, imdb, _ in ok_dis]

            if not liked_ids and not disliked_ids:
                unresolved_msg = ""
                if miss_like or miss_dis:
                    unresolved_msg = " Unresolved titles: " + ", ".join(
                        miss_like + miss_dis
                    )
                st.session_state.recommendation_output = {
                    "error": f"Need at least one resolvable like or dislike.{unresolved_msg}"
                }

                matched_likes_display_list = [
                    f"{title.strip()} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})"
                    for _, id_, title in ok_like
                ]
                st.session_state.likes_txt_content = "\n".join(
                    matched_likes_display_list
                )

                matched_dislikes_display_list = [
                    f"{title.strip()} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})"
                    for _, id_, title in ok_dis
                ]
                st.session_state.dislikes_txt_content = "\n".join(
                    matched_dislikes_display_list
                )

            else:
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
                st.session_state.recommendation_output = {
                    "recs": recs,
                    "miss_like": miss_like,
                    "miss_dis": miss_dis,
                    "error": None,
                }

                matched_likes_display_list = [
                    f"{title.strip()} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})"
                    for _, id_, title in ok_like
                ]
                st.session_state.likes_txt_content = "\n".join(
                    matched_likes_display_list
                )

                matched_dislikes_display_list = [
                    f"{title.strip()} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})"
                    for _, id_, title in ok_dis
                ]
                st.session_state.dislikes_txt_content = "\n".join(
                    matched_dislikes_display_list
                )

                st.session_state.last_submitted_likes_recommend = (
                    current_likes_input_str
                )
                st.session_state.last_submitted_dislikes_recommend = (
                    current_dislikes_input_str
                )
                st.rerun()

if st.session_state.preview_output:
    output = st.session_state.preview_output
    if output.get("error"):
        st.info(output["error"])
    else:
        ok_like, miss_like = output["ok_like"], output["miss_like"]
        ok_dis, miss_dis = output["ok_dis"], output["miss_dis"]

        if (
            not ok_like
            and not miss_like
            and not ok_dis
            and not miss_dis
            and (
                st.session_state.last_submitted_likes_preview
                or st.session_state.last_submitted_dislikes_preview
            )
        ):
            st.info("No titles were recognized from your input for preview.")

        if ok_like:
            st.subheader("👍 We’ll search as Likes (Preview)")
            st.dataframe(
                [
                    {
                        "Input": i,
                        "Matched title": f"{t} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})",
                    }
                    for i, id_, t in ok_like
                ],
                use_container_width=True,
                hide_index=True,
            )
        if ok_dis:
            st.subheader("👎 We’ll avoid as Dislikes (Preview)")
            st.dataframe(
                [
                    {
                        "Input": i,
                        "Matched title": f"{t} ({META_BY_ID.get(id_, {}).get('year', 'N/A')})",
                    }
                    for i, id_, t in ok_dis
                ],
                use_container_width=True,
                hide_index=True,
            )
        if miss_like or miss_dis:
            st.subheader("⚠ Not recognised (Preview)")
            st.warning("\n".join(miss_like + miss_dis))


if st.session_state.recommendation_output:
    output = st.session_state.recommendation_output
    if output.get("error"):
        st.error(output["error"])
    else:
        recs = output["recs"]

        if recs:
            for rec_item in recs:
                imdb_id_from_url = (
                    rec_item.get("url", "").split("/title/")[-1].split("/")[0]
                )
                if not imdb_id_from_url.startswith("tt"):
                    imdb_id_display = "N/A"
                else:
                    imdb_id_display = imdb_id_from_url

                movie_title_display = rec_item.get("title", "N/A")
                movie_year_val = rec_item.get("year")
                movie_year_display = (
                    str(int(movie_year_val))
                    if pd.notna(movie_year_val) and str(movie_year_val).strip() != ""
                    else "N/A"
                )

                rating_display = rec_item.get("rating", "N/A")
                norm_rating_val = rec_item.get(
                    "normalized rating", rec_item.get("norm_rating", "N/A")
                )
                norm_rating_display = (
                    f"{norm_rating_val:.2f}"
                    if isinstance(norm_rating_val, float)
                    else norm_rating_val
                )

                score_val = rec_item.get("score", "N/A")
                score_display = (
                    f"{score_val:.3f}" if isinstance(score_val, float) else score_val
                )
                url_display = rec_item.get("url", "#")

                main_cols = st.columns([0.75, 0.125, 0.125])

                with main_cols[0]:
                    st.markdown(
                        f"""
                            <a href="{url_display}" target="_blank" style="font-size: 1.05em; font-weight: 500; text-decoration: none;">{movie_title_display} ({movie_year_display})</a>
                            <br>
                            <small>IMDb Rating: {rating_display}   |   Norm. Rating: {norm_rating_display}   |   Score: {score_display}</small>
                            """,
                        unsafe_allow_html=True,
                    )

                with main_cols[1]:
                    st.button(
                        "👍",
                        key=f"add_like_{imdb_id_display}",
                        help=f"Add '{movie_title_display}' to Likes",
                        use_container_width=True,
                    )

                with main_cols[2]:
                    st.button(
                        "👎",
                        key=f"add_dislike_{imdb_id_display}",
                        help=f"Add '{movie_title_display}' to Dislikes",
                        use_container_width=True,
                    )

                st.divider()
