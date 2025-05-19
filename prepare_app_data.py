"""
prepare_app_data.py – Pre-processes movie data for the Streamlit app.

Reads artifacts/enriched_movies.pkl and generates:
1. app_data.pkl: DataFrame with essential columns, processed lists for genres, etc.
                  (saved in the script's directory)
2. app_tokens.pkl: imdb_id -> {tokens} for the recommendation backend.
                   (saved in the script's directory)
"""

from pathlib import Path
import pickle
import pandas as pd
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


SCRIPT_DIR = Path(__file__).parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
META_PKL_SOURCE = ARTIFACTS_DIR / "enriched_movies.pkl"


APP_DATA_OUTPUT_PKL = SCRIPT_DIR / "app_data.pkl"


def main():
    logger.info(f"Starting data preparation. Source: {META_PKL_SOURCE}")

    if not META_PKL_SOURCE.exists():
        logger.error(f"CRITICAL: Source data file {META_PKL_SOURCE} not found.")
        return

    try:
        raw_data = pickle.load(open(META_PKL_SOURCE, "rb"))
    except Exception as e:
        logger.error(f"Failed to load {META_PKL_SOURCE}: {e}")
        return

    df_full = pd.DataFrame(raw_data) if isinstance(raw_data, list) else raw_data
    df_full = df_full.dropna(subset=["imdb_id", "title"])

    logger.info(f"Loaded {len(df_full)} records from source.")

    logger.info("Processing DataFrame for Streamlit app...")

    base_required_cols = ["imdb_id", "title"]
    desired_cols_for_meta = ["year", "rating", "url"]
    list_cols_to_process = ["genres", "countries", "languages"]

    missing_essential_cols = [
        col for col in base_required_cols if col not in df_full.columns
    ]
    if missing_essential_cols:
        logger.error(
            f"Essential columns {missing_essential_cols} missing in {META_PKL_SOURCE}. Aborting."
        )
        return

    cols_to_select = base_required_cols[:]
    for col in desired_cols_for_meta + list_cols_to_process:
        if col in df_full.columns and col not in cols_to_select:
            cols_to_select.append(col)
        elif col not in df_full.columns:
            logger.warning(
                f"Column '{col}' not found in source. It will be missing or default in app_data.pkl."
            )
            if col in list_cols_to_process:
                df_full[col] = [[] for _ in range(len(df_full))]
                if col not in cols_to_select:
                    cols_to_select.append(col)

    df_app = df_full[cols_to_select].copy()

    if "year" in df_app.columns:
        df_app["year"] = (
            pd.to_numeric(df_app["year"], errors="coerce").fillna(0).astype(int)
        )
    else:
        df_app["year"] = 0

    if "rating" in df_app.columns:
        df_app["rating"] = (
            pd.to_numeric(df_app["rating"], errors="coerce").fillna(0.0).astype(float)
        )
    else:
        df_app["rating"] = 0.0

    for col in list_cols_to_process:
        if col not in df_app.columns:
            df_app[col] = [[] for _ in range(len(df_app))]
        else:
            df_app[col] = df_app[col].apply(
                lambda x: [
                    str(item).strip()
                    for item in x
                    if pd.notna(item) and str(item).strip()
                ]
                if isinstance(x, list)
                else ([str(x).strip()] if pd.notna(x) and str(x).strip() else [])
            )

    try:
        df_app.to_pickle(APP_DATA_OUTPUT_PKL)
        logger.info(
            f"Successfully saved processed app data to {APP_DATA_OUTPUT_PKL} ({len(df_app)} records)"
        )
    except Exception as e:
        logger.error(f"Failed to save app data to {APP_DATA_OUTPUT_PKL}: {e}")
        return


    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
