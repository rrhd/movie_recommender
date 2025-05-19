#!/usr/bin/env python3
"""
update.py
-------------
Orchestrate updates of app data, title lookup, tokens, and Pinecone upsert
based on any new artifacts.
"""

import runpy
from pathlib import Path


def main():
    repo_root = Path(__file__).parent
    # 1. prepare Streamlit app data and app_tokens.pkl
    print("[STEP 1] prepare_app_data.py → app_data.pkl")
    runpy.run_path(str(repo_root / "prepare_app_data.py"), run_name="__main__")

    # 2. build artifacts/tokens.pkl
    print("[STEP 2] build_tokens.py → ./tokens.pkl")
    runpy.run_path(str(repo_root / "build_tokens.py"), run_name="__main__")

    # 3. title lookup index
    print(
        "[STEP 3] title_lookup.py --build → title_lookup.json, vectorizer, tfidf, maps"
    )
    import title_lookup

    title_lookup.build_lookup()

    # 4. push SVD vectors to Pinecone
    print("[STEP 4] push_svd_vectors_to_pinecone.py → Pinecone upsert")
    import push_svd_vectors_to_pinecone

    push_svd_vectors_to_pinecone.main()

    print("[DONE] all update steps completed.")


if __name__ == "__main__":
    main()
