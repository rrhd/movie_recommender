"""
pinecone_client.py
------------------
Tiny helper that exposes a *singleton* Pinecone Index object.

Usage:
    from pinecone_client import INDEX, DIM
"""

from pathlib import Path
import toml
from pinecone import Pinecone


# ---------------------------------------------------------------------
# read secrets (streamlit passes them at runtime, but we fall back to file)
# ---------------------------------------------------------------------
def _load_secrets() -> dict:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "PINECONE_API_KEY" in st.secrets:
            return st.secrets
    except ModuleNotFoundError:
        pass  # running outside Streamlit – fall back
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    return toml.load(secrets_path)


_secrets = _load_secrets()
PINECONE_API_KEY = _secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = _secrets["PINECONE_INDEX_NAME"]
PINECONE_HOST = _secrets["PINECONE_HOST"]

pc = Pinecone(api_key=PINECONE_API_KEY)
desc = pc.describe_index(PINECONE_INDEX_NAME)  # <-- fails hard if wrong name/key
DIM = desc.dimension

# instantiate *once* – keeps underlying HTTP pool alive
INDEX = pc.Index(PINECONE_INDEX_NAME)
__all__ = ["INDEX", "DIM", "PINECONE_HOST", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
