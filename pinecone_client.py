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


def _load_secrets() -> dict:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "PINECONE_API_KEY" in st.secrets:
            return st.secrets
    except ModuleNotFoundError:
        pass
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    return toml.load(secrets_path)


_secrets = _load_secrets()
PINECONE_API_KEY = _secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = _secrets["PINECONE_INDEX_NAME"]
PINECONE_HOST = _secrets["PINECONE_HOST"]

pc = Pinecone(api_key=PINECONE_API_KEY)
desc = pc.describe_index(PINECONE_INDEX_NAME)
DIM = desc.dimension


INDEX = pc.Index(PINECONE_INDEX_NAME)
__all__ = ["INDEX", "DIM", "PINECONE_HOST", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
