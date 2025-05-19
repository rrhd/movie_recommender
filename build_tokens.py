"""Build artifacts/tokens.pkl  –  imdb_id → {genre-tokens ∪ decade-token}"""

from pathlib import Path
import pickle, pandas as pd, numpy as np
from typing import Any

ROOT_DIR  = Path(__file__).parent
ART = ROOT_DIR / "artifacts"
META = ART / "enriched_movies.pkl"
OUT = ROOT_DIR / "tokens.pkl"


def _tokens(m: dict[str, Any]) -> set[str]:
    toks: set[str] = set()
    # Genres
    genres = m.get("genres", [])
    if isinstance(genres, (list, tuple)):
        for g in genres: toks.add(f"g_{str(g).lower()}") # Prefix for clarity

    # Year/Decade
    year = m.get("year")
    if isinstance(year, (int, float, str)):
        try:
            y = int(year)
            toks.add(f"d_{str(y)[:3]}") # Decade
        except Exception: pass

    # Example: Actors (if available and useful)
    actors = m.get("actors", []) # Assuming 'actors' key exists
    if isinstance(actors, list):
        for actor in actors[:3]: # Limit to top N actors
            toks.add(f"a_{str(actor).lower().replace(' ', '_')}")

    # Example: Director
    director = m.get("director")
    if director and isinstance(director, str):
         toks.add(f"dir_{director.lower().replace(' ', '_')}")
    return toks


raw = pickle.load(open(META, "rb"))
df = pd.DataFrame(raw) if isinstance(raw, list) else raw
df = df.dropna(subset=["imdb_id"])

tok_map = {str(r["imdb_id"]): _tokens(r) for r in df.to_dict("records")}
OUT.write_bytes(pickle.dumps(tok_map))
print(f"[OK] wrote {len(tok_map):,} token sets → {OUT.relative_to(Path.cwd())}")
