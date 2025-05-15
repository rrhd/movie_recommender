#!/usr/bin/env python3
"""Build artifacts/tokens.pkl  –  imdb_id → {genre-tokens ∪ decade-token}"""

from pathlib import Path
import pickle, pandas as pd, numpy as np

ART  = Path(__file__).parent / "artifacts"
META = ART / "enriched_movies.pkl"
OUT  = ART / "tokens.pkl"

def _tokens(rec: dict) -> set[str]:
    # ── genre tokens ────────────────────────────────────────────────
    g = rec.get("genres")
    if not isinstance(g, (list, tuple)):
        g = []                      # NaN / None / str / float → empty
    toks = {str(x).lower() for x in g if pd.notna(x) and x != ""}

    # ── decade token ────────────────────────────────────────────────
    y = rec.get("year")
    try:
        toks.add(str(int(float(y)))[:3])   # 1994 → "199"
    except Exception:
        pass
    return toks

raw = pickle.load(open(META, "rb"))
df  = pd.DataFrame(raw) if isinstance(raw, list) else raw
df  = df.dropna(subset=["imdb_id"])

tok_map = {str(r["imdb_id"]): _tokens(r) for r in df.to_dict("records")}
OUT.write_bytes(pickle.dumps(tok_map))
print(f"[OK] wrote {len(tok_map):,} token sets → {OUT.relative_to(Path.cwd())}")
