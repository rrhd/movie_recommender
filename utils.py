from __future__ import annotations

from typing import List, Tuple
from title_lookup import match_many, fuzzy_match_one




def resolve_lines(lines: List[str]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    Return (ok, miss) where ok = list[(typed, imdb_id, pretty)],
    miss = list[str]
    """
    matched, missed = match_many(lines)
    lc_map = {title.lower(): (imdb, title) for imdb, title in matched}

    ok, miss = [], []
    for raw in lines:
        l = raw.lower()
        if l in lc_map:
            imdb, pretty = lc_map[l]
            ok.append((raw, imdb, pretty))
        else:
            imdb, pretty, _ = fuzzy_match_one(raw)
            if imdb:
                ok.append((raw, imdb, pretty))
            else:
                miss.append(raw)
    return ok, missed + miss
