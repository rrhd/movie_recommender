"""
backend.py  –  fast(er) Pinecone recall  ➜  re-score  ➜  diversify  ➜  filter

Key upgrades
────────────
• one *batched* fetch for all seed vectors
• ThreadPool for per-seed neighborhood queries
• vectors cached in-process (LRU) to avoid re-download
• **Maximal Marginal Relevance (MMR)** for result diversification
"""

import asyncio
import functools
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np
from cachetools import LRUCache, cached
from pinecone import PineconeException, PineconeAsyncio

from pinecone_client import INDEX, DIM, PINECONE_API_KEY, PINECONE_INDEX_NAME

SCRIPT_DIR_BE = Path(__file__).parent


RAW_K = 100
ALPHA = 1.0
BETA = 0.8
MMR_LAMBDA = 0.70
MMR_EMBEDDING_DIVERSITY_WEIGHT = (
    0.5  # 0 for pure token, 1 for pure embedding, 0.5 for equal mix
)
WORKERS = min(32, (os.cpu_count() or 8) * 2)

# For Centroid Scoring
GAMMA_CENTROID_LIKE = 0.15  # Weight for similarity to liked items' centroid
DELTA_CENTROID_DISLIKE = (
    0.10  # Weight for *dissimilarity* from disliked items' centroid
)

# For Pseudo Label Spreading/Propagation
PROPAGATION_ITERATIONS = 2  # Number of propagation iterations (0 to disable)
PROPAGATION_K_NEIGHBORS = 7  # Max neighbors for each candidate during propagation
PROPAGATION_ALPHA_MIX = (
    0.20  # Mixing factor for propagated score vs. original score in an iteration
)
PROPAGATION_HOP2_DAMPING = 0.6  # Damping factor for 2-hop influence

ARTIFACTS = Path(__file__).parent / "artifacts"
METADATA_PKL = ARTIFACTS / "enriched_movies.pkl"
TOK_PATH_RT = SCRIPT_DIR_BE / "tokens.pkl"
try:
    TOKENS: Dict[str, set[str]] = pickle.load(open(TOK_PATH_RT, "rb"))
    logging.info(f"Successfully loaded tokens from {TOK_PATH_RT.name}")
except FileNotFoundError:
    logging.warning(
        f"{TOK_PATH_RT.name} missing – MMR diversity will use on-the-fly tokenisation or be affected."
    )
    TOKENS = {}
except Exception as e:
    logging.error(
        f"Error loading {TOK_PATH_RT.name}: {e}. Falling back to empty tokens."
    )
    TOKENS = {}


log = logging.getLogger("backend")
if not log.handlers:
    logging.basicConfig(
        format="%(levelname)s:%(name)s:%(message)s",
        level=logging.INFO,
        force=True,
    )


def _robust_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalizes rows of a matrix. Rows with norm ~0 will become zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for zero-norm vectors; their normalized form is also zero.
    safe_norms = np.where(np.abs(norms) < 1e-9, 1.0, norms)
    normalized_matrix = matrix / safe_norms
    # Ensure zero vectors truly are zero after division if norm was ~0
    normalized_matrix[(np.abs(norms) < 1e-9).squeeze()] = 0.0
    return normalized_matrix


def _vectorized_cosine_similarity_square(matrix: np.ndarray) -> np.ndarray:
    """Computes the cosine similarity matrix for all row pairs in a single matrix."""
    # Input: matrix (M, D)
    # Output: similarity_matrix (M, M)
    if matrix.ndim == 1:  # Handle single vector case, making it (1,D)
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] == 0:
        return np.array([])

    normalized_matrix = _robust_normalize_rows(matrix)
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    return np.clip(similarity_matrix, -1.0, 1.0)  # Clip for numerical stability


def _vectorized_cosine_similarity_pairwise(
    matrix1: np.ndarray, matrix2: np.ndarray
) -> np.ndarray:
    """Computes cosine similarity between all rows of matrix1 and all rows of matrix2."""
    # Input: matrix1 (M, D), matrix2 (N, D)
    # Output: similarity_matrix (M, N)
    if matrix1.ndim == 1:
        matrix1 = matrix1.reshape(1, -1)
    if matrix2.ndim == 1:
        matrix2 = matrix2.reshape(1, -1)

    if matrix1.shape[0] == 0 or matrix2.shape[0] == 0:
        return np.array([]).reshape(matrix1.shape[0], matrix2.shape[0])

    normalized_matrix1 = _robust_normalize_rows(matrix1)
    normalized_matrix2 = _robust_normalize_rows(matrix2)
    similarity_matrix = np.dot(normalized_matrix1, normalized_matrix2.T)
    return np.clip(similarity_matrix, -1.0, 1.0)


def _fix_lists(rec: dict) -> dict:
    for key in ("genres", "countries", "languages"):
        v = rec.get(key, [])
        if not isinstance(v, list):
            rec[key] = []
    return rec


def _pseudo_label_propagation(
    current_scores: Dict[str, float],
    candidate_vectors: Dict[str, np.ndarray],
    iterations: int,
    k_neighbors: int,
    alpha_mix: float,
    hop2_damping: float,
    logger: logging.Logger,
) -> Dict[str, float]:
    if (
        not candidate_vectors
        or not current_scores
        or iterations == 0
        or k_neighbors == 0
    ):
        logger.info(
            "Skipping pseudo-label propagation (no iterations, neighbors, data, or scores)."
        )
        return current_scores

    logger.info(
        f"Starting pseudo-label propagation: {iterations} iterations, "
        f"k_neighbors={k_neighbors}, alpha_mix={alpha_mix}, hop2_damping={hop2_damping}."
    )

    propagated_scores = current_scores.copy()

    # Filter for IDs that have both scores and vectors
    valid_ids_for_propagation = [
        cid
        for cid in current_scores.keys()
        if cid in candidate_vectors and candidate_vectors[cid] is not None
    ]

    if len(valid_ids_for_propagation) < 2:
        logger.info("Not enough valid candidates for propagation.")
        return propagated_scores

    # Prepare data for vectorized similarity calculation
    # Ensure consistent ordering for matrix operations
    ordered_cids_prop = valid_ids_for_propagation
    # Stack vectors into a matrix
    try:
        prop_vectors_matrix = np.stack(
            [candidate_vectors[cid] for cid in ordered_cids_prop]
        )
    except (
        ValueError
    ) as e:  # Handles cases with inconsistent vector dimensions or empty list
        logger.error(
            f"Error stacking vectors for propagation: {e}. Skipping propagation."
        )
        return propagated_scores

    local_neighborhoods: dict[str, list[tuple[str, float]]] = defaultdict(list)
    logger.info(
        f"Building local neighborhoods for {len(ordered_cids_prop)} candidates..."
    )

    if prop_vectors_matrix.size > 0:  # Ensure matrix is not empty
        # Calculate all-pairs cosine similarity in a vectorized way
        similarity_matrix = _vectorized_cosine_similarity_square(
            prop_vectors_matrix
        )  # Uses new helper

        for i, cid1 in enumerate(ordered_cids_prop):
            # Similarities of cid1 to all other candidates
            sims_to_others = similarity_matrix[i, :]

            # Exclude self-similarity (can be done by setting similarity_matrix[i, i] = -1 before processing)
            # Or, more robustly during neighbor selection:

            neighbor_indices_sorted = np.argsort(sims_to_others)[
                ::-1
            ]  # Sort descending

            neighbor_sims_tuples = []
            for j_idx in neighbor_indices_sorted:
                if i == j_idx:  # Skip self
                    continue
                sim_val = sims_to_others[j_idx]
                if sim_val > 0.05:  # Apply threshold
                    neighbor_sims_tuples.append((ordered_cids_prop[j_idx], sim_val))
                if len(neighbor_sims_tuples) >= k_neighbors:  # Found enough neighbors
                    break
            local_neighborhoods[cid1] = neighbor_sims_tuples
    else:
        logger.info("No valid vectors to build neighborhoods for propagation.")

    logger.info("Local neighborhoods built. Starting propagation iterations...")
    # Propagation loop (remains structurally similar as it iterates over sparse neighborhoods)
    # Minor optimization: directly use valid_ids_for_propagation which is same as ordered_cids_prop
    for iteration in range(iterations):
        scores_at_iter_start = propagated_scores.copy()
        updates_count = 0
        for cid in ordered_cids_prop:  # Iterate using the ordered list
            original_score_cid = scores_at_iter_start.get(cid, 0.0)

            influence_1hop = 0.0
            weight_sum_1hop = 0.0
            if cid in local_neighborhoods:
                for neighbor_id, sim1 in local_neighborhoods[cid]:
                    neighbor_score = scores_at_iter_start.get(neighbor_id)
                    if neighbor_score is not None:
                        influence_1hop += neighbor_score * sim1
                        weight_sum_1hop += sim1

            influence_2hop = 0.0
            weight_sum_2hop = 0.0
            processed_2hop_neighbors = {cid}
            if cid in local_neighborhoods:
                for n_id, _ in local_neighborhoods[cid]:
                    processed_2hop_neighbors.add(n_id)

                for neighbor_id, sim1 in local_neighborhoods[cid]:
                    if neighbor_id in local_neighborhoods:
                        for second_hop_id, sim2 in local_neighborhoods[neighbor_id]:
                            if second_hop_id in processed_2hop_neighbors:
                                continue
                            second_hop_score = scores_at_iter_start.get(second_hop_id)
                            if second_hop_score is not None:
                                path_similarity = sim1 * sim2 * hop2_damping
                                influence_2hop += second_hop_score * path_similarity
                                weight_sum_2hop += path_similarity
                                processed_2hop_neighbors.add(second_hop_id)

            total_influence = influence_1hop + influence_2hop
            total_weight_sum = weight_sum_1hop + weight_sum_2hop

            if total_weight_sum > 1e-9:
                propagated_value = total_influence / total_weight_sum
                new_score = (
                    1 - alpha_mix
                ) * original_score_cid + alpha_mix * propagated_value
                if new_score != propagated_scores[cid]:  # Update only if changed
                    propagated_scores[cid] = new_score
                    updates_count += 1
        logger.debug(
            f"Propagation iteration {iteration + 1}/{iterations} completed. Scores updated for {updates_count} candidates."
        )
        if (
            updates_count == 0 and iteration > 0
        ):  # Optimization: early stop if no scores change
            logger.info(
                f"Stopping propagation early at iteration {iteration + 1} as no scores changed."
            )
            break

    logger.info("Pseudo-label propagation finished.")
    return propagated_scores


_VEC_CACHE: LRUCache[str, np.ndarray] = LRUCache(maxsize=250_000)


@cached(_VEC_CACHE)
def _fetch_vec_cached(imdb: str) -> np.ndarray:
    """Fetch **once** – further calls served from RAM."""
    try:
        v = INDEX.fetch(ids=[imdb]).vectors[imdb].values
        return np.asarray(v, dtype=np.float32)
    except PineconeException as e:
        log.warning("failed to fetch %s – %s", imdb, e)
        return np.zeros(DIM, dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


def _query_seed(
    seed_id: str, disliked: set[str]
) -> list[tuple[str, float, list, dict]]:
    res = INDEX.query(
        id=seed_id,
        top_k=RAW_K,
        include_values=True,
        include_metadata=True,
    )
    out = []
    for m in res.matches:
        if m.id == seed_id or m.id in disliked:
            continue
        out.append((m.id, m.score, m.values, m.metadata))
    return out


async def _recall_parallel_async(
    liked: list[str],
    disliked: list[str],
) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, list[float]]]:
    seeds = liked + disliked
    if not seeds:
        return {}, {}, defaultdict(list)

    cand_meta: dict[str, dict] = {}
    cand_vec: dict[str, np.ndarray] = {}
    sim_likes: dict[str, list[float]] = defaultdict(list)
    dneg = set(disliked)

    async with PineconeAsyncio(api_key=PINECONE_API_KEY) as pc:
        desc = await pc.describe_index(PINECONE_INDEX_NAME)
        async with pc.IndexAsyncio(host=desc.host) as idx:
            sem = asyncio.Semaphore(WORKERS)

            async def _sem_query(seed_id: str):
                async with sem:
                    return await idx.query(
                        id=seed_id,
                        top_k=RAW_K,
                        include_values=True,
                        include_metadata=True,
                    )

            responses = await asyncio.gather(*(_sem_query(s) for s in seeds))

    for seed_id, resp in zip(seeds, responses):
        is_like = seed_id in liked
        for m in resp.matches:
            if m.id == seed_id or m.id in dneg:
                continue
            cid = m.id
            cand_vec.setdefault(cid, np.asarray(m.values, dtype=np.float32))
            cand_meta.setdefault(cid, m.metadata or {})
            if is_like:
                sim_likes[cid].append(m.score)

    return cand_meta, cand_vec, sim_likes


def _recall_parallel(
    liked: list[str],
    disliked: list[str],
) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, list[float]]]:
    """Sync wrapper around the async recall function."""
    return asyncio.run(_recall_parallel_async(liked, disliked))


def _sim_meta(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient between two token sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def _mmr_ranking(
    scores: Dict[str, float],
    k: int,
    lam: float,
) -> List[str]:
    """
    Maximal Marginal Relevance: select k items balancing relevance (scores)
    against diversity (meta-token overlap), with trade-off λ.
    """
    selected: List[str] = []
    candidates = set(scores)

    toks = {cid: TOKENS.get(cid, set()) for cid in candidates}

    while candidates and len(selected) < k:
        best_c, best_val = None, None
        for cid in candidates:
            rel = scores[cid]
            div = max((_sim_meta(toks[cid], toks[s]) for s in selected), default=0.0)
            mmr_score = lam * rel - (1 - lam) * div
            if best_val is None or mmr_score > best_val:
                best_c, best_val = cid, mmr_score
        if best_c is None:
            break
        selected.append(best_c)
        candidates.remove(best_c)

    return selected


def _freeze(v):
    if isinstance(v, list):
        return tuple(v)
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    return v


@functools.lru_cache(maxsize=2048)
def _cached_recommend(likes_key, dislikes_key, filt_key):
    return _recommend_inner(list(likes_key), list(dislikes_key), **dict(filt_key))


def recommend(liked_ids: List[str], disliked_ids: List[str], **filters):
    if not liked_ids and not disliked_ids:
        return []
    filt_key = tuple(sorted((k, _freeze(v)) for k, v in filters.items()))
    recs = _cached_recommend(
        tuple(sorted(liked_ids)), tuple(sorted(disliked_ids)), filt_key
    )
    # exclude any that the user already likes
    liked_set = set(liked_ids)
    recs = [
        r for r in recs if r["url"].split("/title/")[-1].split("/")[0] not in liked_set
    ]
    return recs


def _recommend_inner(
    liked_ids: List[str],
    disliked_ids: List[str],
    *,
    top_k: int = 15,
    min_year: int = None,  # Add other filters here as in original
    min_rating: float = None,
    min_norm: float = None,
    min_votes: int = None,
    include_genres: List[str] = None,
    exclude_genres: List[str] = None,
    include_countries: List[str] = None,
    exclude_countries: List[str] = None,
    include_languages: List[str] = None,
    exclude_languages: List[str] = None,
) -> List[Dict[str, Any]]:
    log.info(
        f"Recommendation request: {len(liked_ids)} likes, {len(disliked_ids)} dislikes, filters: {{...}}"
    )

    n_like, n_dis = len(liked_ids), len(disliked_ids)
    alpha = ALPHA / max(1.0, np.log1p(n_like))
    beta = 0.0 if n_dis == 0 else BETA / np.sqrt(n_dis)
    pop_w = 0.05 if (n_like or n_dis) else 1.0

    cand_meta, cand_vec, sim_likes = _recall_parallel(liked_ids, disliked_ids)

    if disliked_ids:
        for d_id in disliked_ids:
            _fetch_vec_cached(d_id)  # Prime cache

    cand_ids_from_recall = set(cand_vec.keys())
    # Filter out candidates that are liked or disliked, or have no vector
    cand_ids_set = {
        cid
        for cid in (cand_ids_from_recall - set(liked_ids) - set(disliked_ids))
        if cand_vec.get(cid) is not None
    }

    if not cand_ids_set:
        log.info("No candidates after recall, initial filtering, or missing vectors.")
        return []

    # Convert to list for ordered operations and matrix construction
    ordered_cand_ids = list(cand_ids_set)
    log.info(f"Initial {len(ordered_cand_ids)} candidates for scoring.")

    # Stack candidate vectors into a matrix
    try:
        cand_matrix = np.stack([cand_vec[cid] for cid in ordered_cand_ids])
    except (
        ValueError,
        KeyError,
    ) as e:  # Catch issues if a cid in ordered_cand_ids is somehow not in cand_vec or vectors are malformed
        log.error(
            f"Error stacking candidate vectors: {e}. Aborting scoring for some candidates or all."
        )
        # Filter ordered_cand_ids and rebuild matrix, or return [] if too many errors.
        # For simplicity here, if stacking fails catastrophically, we might return empty.
        # A more robust way would be to filter ordered_cand_ids.
        # Assuming cand_vec[cid] is not None due to earlier check, this is mostly for dimensional consistency.
        if not ordered_cand_ids:
            return []  # if the list became empty
        # Fallback or error:
        log.info(
            "Proceeding with candidates for whom vector stacking was successful (if partial failure handling implemented)."
        )
        # This example assumes if stack fails, it's critical.
        return []

    log.info("Starting initial scoring with centroid components...")
    liked_centroid_vec: np.ndarray | None = None
    if liked_ids:
        liked_item_vecs = [
            v
            for i in liked_ids
            if (v := _fetch_vec_cached(i)) is not None and np.any(v)
        ]
        if liked_item_vecs:
            liked_centroid_vec = np.mean(np.stack(liked_item_vecs), axis=0)

    disliked_centroid_vec: np.ndarray | None = None
    if disliked_ids:
        disliked_item_vecs = [
            v
            for i in disliked_ids
            if (v := _fetch_vec_cached(i)) is not None and np.any(v)
        ]
        if disliked_item_vecs:
            disliked_centroid_vec = np.mean(np.stack(disliked_item_vecs), axis=0)

    # --- Vectorized Scoring Calculations ---
    num_candidates = len(ordered_cand_ids)
    scores_array = np.zeros(num_candidates)  # Initialize scores array

    # Positive similarities (already computed per candidate)
    pos_sim_array = np.array(
        [
            float(np.mean(sim_likes[cid])) if sim_likes.get(cid) else 0.0
            for cid in ordered_cand_ids
        ]
    )
    scores_array += alpha * pos_sim_array

    # Negative similarities (vectorized)
    if disliked_ids and cand_matrix.size > 0:
        dis_item_vectors_list = [
            v
            for i in disliked_ids
            if (v := _fetch_vec_cached(i)) is not None and np.any(v)
        ]
        if dis_item_vectors_list:
            dis_matrix = np.stack(dis_item_vectors_list)
            # Shape: (num_candidates, num_disliked_items)
            neg_sim_matrix_pairwise = _vectorized_cosine_similarity_pairwise(
                cand_matrix, dis_matrix
            )
            # Mean similarity to disliked items for each candidate
            # Ensure neg_sim_matrix_pairwise is not empty before mean
            if neg_sim_matrix_pairwise.size > 0:
                mean_neg_sim_array = np.mean(neg_sim_matrix_pairwise, axis=1)
                scores_array -= beta * mean_neg_sim_array

    # Base popularity
    base_popularity_array = np.array(
        [cand_meta.get(cid, {}).get("norm_rating", 0.0) for cid in ordered_cand_ids]
    )
    scores_array += pop_w * base_popularity_array

    # Centroid similarities (vectorized)
    if liked_centroid_vec is not None and cand_matrix.size > 0:
        sim_to_liked_centroid_array = _vectorized_cosine_similarity_pairwise(
            cand_matrix, liked_centroid_vec.reshape(1, -1)
        ).flatten()
        scores_array += GAMMA_CENTROID_LIKE * sim_to_liked_centroid_array

    if disliked_centroid_vec is not None and cand_matrix.size > 0:
        sim_to_disliked_centroid_array = _vectorized_cosine_similarity_pairwise(
            cand_matrix, disliked_centroid_vec.reshape(1, -1)
        ).flatten()
        scores_array -= DELTA_CENTROID_DISLIKE * sim_to_disliked_centroid_array

    # Convert scores array back to dictionary
    scores: Dict[str, float] = {
        cid: scores_array[i] for i, cid in enumerate(ordered_cand_ids)
    }
    log.info(f"Initial scoring completed for {len(scores)} candidates.")

    # --- 2. Pseudo Label Propagation ---
    if PROPAGATION_ITERATIONS > 0 and PROPAGATION_K_NEIGHBORS > 0 and len(scores) > 1:
        # Pass only vectors of items that actually have scores
        candidate_vectors_for_propagation = {
            cid: cand_vec[cid]
            for cid in scores.keys()
            if cid in cand_vec and cand_vec[cid] is not None
        }
        if (
            candidate_vectors_for_propagation
        ):  # Ensure there are vectors to propagate with
            scores = _pseudo_label_propagation(  # This calls the modified _pseudo_label_propagation
                scores,
                candidate_vectors_for_propagation,
                PROPAGATION_ITERATIONS,
                PROPAGATION_K_NEIGHBORS,
                PROPAGATION_ALPHA_MIX,
                PROPAGATION_HOP2_DAMPING,
                log,
            )
            log.info("Pseudo-label propagation applied.")
        else:
            log.info(
                "Skipped propagation as no candidate vectors were available for scored items."
            )
    else:
        log.info(
            "Skipping pseudo-label propagation based on settings or candidate count."
        )

    # --- 3. Filtering ---
    def _keep(mid: str) -> bool:
        m = cand_meta.get(mid, {})
        if not m and mid in scores:
            log.warning(
                f"Metadata missing for candidate {mid}, will likely be filtered out by criteria."
            )
        if min_year and (m.get("year") or 0) < min_year:
            return False
        if min_rating and (m.get("rating") or 0) < min_rating:
            return False
        if min_norm and (m.get("norm_rating") or 0) < min_norm:
            return False
        if min_votes and (m.get("votes") or 0) < min_votes:
            return False
        meta_genres = m.get("genres", [])
        if not isinstance(meta_genres, list):
            meta_genres = []
        meta_countries = m.get("countries", [])
        if not isinstance(meta_countries, list):
            meta_countries = []
        meta_languages = m.get("languages", [])
        if not isinstance(meta_languages, list):
            meta_languages = []
        if include_genres and not set(meta_genres).intersection(include_genres):
            return False
        if include_countries and not set(meta_countries).intersection(
            include_countries
        ):
            return False
        if include_languages and not set(meta_languages).intersection(
            include_languages
        ):
            return False
        if exclude_genres and set(meta_genres).intersection(exclude_genres):
            return False
        if exclude_countries and set(meta_countries).intersection(exclude_countries):
            return False
        if exclude_languages and set(meta_languages).intersection(exclude_languages):
            return False
        return True

    ids_passing_filters = [cid for cid in scores.keys() if _keep(cid)]
    if not ids_passing_filters:
        log.info("No candidates remaining after filtering.")
        return []
    log.info(f"{len(ids_passing_filters)} candidates remaining after filtering.")
    scores_for_mmr = {mid: scores[mid] for mid in ids_passing_filters}

    def _mmr_ranking_inner(
        scores_subset: Dict[str, float],
        k_mmr: int,
        lambda_mmr: float,
    ) -> List[str]:
        selected_ids: List[str] = []
        candidate_pool_ids = set(scores_subset.keys())
        item_tokens = {cid: TOKENS.get(cid, set()) for cid in candidate_pool_ids}
        nonlocal cand_vec  # Assuming cand_vec is available from the outer scope

        log.debug(
            f"MMR starting with {len(candidate_pool_ids)} candidates for top {k_mmr}."
        )

        # Pre-fetch vectors for selected items to optimize inner loop
        selected_vectors_map: Dict[str, np.ndarray] = {}

        while candidate_pool_ids and len(selected_ids) < k_mmr:
            best_candidate_id, best_mmr_score = None, -float("inf")

            # Prepare selected vectors matrix for efficient computation if any are selected
            # This matrix will be (num_selected, D)
            # We rebuild it in each outer loop if new items are added, or cache/update.
            # For simplicity, we can rebuild it or update it. Let's try updating for efficiency.

            # Efficiently get vectors for currently selected items
            # This list will grow with each selected item
            current_selected_vecs_list = [
                selected_vectors_map[s_id]
                for s_id in selected_ids
                if s_id in selected_vectors_map
                and selected_vectors_map[s_id] is not None
            ]
            selected_matrix = None
            if current_selected_vecs_list:
                try:
                    selected_matrix = np.stack(current_selected_vecs_list)
                except (
                    ValueError
                ):  # Should not happen if vectors are consistently shaped
                    log.warning("MMR: Inconsistent vector shapes for selected items.")
                    selected_matrix = None

            for cid in candidate_pool_ids:
                relevance_score = scores_subset[cid]
                token_diversity_penalty = 0.0
                embedding_diversity_penalty = 0.0

                current_cand_vector = cand_vec.get(cid)

                if selected_ids:
                    # Token-based diversity (remains the same, as it's not easily vectorized with current _sim_meta)
                    if item_tokens.get(cid):
                        token_diversity_penalty = max(
                            (
                                _sim_meta(
                                    item_tokens[cid], item_tokens.get(s_id, set())
                                )  # Use .get for safety
                                for s_id in selected_ids
                            ),
                            default=0.0,
                        )

                    # Embedding-based diversity penalty (vectorized)
                    if (
                        current_cand_vector is not None
                        and selected_matrix is not None
                        and selected_matrix.size > 0
                    ):
                        # Calculate cosine similarities between current_cand_vector and all in selected_matrix
                        # _vectorized_cosine_similarity_pairwise expects two matrices.
                        # current_cand_vector (D,) -> reshape to (1, D)
                        # selected_matrix (num_selected, D)
                        sims_to_selected = (
                            _vectorized_cosine_similarity_pairwise(
                                current_cand_vector.reshape(1, -1), selected_matrix
                            ).flatten()
                        )  # Result is (1, num_selected), flatten to (num_selected,)

                        if sims_to_selected.size > 0:
                            embedding_diversity_penalty = np.max(sims_to_selected)
                        else:  # Should not happen if selected_matrix was not None and not empty
                            embedding_diversity_penalty = 0.0

                combined_diversity_penalty = (
                    MMR_EMBEDDING_DIVERSITY_WEIGHT * embedding_diversity_penalty
                    + (1 - MMR_EMBEDDING_DIVERSITY_WEIGHT) * token_diversity_penalty
                )
                current_mmr_value = (
                    lambda_mmr * relevance_score
                    - (1 - lambda_mmr) * combined_diversity_penalty
                )

                if current_mmr_value > best_mmr_score:
                    best_candidate_id, best_mmr_score = cid, current_mmr_value

            if best_candidate_id is None:
                log.debug("MMR: No best candidate found in this iteration, stopping.")
                break

            selected_ids.append(best_candidate_id)
            candidate_pool_ids.remove(best_candidate_id)

            # Cache the vector of the newly selected item
            if (
                best_candidate_id not in selected_vectors_map
                and best_candidate_id in cand_vec
            ):
                selected_vectors_map[best_candidate_id] = cand_vec[best_candidate_id]

            log.debug(
                f"MMR selected: {best_candidate_id} (score: {best_mmr_score:.4f}). {len(selected_ids)}/{k_mmr} selected."
            )
        return selected_ids

    # --- 4. MMR with Combined Diversity (Inner Function) ---
    top_diverse_ids = _mmr_ranking_inner(scores_for_mmr, top_k, MMR_LAMBDA)
    log.info(f"MMR ranking finished, {len(top_diverse_ids)} diverse recommendations.")

    # --- 5. Prepare Output ---
    output_recs: List[Dict[str, Any]] = []
    for mid in top_diverse_ids:
        m_meta = cand_meta.get(mid, {})
        output_recs.append(
            {
                "title": m_meta.get("title", "N/A"),
                "year": m_meta.get("year", ""),
                "rating": m_meta.get("rating", ""),
                "norm_rating": m_meta.get("norm_rating", ""),
                "score": round(scores.get(mid, 0.0), 4),
                "url": f"https://www.imdb.com/title/{mid}/",
            }
        )
    return output_recs


def main():
    liked_ids = ["tt1234567", "tt2345678"]
    disliked_ids = ["tt3456789"]
    filters = {
        "min_year": 2000,
        "min_rating": 7.0,
        "include_genres": ["Action", "Drama"],
        "exclude_genres": ["Horror"],
    }
    recommendations = recommend(liked_ids, disliked_ids, **filters)
    for rec in recommendations:
        print(rec)
