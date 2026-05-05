from typing import Any, Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """COMBINE RANKED LISTS USING RECIPROCAL RANK FUSION. **"""
    fused_scores = {}
    chunk_lookup = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            chunk_lookup[chunk_id] = chunk

    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    fused_results = []

    for rank, chunk_id in enumerate(sorted_chunk_ids[:top_k], start=1):
        chunk = chunk_lookup[chunk_id].copy()
        chunk["rank"] = rank
        chunk["score"] = fused_scores[chunk_id]
        chunk["retrieval_strategy"] = "hybrid_rrf"
        fused_results.append(chunk)

    return fused_results