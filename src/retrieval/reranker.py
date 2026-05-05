from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """CROSS-ENCODER RERANKER FOR QUERY-CHUNK PAIRS. **"""

    def __init__(self, model_name: str):
        """INITIALIZE CROSS-ENCODER MODEL. **"""
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """RERANK CHUNKS USING CROSS-ENCODER SCORES. **"""

        if not chunks:
            return []

        pairs = [
            [query, chunk["chunk_text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for chunk, score in zip(chunks, scores):
            updated_chunk = chunk.copy()
            updated_chunk["reranker_score"] = float(score)
            updated_chunk["retrieval_strategy"] = (
                updated_chunk.get("retrieval_strategy", "") + "+reranker"
            )
            reranked.append(updated_chunk)

        reranked = sorted(
            reranked,
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        for rank, chunk in enumerate(reranked[:top_n], start=1):
            chunk["rank"] = rank

        return reranked[:top_n]