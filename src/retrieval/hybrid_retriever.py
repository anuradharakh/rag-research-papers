from typing import Any, Dict, List

from src.indexing.bm25_store import load_bm25_index, tokenize
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """HYBRID RETRIEVER USING DENSE SEARCH + BM25 + RRF. **"""

    def __init__(
        self,
        experiment_name: str,
        index_dir: str,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        """INITIALIZE HYBRID RETRIEVER. **"""
        self.experiment_name = experiment_name
        self.index_dir = index_dir
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

        payload = load_bm25_index(experiment_name, index_dir)
        self.bm25 = payload["bm25"]
        self.chunks = payload["chunks"]

    def retrieve(self, query: str, fetch_k: int = 20, top_k: int = 3) -> List[Dict[str, Any]]:
        """RETRIEVE TOP-K CHUNKS USING HYBRID SEARCH. **"""
        dense_results = self.dense_retriever.retrieve(query=query, top_k=fetch_k)

        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )[:fetch_k]

        bm25_results = []

        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]
            bm25_results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_text": chunk["chunk_text"],
                    "metadata": {
                        "doc_id": chunk.get("doc_id", ""),
                        "title": chunk.get("title", ""),
                        "chunk_type": chunk.get("chunk_type", ""),
                        "parent_id": chunk.get("parent_id") or "",
                    },
                    "score": float(bm25_scores[index]),
                    "retrieval_strategy": "bm25",
                }
            )

        return reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            rrf_k=self.rrf_k,
            top_k=top_k,
        )