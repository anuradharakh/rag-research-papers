from typing import Any, Dict, List

from src.indexing.bm25_store import load_bm25_index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion


def _get_chunk_text(chunk: Dict[str, Any]) -> str:
    """GET CHUNK TEXT FROM SUPPORTED CHUNK FORMATS. **"""
    return (
        chunk.get("chunk_text")
        or chunk.get("parent_text")
        or chunk.get("text")
        or ""
    )


def _get_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """GET METADATA FROM SUPPORTED CHUNK FORMATS. **"""
    metadata = chunk.get("metadata", {})

    return {
        "doc_id": metadata.get("doc_id", chunk.get("doc_id", "")),
        "title": metadata.get("title", chunk.get("title", "")),
        "chunk_type": metadata.get("chunk_type", chunk.get("chunk_type", "")),
        "parent_id": metadata.get("parent_id", chunk.get("parent_id", "")),
        "page_number": metadata.get("page_number", chunk.get("page_number", "")),
        "modality": metadata.get("modality", chunk.get("modality", "text")),
        "table_id": metadata.get("table_id", chunk.get("table_id", "")),
        "figure_id": metadata.get("figure_id", chunk.get("figure_id", "")),
        "image_path": metadata.get("image_path", chunk.get("image_path", "")),
    }


class HybridRetriever:
    """HYBRID BM25 + DENSE RETRIEVER USING RRF. **"""

    def __init__(
        self,
        experiment_name: str,
        index_dir: str,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        self.experiment_name = experiment_name
        self.index_dir = index_dir
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

        payload = load_bm25_index(
            experiment_name=experiment_name,
            index_dir=index_dir,
        )

        self.bm25 = payload["bm25"]
        self.chunks = payload["chunks"]
        self.tokenized_corpus = payload.get("tokenized_corpus", [])

    def tokenize(self, text: str) -> List[str]:
        """TOKENIZE TEXT FOR BM25. **"""
        return text.lower().split()

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """RETRIEVE TOP-K CHUNKS USING BM25. **"""
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]
            chunk_text = _get_chunk_text(chunk)
            metadata = _get_chunk_metadata(chunk)

            if not chunk_text.strip():
                continue

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id") or chunk.get("parent_id", ""),
                    "chunk_text": chunk_text,
                    "metadata": metadata,
                    "score": float(scores[index]),
                    "retrieval_strategy": "bm25",
                }
            )

        return results

    def retrieve(
        self,
        query: str,
        fetch_k: int = 20,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """RETRIEVE USING DENSE + BM25 AND FUSE WITH RRF. **"""
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=fetch_k,
        )

        bm25_results = self.retrieve_bm25(
            query=query,
            top_k=fetch_k,
        )

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            rrf_k=self.rrf_k,
            top_k=top_k,
        )

        for result in fused_results:
            result["retrieval_strategy"] = "hybrid_rrf"

        return fused_results