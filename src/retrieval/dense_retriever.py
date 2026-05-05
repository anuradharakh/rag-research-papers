from pathlib import Path
from typing import Any, Dict, List

import chromadb

from src.indexing.embedding import EmbeddingModel


class DenseRetriever:
    """DENSE VECTOR RETRIEVER USING CHROMA. **"""

    def __init__(
        self,
        experiment_name: str,
        index_dir: str,
        embedding_model: EmbeddingModel,
    ):
        """INITIALIZE DENSE RETRIEVER. **"""
        self.experiment_name = experiment_name
        self.index_dir = str(Path(index_dir) / experiment_name)
        self.embedding_model = embedding_model

        self.client = chromadb.PersistentClient(path=self.index_dir)
        self.collection = self.client.get_collection(
            name=experiment_name.replace("-", "_")
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """RETRIEVE TOP-K CHUNKS USING DENSE VECTOR SEARCH. **"""
        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        for rank, chunk_id in enumerate(results["ids"][0], start=1):
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "chunk_text": results["documents"][0][rank - 1],
                    "metadata": results["metadatas"][0][rank - 1],
                    "score": results["distances"][0][rank - 1],
                    "retrieval_strategy": "dense",
                }
            )

        return retrieved_chunks