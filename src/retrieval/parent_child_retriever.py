from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.dense_retriever import DenseRetriever
from src.utils.io import read_jsonl


class ParentChildRetriever:
    """PARENT-CHILD RETRIEVER THAT RETRIEVES CHILDREN AND RETURNS PARENTS. **"""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        dense_retriever: DenseRetriever,
    ):
        """INITIALIZE PARENT CHILD RETRIEVER. **"""
        self.experiment_name = experiment_name
        self.dense_retriever = dense_retriever

        parent_path = Path(output_dir) / experiment_name / "parent_chunks.jsonl"
        parents = read_jsonl(str(parent_path))

        self.parent_lookup = {
            parent["parent_id"]: parent
            for parent in parents
        }

    def retrieve(self, query: str, top_k: int = 3, fetch_k: int = 10) -> List[Dict[str, Any]]:
        """RETRIEVE CHILD CHUNKS AND EXPAND TO UNIQUE PARENT CHUNKS. **"""
        child_results = self.dense_retriever.retrieve(query=query, top_k=fetch_k)

        parent_results = []
        seen_parent_ids = set()

        for child in child_results:
            parent_id = child["metadata"].get("parent_id")

            if not parent_id or parent_id in seen_parent_ids:
                continue

            parent = self.parent_lookup.get(parent_id)

            if not parent:
                continue

            seen_parent_ids.add(parent_id)

            parent_results.append(
                {
                    "rank": len(parent_results) + 1,
                    "chunk_id": parent_id,
                    "chunk_text": parent["parent_text"],
                    "metadata": {
                        "doc_id": parent.get("doc_id", ""),
                        "title": parent.get("title", ""),
                        "chunk_type": "parent",
                        "parent_id": parent_id,
                    },
                    "score": child.get("score"),
                    "retrieval_strategy": "parent_child_dense",
                    "matched_child_id": child["chunk_id"],
                }
            )

            if len(parent_results) >= top_k:
                break

        return parent_results