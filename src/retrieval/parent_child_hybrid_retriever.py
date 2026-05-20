from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.io import read_jsonl


class ParentChildHybridRetriever:
    """PARENT-CHILD HYBRID RETRIEVER. **"""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        hybrid_retriever: HybridRetriever,
    ):
        self.experiment_name = experiment_name
        self.hybrid_retriever = hybrid_retriever

        parent_path = Path(output_dir) / experiment_name / "parent_chunks.jsonl"
        parents = read_jsonl(str(parent_path))

        self.parent_lookup = {
            parent["parent_id"]: parent
            for parent in parents
            if parent.get("parent_id")
        }

    def retrieve(
        self,
        query: str,
        fetch_k: int = 20,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """RETRIEVE CHILD CHUNKS, EXPAND TO PARENTS, RETURN TOP-K PARENTS. **"""

        child_results = self.hybrid_retriever.retrieve(
            query=query,
            fetch_k=fetch_k,
            top_k=fetch_k,
        )

        parent_results = []
        seen_parent_ids = set()

        for child in child_results:
            metadata = child.get("metadata", {})
            parent_id = metadata.get("parent_id")

            if not parent_id:
                continue

            if parent_id in seen_parent_ids:
                continue

            parent = self.parent_lookup.get(parent_id)

            if not parent:
                continue

            parent_text = parent.get("parent_text", "")

            if not parent_text.strip():
                continue

            seen_parent_ids.add(parent_id)

            parent_results.append(
                {
                    "chunk_id": parent_id,
                    "chunk_text": parent_text,
                    "metadata": {
                        "doc_id": parent.get("doc_id", metadata.get("doc_id", "")),
                        "title": parent.get("title", metadata.get("title", "")),
                        "page_number": parent.get("page_number", metadata.get("page_number")),
                        "parent_id": parent_id,
                        "modality": parent.get("modality", metadata.get("modality", "text")),
                        "table_id": parent.get("table_id", metadata.get("table_id", "")),
                        "figure_id": parent.get("figure_id", metadata.get("figure_id", "")),
                        "image_path": parent.get("image_path", metadata.get("image_path", "")),
                    },
                    "score": child.get("score", 0.0),
                    "retrieval_strategy": "parent_child_hybrid",
                    "matched_child_id": child.get("chunk_id", ""),
                }
            )

            if len(parent_results) >= top_k:
                break

        return parent_results