import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    """LOAD JSON FILE. **"""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_doc_id(doc_id: str) -> str:
    """NORMALIZE DOCUMENT ID FOR MATCHING. **"""
    return str(doc_id).replace(".pdf", "").strip()


def compute_hit_rate_at_k(
    retrieval_results: List[Dict[str, Any]],
    qrels: Dict[str, Any],
    k: int = 3,
) -> Dict[str, Any]:
    """COMPUTE HIT RATE AT K USING QUERY RESULTS AND QRELS. **"""
    total = 0
    hits = 0
    per_query = []

    for result in retrieval_results:
        query_id = result["query_id"]
        retrieved_chunks = result["retrieved_chunks"][:k]

        if query_id not in qrels:
            continue

        expected_docs = qrels[query_id]

        if isinstance(expected_docs, dict):
            if "doc_id" in expected_docs:
                expected_doc_ids = [normalize_doc_id(str(expected_docs["doc_id"]))]
            else:
                expected_doc_ids = [
                    normalize_doc_id(str(doc_id))
                    for doc_id in expected_docs.keys()
                ]

        elif isinstance(expected_docs, list):
            expected_doc_ids = [
                normalize_doc_id(str(item["doc_id"]) if isinstance(item, dict) else str(item))
                for item in expected_docs
            ]

        else:
            expected_doc_ids = [normalize_doc_id(str(expected_docs))]

        retrieved_doc_ids = [
            normalize_doc_id(chunk["metadata"].get("doc_id", ""))
            for chunk in retrieved_chunks
        ]

        is_hit = any(doc_id in retrieved_doc_ids for doc_id in expected_doc_ids)

        total += 1
        hits += int(is_hit)

        per_query.append(
            {
                "query_id": query_id,
                "question": result["question"],
                "expected_doc_ids": expected_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "hit": is_hit,
            }
        )

    hit_rate = hits / total if total else 0.0

    return {
        "k": k,
        "total_queries": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "per_query": per_query,
    }