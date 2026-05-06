import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    """LOAD JSON FILE. **"""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str) -> None:
    """SAVE JSON FILE. **"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def find_failed_retrievals(metrics_path: str) -> List[Dict[str, Any]]:
    """RETURN QUERIES WHERE HIT RATE FAILED. **"""
    metrics = load_json(metrics_path)
    return [item for item in metrics.get("per_query", []) if not item.get("hit", False)]


def attach_generation_outputs(
    failed_queries: List[Dict[str, Any]],
    generated_answers_path: str,
) -> List[Dict[str, Any]]:
    """ATTACH GENERATED ANSWERS TO FAILED RETRIEVALS WHEN AVAILABLE. **"""
    generated_path = Path(generated_answers_path)

    if not generated_path.exists():
        return failed_queries

    generated_answers = load_json(str(generated_path))
    generated_lookup = {
        item["query_id"]: item
        for item in generated_answers
    }

    enriched = []

    for failure in failed_queries:
        query_id = failure["query_id"]
        generated_item = generated_lookup.get(query_id, {})

        enriched.append(
            {
                **failure,
                "generated_answer": generated_item.get("answer", ""),
                "retrieved_chunks": generated_item.get("retrieved_chunks", []),
            }
        )

    return enriched


def classify_failure(failure: Dict[str, Any]) -> str:
    """CLASSIFY FAILURE TYPE USING SIMPLE HEURISTICS. **"""
    expected_docs = set(failure.get("expected_doc_ids", []))
    retrieved_docs = set(failure.get("retrieved_doc_ids", []))

    if not expected_docs.intersection(retrieved_docs):
        return "retrieval_error"

    if failure.get("generated_answer"):
        return "generation_or_grounding_error"

    return "unknown"


def build_error_analysis(
    experiment_name: str,
    output_dir: str,
    max_failures: int = 10,
) -> Dict[str, Any]:
    """BUILD ERROR ANALYSIS REPORT FOR ONE EXPERIMENT. **"""
    experiment_dir = Path(output_dir) / experiment_name

    metrics_path = experiment_dir / "retrieval_metrics.json"
    generated_answers_path = experiment_dir / "generated_answers.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    failed_queries = find_failed_retrievals(str(metrics_path))
    failed_queries = attach_generation_outputs(
        failed_queries=failed_queries,
        generated_answers_path=str(generated_answers_path),
    )

    analyzed_failures = []

    for failure in failed_queries[:max_failures]:
        analyzed_failures.append(
            {
                **failure,
                "failure_type": classify_failure(failure),
                "diagnosis": "",
                "proposed_fix": "",
            }
        )

    return {
        "experiment_name": experiment_name,
        "total_failures": len(failed_queries),
        "sampled_failures": len(analyzed_failures),
        "failures": analyzed_failures,
        "suggested_improvements": [
            "Inspect missed queries by modality to determine whether failures are concentrated in text-table or text-image questions.",
            "Improve multimodal parsing by adding table-aware extraction and optional figure captioning/OCR.",
            "Tune hybrid retrieval fetch_k and RRF parameters to improve recall before reranking.",
        ],
    }


def save_error_analysis(
    experiment_name: str,
    output_dir: str,
    max_failures: int = 10,
) -> str:
    """SAVE ERROR ANALYSIS JSON FOR ONE EXPERIMENT. **"""
    analysis = build_error_analysis(
        experiment_name=experiment_name,
        output_dir=output_dir,
        max_failures=max_failures,
    )

    output_path = Path(output_dir) / experiment_name / "error_analysis.json"
    save_json(analysis, str(output_path))

    return str(output_path)