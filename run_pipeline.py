import json
from pathlib import Path

from src.chunking.parent_child_chunker import create_parent_child_chunks
from src.chunking.recursive_chunker import create_recursive_chunks
from src.evaluation.hit_rate import compute_hit_rate_at_k, load_json
from src.indexing.bm25_store import build_bm25_index
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_store import build_chroma_index
from src.ingestion.pdf_parser import parse_pdf_directory
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.parent_child_retriever import ParentChildRetriever
from src.utils.config_loader import get_enabled_experiments, load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import log_info, log_success, log_warning
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.parent_child_hybrid_retriever import ParentChildHybridRetriever


def run_experiment(experiment_name: str, experiment_config: dict, global_config: dict) -> None:
    """RUN ONE EXPERIMENT PIPELINE. **"""
    log_info(f"Starting experiment: {experiment_name}")
    log_info(f"Description: {experiment_config.get('description', 'No description')}")

    pipeline_config = global_config["pipeline"]

    if pipeline_config.get("run_ingestion", False):
        log_info("Step 1: Ingestion already completed once globally.")

    if pipeline_config.get("run_chunking", False):
        run_chunking_for_experiment(experiment_name, experiment_config, global_config)

    if pipeline_config.get("run_indexing", False):
        run_indexing_for_experiment(experiment_name, experiment_config, global_config)

    if pipeline_config.get("run_retrieval_eval", False):
        run_retrieval_evaluation(experiment_name, experiment_config, global_config)

    if pipeline_config.get("run_generation", False):
        log_info("Step 5: Generation will run here.")
    else:
        log_warning("Generation is disabled.")

    if pipeline_config.get("run_ragas_eval", False):
        log_info("Step 6: RAGAS evaluation will run here.")
    else:
        log_warning("RAGAS evaluation is disabled.")

    log_success(f"Finished experiment: {experiment_name}")


def run_ingestion_once(config: dict) -> str:
    """RUN PDF INGESTION ONCE AND SAVE PARSED DOCUMENTS. **"""
    pdf_dir = config["paths"]["pdf_dir"]
    processed_dir = config["paths"]["processed_dir"]
    output_path = str(Path(processed_dir) / "parsed_documents.jsonl")

    log_info(f"Parsing PDFs from: {pdf_dir}")

    parsed_documents = parse_pdf_directory(pdf_dir)

    if not parsed_documents:
        log_warning("No parsed documents were created. Please add PDFs to data/raw/pdfs.")
        return output_path

    write_jsonl(parsed_documents, output_path)

    log_success(f"Saved parsed documents to: {output_path}")
    return output_path


def run_chunking_for_experiment(
    experiment_name: str,
    experiment_config: dict,
    global_config: dict,
) -> None:
    """RUN CHUNKING FOR ONE EXPERIMENT. **"""
    parsed_path = Path(global_config["paths"]["processed_dir"]) / "parsed_documents.jsonl"
    output_dir = Path(global_config["paths"]["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = read_jsonl(str(parsed_path))
    chunking_strategy = experiment_config["chunking_strategy"]

    if chunking_strategy == "recursive":
        chunks = create_recursive_chunks(
            documents=documents,
            chunk_config=global_config["chunking"]["recursive"],
        )

        output_path = output_dir / "chunks.jsonl"
        write_jsonl(chunks, str(output_path))
        log_success(f"Saved {len(chunks)} recursive chunks to {output_path}")

    elif chunking_strategy == "parent_child":
        chunk_sets = create_parent_child_chunks(
            documents=documents,
            chunk_config=global_config["chunking"]["parent_child"],
        )

        parent_path = output_dir / "parent_chunks.jsonl"
        child_path = output_dir / "child_chunks.jsonl"

        write_jsonl(chunk_sets["parents"], str(parent_path))
        write_jsonl(chunk_sets["children"], str(child_path))

        log_success(f"Saved {len(chunk_sets['parents'])} parent chunks to {parent_path}")
        log_success(f"Saved {len(chunk_sets['children'])} child chunks to {child_path}")

    else:
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")


def get_chunks_path_for_experiment(
    experiment_name: str,
    experiment_config: dict,
    global_config: dict,
) -> Path:
    """RETURN CHUNKS PATH USED FOR INDEXING. **"""
    output_dir = Path(global_config["paths"]["output_dir"]) / experiment_name

    if experiment_config["chunking_strategy"] == "parent_child":
        return output_dir / "child_chunks.jsonl"

    return output_dir / "chunks.jsonl"


def run_indexing_for_experiment(
    experiment_name: str,
    experiment_config: dict,
    global_config: dict,
) -> None:
    """RUN VECTOR INDEXING FOR ONE EXPERIMENT. **"""
    index_dir = Path(global_config["paths"]["index_dir"]) / experiment_name
    chunks_path = get_chunks_path_for_experiment(
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        global_config=global_config,
    )

    chunks = read_jsonl(str(chunks_path))

    if not chunks:
        log_warning(f"No chunks found for indexing: {experiment_name}")
        return

    embedding_config = global_config["models"]["embedding"]

    log_info(f"Loading embedding model: {embedding_config['name']}")
    embedding_model = EmbeddingModel(
        model_name=embedding_config["name"],
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )

    texts = [chunk["chunk_text"] for chunk in chunks]

    log_info(f"Embedding {len(texts)} chunks for {experiment_name}")
    embeddings = embedding_model.embed_texts(
        texts=texts,
        batch_size=embedding_config.get("batch_size", 64),
    )

    log_info(f"Building Chroma index for {experiment_name}")
    build_chroma_index(
        experiment_name=experiment_name,
        chunks=chunks,
        embeddings=embeddings,
        index_dir=str(index_dir),
    )

    if experiment_config["retrieval_strategy"] == "hybrid":
        log_info(f"Building BM25 index for {experiment_name}")
        build_bm25_index(
            experiment_name=experiment_name,
            chunks=chunks,
            index_dir=global_config["paths"]["index_dir"],
        )
        log_success(f"BM25 index created for {experiment_name}")

    log_success(f"Vector index created for {experiment_name}: {index_dir}")


def build_experiment_retriever(
    experiment_name: str,
    experiment_config: dict,
    global_config: dict,
):
    """BUILD RETRIEVER BASED ON EXPERIMENT CONFIG. **"""
    embedding_config = global_config["models"]["embedding"]
    embedding_model = EmbeddingModel(
        model_name=embedding_config["name"],
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )

    dense_retriever = DenseRetriever(
        experiment_name=experiment_name,
        index_dir=global_config["paths"]["index_dir"],
        embedding_model=embedding_model,
    )

    chunking_strategy = experiment_config["chunking_strategy"]
    retrieval_strategy = experiment_config["retrieval_strategy"]

    if chunking_strategy == "parent_child" and retrieval_strategy == "dense":
        return (
            ParentChildRetriever(
                experiment_name=experiment_name,
                output_dir=global_config["paths"]["output_dir"],
                dense_retriever=dense_retriever,
            ),
            "parent_child",
        )

    if chunking_strategy == "parent_child" and retrieval_strategy == "hybrid":
        hybrid_retriever = HybridRetriever(
            experiment_name=experiment_name,
            index_dir=global_config["paths"]["index_dir"],
            dense_retriever=dense_retriever,
            rrf_k=global_config["retrieval"]["hybrid"]["rrf_k"],
        )

        return (
            ParentChildHybridRetriever(
                experiment_name=experiment_name,
                output_dir=global_config["paths"]["output_dir"],
                hybrid_retriever=hybrid_retriever,
            ),
            "parent_child_hybrid",
        )

    if retrieval_strategy == "dense":
        return dense_retriever, "dense"

    if retrieval_strategy == "hybrid":
        return (
            HybridRetriever(
                experiment_name=experiment_name,
                index_dir=global_config["paths"]["index_dir"],
                dense_retriever=dense_retriever,
                rrf_k=global_config["retrieval"]["hybrid"]["rrf_k"],
            ),
            "hybrid",
        )

    raise ValueError(f"Unsupported retrieval strategy: {retrieval_strategy}")


def run_retrieval_evaluation(
    experiment_name: str,
    experiment_config: dict,
    global_config: dict,
) -> None:
    """RUN HIT RATE RETRIEVAL EVALUATION FOR ONE EXPERIMENT. **"""
    queries = load_json(global_config["paths"]["queries_path"])
    qrels = load_json(global_config["paths"]["qrels_path"])

    output_dir = Path(global_config["paths"]["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever, retrieval_mode = build_experiment_retriever(
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        global_config=global_config,
    )

    top_k = global_config["retrieval"]["top_k"]
    fetch_k = global_config["retrieval"]["fetch_k"]

    reranker = None

    if experiment_config.get("reranker_enabled", False):
        reranker_config = global_config["models"]["reranker"]
        log_info(f"Loading reranker model: {reranker_config['name']}")
        reranker = CrossEncoderReranker(model_name=reranker_config["name"])

    retrieval_results = []

    for query_id, query_payload in queries.items():
        question = query_payload["query"] if isinstance(query_payload, dict) else str(query_payload)

        candidate_k = fetch_k if reranker is not None else top_k

        if retrieval_mode in ["hybrid", "parent_child", "parent_child_hybrid"]:
            retrieved_chunks = retriever.retrieve(
                query=question,
                fetch_k=fetch_k,
                top_k=candidate_k,
            )
        else:
            retrieved_chunks = retriever.retrieve(
                query=question,
                top_k=candidate_k,
            )

        if reranker is not None:
            retrieved_chunks = reranker.rerank(
                query=question,
                chunks=retrieved_chunks,
                top_n=top_k,
            )

        retrieval_results.append(
            {
                "query_id": query_id,
                "question": question,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    retrieval_output_path = output_dir / "retrieval_results.json"

    with retrieval_output_path.open("w", encoding="utf-8") as file:
        json.dump(retrieval_results, file, indent=2, ensure_ascii=False)

    hit_rate_result = compute_hit_rate_at_k(
        retrieval_results=retrieval_results,
        qrels=qrels,
        k=global_config["evaluation"]["hit_rate_k"],
    )

    metrics_output_path = output_dir / "retrieval_metrics.json"

    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(hit_rate_result, file, indent=2, ensure_ascii=False)

    log_success(
        f"{experiment_name} Hit Rate@{hit_rate_result['k']}: "
        f"{hit_rate_result['hit_rate']:.4f} "
        f"({hit_rate_result['hits']}/{hit_rate_result['total_queries']})"
    )

def main() -> None:
    """MAIN PIPELINE ENTRYPOINT. **"""
    config = load_config("config.yaml")
    enabled_experiments = get_enabled_experiments(config)

    log_info(f"Project: {config['project']['name']}")
    log_info(f"Enabled experiments: {list(enabled_experiments.keys())}")

    if not enabled_experiments:
        log_warning("No experiments enabled in config.yaml.")
        return

    if config["pipeline"].get("run_ingestion", False):
        run_ingestion_once(config)

    for experiment_name, experiment_config in enabled_experiments.items():
        run_experiment(experiment_name, experiment_config, config)

    log_success("All enabled experiment(s) completed.")


if __name__ == "__main__":
    main()