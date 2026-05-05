from src.utils.config_loader import load_config, get_enabled_experiments
from src.utils.logger import log_info, log_success, log_warning
from pathlib import Path
from src.ingestion.pdf_parser import parse_pdf_directory
from src.utils.io import write_jsonl
from src.utils.io import read_jsonl
from src.chunking.recursive_chunker import create_recursive_chunks
from src.chunking.parent_child_chunker import create_parent_child_chunks
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_store import build_chroma_index


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
        log_info(f"Step 4: Retrieval strategy = {experiment_config['retrieval_strategy']}")

    if pipeline_config.get("run_generation", False):
        log_info("Step 5: Generation will run here.")
    else:
        log_warning("Generation is disabled.")

    if pipeline_config.get("run_ragas_eval", False):
        log_info("Step 6: RAGAS evaluation will run here.")
    else:
        log_warning("RAGAS evaluation is disabled.")

    log_success(f"Finished experiment skeleton: {experiment_name}")


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

def run_indexing_for_experiment(
    experiment_name: str,
    experiment_config: dict,
    global_config: dict,
) -> None:
    """RUN VECTOR INDEXING FOR ONE EXPERIMENT. **"""

    output_dir = Path(global_config["paths"]["output_dir"]) / experiment_name
    index_dir = Path(global_config["paths"]["index_dir"]) / experiment_name

    if experiment_config["chunking_strategy"] == "parent_child":
        chunks_path = output_dir / "child_chunks.jsonl"
    else:
        chunks_path = output_dir / "chunks.jsonl"

    chunks = read_jsonl(str(chunks_path))

    if not chunks:
        log_warning(f"No chunks found for indexing: {experiment_name}")
        return
    
    log_info(f"Embedding model: {embedding_config['name']}")

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

    log_success(f"Vector index created for {experiment_name}: {index_dir}")


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

    log_success("All enabled experiment skeletons completed.")


if __name__ == "__main__":
    main()