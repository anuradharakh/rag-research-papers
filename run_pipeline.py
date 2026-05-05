from src.utils.config_loader import load_config, get_enabled_experiments
from src.utils.logger import log_info, log_success, log_warning
from pathlib import Path
from src.ingestion.pdf_parser import parse_pdf_directory
from src.utils.io import write_jsonl


def run_experiment(experiment_name: str, experiment_config: dict, global_config: dict) -> None:
    """RUN ONE EXPERIMENT PIPELINE. **"""

    log_info(f"Starting experiment: {experiment_name}")
    log_info(f"Description: {experiment_config.get('description', 'No description')}")

    pipeline_config = global_config["pipeline"]

    if pipeline_config.get("run_ingestion", False):
        log_info("Step 1: Ingestion already completed once globally.")

    if pipeline_config.get("run_chunking", False):
        log_info(f"Step 2: Chunking strategy = {experiment_config['chunking_strategy']}")

    if pipeline_config.get("run_indexing", False):
        log_info("Step 3: Indexing will run here.")

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