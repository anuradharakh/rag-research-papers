from src.evaluation.error_analysis import save_error_analysis

EXPERIMENT_NAME = "A4_parent_child_hybrid_rerank"
OUTPUT_DIR = "outputs"

if __name__ == "__main__":
    path = save_error_analysis(
        experiment_name=EXPERIMENT_NAME,
        output_dir=OUTPUT_DIR,
        max_failures=10,
    )
    print(f"Error analysis saved to: {path}")