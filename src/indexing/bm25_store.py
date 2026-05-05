import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """TOKENIZE TEXT FOR BM25. **"""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(
    experiment_name: str,
    chunks: List[Dict[str, Any]],
    index_dir: str,
) -> None:
    """BUILD AND SAVE BM25 INDEX. **"""
    experiment_index_dir = Path(index_dir) / experiment_name
    experiment_index_dir.mkdir(parents=True, exist_ok=True)

    tokenized_corpus = [tokenize(chunk["chunk_text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunks": chunks,
    }

    output_path = experiment_index_dir / "bm25.pkl"

    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def load_bm25_index(experiment_name: str, index_dir: str) -> Dict[str, Any]:
    """LOAD BM25 INDEX. **"""
    input_path = Path(index_dir) / experiment_name / "bm25.pkl"

    if not input_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {input_path}")

    with input_path.open("rb") as file:
        return pickle.load(file)