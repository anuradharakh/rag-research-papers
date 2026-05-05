import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str) -> None:
    """CREATE DIRECTORY IF IT DOES NOT EXIST. **"""
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    """WRITE RECORDS TO JSONL FILE. **"""
    path = Path(output_path)
    ensure_dir(str(path.parent))

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """READ RECORDS FROM JSONL FILE. **"""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]