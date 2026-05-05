from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_recursive_chunks(
    documents: List[Dict[str, Any]],
    chunk_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """CREATE RECURSIVE TEXT CHUNKS FROM PARSED DOCUMENTS. **"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_config["chunk_size"],
        chunk_overlap=chunk_config["chunk_overlap"],
        separators=chunk_config["separators"],
    )

    chunks = []

    for document in documents:
        doc_id = document["doc_id"]
        title = document.get("title", doc_id)
        text = document.get("text", "")

        if not text:
            continue

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_recursive_{chunk_index}",
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_text": chunk_text,
                    "chunk_type": "recursive",
                    "parent_id": None,
                    "metadata": {
                        "file_name": document.get("file_name", ""),
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks