from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_parent_child_chunks(
    documents: List[Dict[str, Any]],
    chunk_config: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """CREATE PARENT AND CHILD CHUNKS FROM PARSED DOCUMENTS. **"""

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_config["parent_chunk_size"],
        chunk_overlap=chunk_config["parent_chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_config["child_chunk_size"],
        chunk_overlap=chunk_config["child_chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )

    parent_chunks = []
    child_chunks = []

    for document in documents:
        doc_id = document["doc_id"]
        title = document.get("title", doc_id)
        text = document.get("text", "")

        if not text:
            continue

        parent_texts = parent_splitter.split_text(text)

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"{doc_id}_parent_{parent_index}"

            parent_record = {
                "parent_id": parent_id,
                "doc_id": doc_id,
                "title": title,
                "parent_text": parent_text,
                "chunk_type": "parent",
                "metadata": {
                    "file_name": document.get("file_name", ""),
                    "parent_index": parent_index,
                },
            }

            parent_chunks.append(parent_record)

            child_texts = child_splitter.split_text(parent_text)

            for child_index, child_text in enumerate(child_texts):
                child_chunks.append(
                    {
                        "chunk_id": f"{parent_id}_child_{child_index}",
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "title": title,
                        "chunk_text": child_text,
                        "chunk_type": "child",
                        "metadata": {
                            "file_name": document.get("file_name", ""),
                            "parent_index": parent_index,
                            "child_index": child_index,
                        },
                    }
                )

    return {
        "parents": parent_chunks,
        "children": child_chunks,
    }