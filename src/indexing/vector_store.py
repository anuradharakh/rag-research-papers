from pathlib import Path
from typing import Any, Dict, List

import chromadb


def get_chroma_client(index_dir: str):
    """CREATE OR LOAD CHROMA CLIENT. **"""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=index_dir)


def build_chroma_index(
    experiment_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    index_dir: str,
    batch_size: int = 1000,
) -> None:
    """BUILD CHROMA INDEX FOR ONE EXPERIMENT IN SAFE BATCHES. **"""
    client = get_chroma_client(index_dir)

    collection_name = experiment_name.replace("-", "_")
    existing_collections = [collection.name for collection in client.list_collections()]

    if collection_name in existing_collections:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["chunk_text"] for chunk in chunks]

    metadatas = [_get_metadata(chunk) for chunk in chunks]
    
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def _get_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {})

    return {
        "doc_id": metadata.get("doc_id", chunk.get("doc_id", "")),
        "title": metadata.get("title", chunk.get("title", "")),
        "chunk_type": metadata.get("chunk_type", chunk.get("chunk_type", "")),
        "parent_id": metadata.get("parent_id", chunk.get("parent_id", "")),
        "page_number": metadata.get("page_number", chunk.get("page_number", "")),
        "modality": metadata.get("modality", chunk.get("modality", "text")),
        "table_id": metadata.get("table_id", chunk.get("table_id", "")),
        "figure_id": metadata.get("figure_id", chunk.get("figure_id", "")),
        "image_path": metadata.get("image_path", chunk.get("image_path", "")),
    }   