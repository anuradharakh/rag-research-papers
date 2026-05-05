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

    metadatas = [
        {
            "doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "parent_id": chunk.get("parent_id") or "",
        }
        for chunk in chunks
    ]

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )