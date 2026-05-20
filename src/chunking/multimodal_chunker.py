from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _create_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """CREATE RECURSIVE TEXT SPLITTER. **"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )


def create_multimodal_parent_child_chunks(
    documents: List[Dict[str, Any]],
    chunk_config: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """CREATE PARENT/CHILD CHUNKS FOR TEXT, TABLES, AND FIGURE CAPTIONS. **"""

    parent_config = chunk_config["parent_child"]
    multimodal_config = chunk_config["multimodal"]

    parent_splitter = _create_splitter(
        chunk_size=parent_config["parent_chunk_size"],
        chunk_overlap=parent_config["parent_chunk_overlap"],
    )

    child_splitter = _create_splitter(
        chunk_size=parent_config["child_chunk_size"],
        chunk_overlap=parent_config["child_chunk_overlap"],
    )

    table_splitter = _create_splitter(
        chunk_size=multimodal_config.get("table_chunk_size", 1200),
        chunk_overlap=100,
    )

    figure_splitter = _create_splitter(
        chunk_size=multimodal_config.get("figure_caption_chunk_size", 500),
        chunk_overlap=50,
    )

    parent_chunks = []
    child_chunks = []

    for document in documents:
        doc_id = document["doc_id"]
        title = document.get("title", doc_id)
        pages = document.get("pages", [])

        for page in pages:
            page_number = page.get("page_number")

            # -------- TEXT CHUNKS --------
            text = page.get("text", "")

            text_parents = parent_splitter.split_text(text)

            for parent_index, parent_text in enumerate(text_parents):
                parent_id = f"{doc_id}_page_{page_number}_text_parent_{parent_index}"

                parent_chunks.append(
                    {
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "doc_id": doc_id,
                        "title": title,
                        "page_number": page_number,
                        "modality": "text",
                    }
                )

                text_children = child_splitter.split_text(parent_text)

                for child_index, child_text in enumerate(text_children):
                    child_chunks.append(
                        {
                            "chunk_id": f"{parent_id}_child_{child_index}",
                            "chunk_text": child_text,
                            "metadata": {
                                "doc_id": doc_id,
                                "title": title,
                                "page_number": page_number,
                                "parent_id": parent_id,
                                "modality": "text",
                            },
                        }
                    )

            # -------- TABLE CHUNKS --------
            for table_index, table in enumerate(page.get("tables", [])):
                table_text = table.get("markdown", "")

                if not table_text.strip():
                    continue

                table_parents = table_splitter.split_text(table_text)

                for parent_index, parent_text in enumerate(table_parents):
                    parent_id = f"{doc_id}_page_{page_number}_table_{table_index}_parent_{parent_index}"

                    parent_chunks.append(
                        {
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "doc_id": doc_id,
                            "title": title,
                            "page_number": page_number,
                            "modality": "table",
                            "table_id": table.get("table_id"),
                        }
                    )

                    child_chunks.append(
                        {
                            "chunk_id": f"{parent_id}_child_0",
                            "chunk_text": parent_text,
                            "metadata": {
                                "doc_id": doc_id,
                                "title": title,
                                "page_number": page_number,
                                "parent_id": parent_id,
                                "modality": "table",
                                "table_id": table.get("table_id"),
                            },
                        }
                    )

            # -------- FIGURE CAPTION CHUNKS --------
            for figure_index, figure in enumerate(page.get("figures", [])):
                caption = figure.get("caption", "")

                if not caption.strip():
                    continue

                figure_text = (
                    f"Figure caption: {caption}\n"
                    f"Image path: {figure.get('image_path', '')}"
                )

                figure_parents = figure_splitter.split_text(figure_text)

                for parent_index, parent_text in enumerate(figure_parents):
                    parent_id = f"{doc_id}_page_{page_number}_figure_{figure_index}_parent_{parent_index}"

                    parent_chunks.append(
                        {
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "doc_id": doc_id,
                            "title": title,
                            "page_number": page_number,
                            "modality": "figure",
                            "figure_id": figure.get("figure_id"),
                            "image_path": figure.get("image_path"),
                        }
                    )

                    child_chunks.append(
                        {
                            "chunk_id": f"{parent_id}_child_0",
                            "chunk_text": parent_text,
                            "metadata": {
                                "doc_id": doc_id,
                                "title": title,
                                "page_number": page_number,
                                "parent_id": parent_id,
                                "modality": "figure",
                                "figure_id": figure.get("figure_id"),
                                "image_path": figure.get("image_path"),
                            },
                        }
                    )

    return {
        "parents": parent_chunks,
        "children": child_chunks,
    }