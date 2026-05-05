from pathlib import Path
from typing import Any, Dict, List

import fitz
from tqdm import tqdm


def parse_pdf(pdf_path: Path) -> Dict[str, Any]:
    """PARSE ONE PDF INTO TEXT AND PAGE METADATA. **"""
    doc = fitz.open(pdf_path)

    pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        pages.append(
            {
                "page_number": page_index,
                "text": text,
            }
        )

        if text:
            full_text_parts.append(f"\n\n[PAGE {page_index}]\n{text}")

    metadata = doc.metadata or {}

    parsed_doc = {
        "doc_id": pdf_path.stem,
        "file_name": pdf_path.name,
        "title": metadata.get("title") or pdf_path.stem,
        "authors": metadata.get("author") or "",
        "page_count": len(doc),
        "text": "\n".join(full_text_parts).strip(),
        "pages": pages,
        "metadata": metadata,
    }

    doc.close()
    return parsed_doc


def parse_pdf_directory(pdf_dir: str) -> List[Dict[str, Any]]:
    """PARSE ALL PDF FILES IN DIRECTORY. **"""
    pdf_directory = Path(pdf_dir)

    if not pdf_directory.exists():
        print(f"[WARNING] PDF directory does not exist: {pdf_dir}")
        return []

    pdf_paths = sorted(pdf_directory.glob("*.pdf"))

    if not pdf_paths:
        print(f"[WARNING] No PDF files found in: {pdf_dir}")
        return []

    parsed_documents = []

    for pdf_path in tqdm(pdf_paths, desc="Parsing PDFs"):
        try:
            parsed_documents.append(parse_pdf(pdf_path))
        except Exception as error:
            parsed_documents.append(
                {
                    "doc_id": pdf_path.stem,
                    "file_name": pdf_path.name,
                    "error": str(error),
                    "text": "",
                    "pages": [],
                }
            )

    return parsed_documents