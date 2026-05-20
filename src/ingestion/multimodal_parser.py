from pathlib import Path
from typing import Any, Dict, List

import fitz
import pdfplumber

from src.ingestion.figure_caption_extractor import extract_figure_captions


def table_to_markdown(table: List[List[Any]]) -> str:
    """CONVERT EXTRACTED PDF TABLE TO MARKDOWN. **"""

    if not table or len(table) < 2:
        return ""

    rows = [
        [str(cell).strip() if cell else "" for cell in row]
        for row in table
    ]

    if not rows or not rows[0]:
        return ""

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"

    body = "\n".join(
        "| " + " | ".join(row) + " |"
        for row in rows[1:]
    )

    return f"{header}\n{separator}\n{body}"


def extract_tables_from_page(
    page: Any,
    pdf_name: str,
    page_number: int,
) -> List[Dict[str, Any]]:
    """EXTRACT TABLES FROM ONE PDF PAGE AS MARKDOWN. **"""

    tables = []

    extracted_tables = page.extract_tables() or []

    for table_index, table in enumerate(extracted_tables):
        markdown = table_to_markdown(table)

        if not markdown.strip():
            continue

        tables.append(
            {
                "table_id": f"{pdf_name}_table_{page_number}_{table_index}",
                "markdown": markdown,
                "page_number": page_number,
            }
        )

    return tables


def extract_figures_from_page(
    fitz_doc: fitz.Document,
    page_index: int,
    pdf_name: str,
    page_number: int,
    figure_output_dir: Path,
    captions: List[str],
    save_extracted_images: bool = True,
) -> List[Dict[str, Any]]:
    """EXTRACT FIGURE IMAGES AND ATTACH AVAILABLE CAPTIONS. **"""

    figures = []
    fitz_page = fitz_doc.load_page(page_index)
    image_list = fitz_page.get_images(full=True)

    for image_index, image in enumerate(image_list):
        xref = image[0]

        figure_id = f"{pdf_name}_figure_{page_number}_{image_index}"
        caption = captions[image_index] if image_index < len(captions) else ""

        image_path = ""

        if save_extracted_images:
            try:
                base_image = fitz_doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_path_obj = figure_output_dir / f"page_{page_number}_img_{image_index}.{image_ext}"

                with image_path_obj.open("wb") as image_file:
                    image_file.write(image_bytes)

                image_path = str(image_path_obj)

            except Exception:
                image_path = ""

        figures.append(
            {
                "figure_id": figure_id,
                "image_path": image_path,
                "caption": caption,
                "page_number": page_number,
            }
        )

    return figures


def parse_multimodal_pdf(
    pdf_path: str,
    output_figure_dir: str,
    caption_patterns: List[str],
    max_caption_lines: int = 3,
    save_extracted_images: bool = True,
) -> Dict[str, Any]:
    """PARSE PDF WITH TEXT, TABLES, FIGURE METADATA, AND FIGURE CAPTIONS. **"""

    pdf_path_obj = Path(pdf_path)
    pdf_name = pdf_path_obj.stem

    figure_output_dir = Path(output_figure_dir) / pdf_name
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    fitz_doc = fitz.open(str(pdf_path_obj))

    parsed_pages = []

    metadata = fitz_doc.metadata or {}

    with pdfplumber.open(str(pdf_path_obj)) as plumber_pdf:
        for page_index, page in enumerate(plumber_pdf.pages):
            page_number = page_index + 1

            text = page.extract_text() or ""

            tables = extract_tables_from_page(
                page=page,
                pdf_name=pdf_name,
                page_number=page_number,
            )

            captions = extract_figure_captions(
                text=text,
                patterns=caption_patterns,
                max_caption_lines=max_caption_lines,
            )

            figures = extract_figures_from_page(
                fitz_doc=fitz_doc,
                page_index=page_index,
                pdf_name=pdf_name,
                page_number=page_number,
                figure_output_dir=figure_output_dir,
                captions=captions,
                save_extracted_images=save_extracted_images,
            )

            parsed_pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "tables": tables,
                    "figures": figures,
                }
            )

    fitz_doc.close()

    return {
        "doc_id": pdf_name,
        "title": metadata.get("title") or pdf_name,
        "authors": metadata.get("author", ""),
        "metadata": metadata,
        "pages": parsed_pages,
    }


def parse_multimodal_pdf_directory(
    pdf_dir: str,
    output_figure_dir: str,
    caption_patterns: List[str],
    max_caption_lines: int = 3,
    save_extracted_images: bool = True,
) -> List[Dict[str, Any]]:
    """PARSE ALL PDFS IN DIRECTORY WITH MULTIMODAL EXTRACTION. **"""

    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    parsed_documents = []

    for pdf_path in pdf_files:
        parsed_documents.append(
            parse_multimodal_pdf(
                pdf_path=str(pdf_path),
                output_figure_dir=output_figure_dir,
                caption_patterns=caption_patterns,
                max_caption_lines=max_caption_lines,
                save_extracted_images=save_extracted_images,
            )
        )

    return parsed_documents