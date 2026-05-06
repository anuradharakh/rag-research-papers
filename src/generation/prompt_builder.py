from typing import Any, Dict, List


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """FORMAT RETRIEVED CHUNKS AS CITABLE CONTEXT. **"""
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("doc_id", "unknown")
        title = metadata.get("title", doc_id)

        context_parts.append(
            f"[Source {index}: {title}, doc_id={doc_id}]\n"
            f"{chunk['chunk_text']}"
        )

    return "\n\n".join(context_parts)


def build_grounded_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
    fallback_message: str,
) -> str:
    """BUILD GROUNDED RAG PROMPT. **"""
    context = format_context(chunks)

    return f"""
You are a careful research paper question-answering assistant.

Use ONLY the provided context to answer the question.
If the context does not contain enough information, say:
"{fallback_message}"

Rules:
- Do not use outside knowledge.
- Keep the answer concise and factual.
- Include inline citations using [Source N].
- If multiple sources support the answer, cite all relevant sources.

Context:
{context}

Question:
{question}

Answer:
""".strip()