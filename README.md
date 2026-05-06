# Research Paper RAG Assistant

A configurable Retrieval-Augmented Generation (RAG) pipeline for question answering over AI/ML research papers.  
The system supports multiple RAG architectures, retrieval evaluation, answer generation, RAGAS evaluation, and error analysis.


## Project Overview

This project builds a RAG pipeline over a corpus of scientific PDF papers.

The pipeline supports:

- PDF ingestion
- recursive chunking
- parent-child hierarchical chunking
- dense vector retrieval
- BM25 keyword retrieval
- hybrid retrieval with Reciprocal Rank Fusion (RRF)
- cross-encoder reranking
- grounded answer generation
- Hit Rate@3 evaluation
- RAGAS evaluation
- error analysis


## Architectures Implemented

### A1. Baseline Dense RAG

A standard semantic retrieval pipeline using recursive chunking and dense vector search.

```text
PDFs
→ Recursive Chunking
→ Embedding Generation
→ Chroma Vector Index
→ Dense Retrieval
→ Top-k Chunks
```
### A2. Hybrid BM25 + Dense + Reranker
A hybrid retrieval architecture combining semantic and lexical retrieval with cross-encoder reranking

```text
PDFs
→ Recursive Chunking
→ Embeddings + BM25 Index
→ Dense Retrieval + BM25 Retrieval
→ Reciprocal Rank Fusion (RRF)
→ Cross-Encoder Reranker
→ Top-k Chunks
```

### A3. Parent-Child Hierarchical RAG
A hierarchical retrieval architecture using child chunks for retrieval and parent chunks for contextual expansion

```text
PDFs
→ Parent Chunks + Child Chunks
→ Child Embedding Index
→ Dense Retrieval on Child Chunks
→ Parent Context Expansion
→ Top-k Parent Contexts
```

### A4. Final Recommended Architecture
A production-style RAG architecture combining hierarchical chunking, hybrid retrieval, reciprocal rank fusion (RRF), reranking, and grounded answer generation

```text
PDFs
→ Parent-Child Chunking
→ Dense Embeddings + BM25 Index
→ Hybrid Retrieval
→ Reciprocal Rank Fusion (RRF)
→ Parent Context Expansion
→ Cross-Encoder Reranker
→ Top-k Parent Contexts
→ Grounded Answer Generation
```

# Results

## Retrieval Evaluation

Evaluation was performed using **Hit Rate@3** across 496 benchmark queries.

| Experiment | Architecture | Hits | Hit Rate@3 |
|---|---|---:|---:|
| A1 | Baseline Dense RAG | 481 / 496 | 0.9698 |
| A2 | Hybrid BM25 + Dense + RRF + Reranker | 492 / 496 | 0.9919 |
| A3 | Parent-Child Dense RAG | 481 / 496 | 0.9698 |
| A4 | Parent-Child + Hybrid + RRF + Reranker | 491 / 496 | 0.9899 |

---

## Key Findings

- Hybrid retrieval significantly improved retrieval performance over dense-only retrieval.
- BM25 helped retrieve exact scientific terminology, datasets, and model names.
- Reciprocal Rank Fusion (RRF) improved overall retrieval recall.
- Cross-encoder reranking improved top-k precision.
- Parent-child retrieval improved contextual completeness for long research papers.
- A2 achieved the highest retrieval score.
- A4 is recommended as the final production-style architecture because it combines strong retrieval accuracy with richer contextual retrieval.

---

## RAGAS Pilot Evaluation

RAGAS evaluation was performed on a small pilot sample of 5 queries using the A4 architecture.

| Metric | Score |
|---|---:|
| Faithfulness | 0.7659 |
| Answer Relevancy | 0.9459 |
| Context Precision | 0.7667 |
| Context Recall | 0.7000 |
| Answer Correctness | 0.6097 |

---

## Error Analysis

Only 5 retrieval failures were observed out of 496 benchmark queries.

Common causes of failures:

- ambiguous questions
- insufficient context in retrieved chunks
- complex multimodal content such as tables and figures
- limitations in PDF parsing structure

Potential improvements:

- table-aware chunking
- figure caption extraction
- OCR-based image processing
- modality-aware retrieval
- larger reranker candidate sets

## Known Limitations

- Full image understanding is not implemented.
- Multimodal support currently focuses on text and table-style content.
- RAGAS evaluation was performed on a pilot sample due to API cost constraints.
- Parent-child expansion may introduce broader context and slightly affect ranking precision.
- PDF parsing quality depends on the structure and formatting of individual papers.

## TODO / Next Steps

### A5. HyDE + Multi-Query Expansion

Planned improvements:

- HyDE-based query generation
- multi-query expansion
- retrieval diversification
- improved recall for ambiguous queries

Pipeline:

```text
Query
→ HyDE + Multi-Query Expansion
→ Hybrid Retrieval
→ RRF
→ Reranker
→ Generation
```

### A6. Multimodal Layer

Planned multimodal extensions:

- table-aware extraction
- figure caption extraction
- OCR-based image text extraction
- multimodal retrieval support


Pipeline:

```text
PDF
→ Text + Table + Figure Extraction
→ Multimodal Chunking
→ Hybrid Retrieval
→ Reranking
→ Generation
```

## Streamlit User Interface

Planned UI features:

- select architecture (A1–A6)
- ask questions interactively
- compare retrieval and generated answers
- visualize retrieved chunks and citations

## Repository Structure

```text
research-paper-rag/
├── README.md
├── requirements.txt
├── config.yaml
├── run_pipeline.py
├── run_error_analysis.py
│
├── data/
│   ├── raw/
│   │   ├── pdfs/
│   │   ├── queries.json
│   │   ├── answers.json
│   │   └── qrels.json
│   ├── processed/
│   └── indexes/
│
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── indexing/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
│
└── outputs/
    ├── A1_baseline_dense/
    ├── A2_hybrid_reranker/
    ├── A3_parent_child/
    └── A4_parent_child_hybrid_rerank/
```
## Tech Stack

| Category | Technology / Library | Purpose |
|---|---|---|
| Programming Language | Python | Core development language |
| PDF Parsing | PyMuPDF (`fitz`) | Extract text from research paper PDFs |
| Chunking | LangChain Text Splitters | Recursive and parent-child chunking |
| Embedding Model | `BAAI/bge-base-en-v1.5` | Dense semantic embeddings |
| Vector Database | ChromaDB | Dense vector storage and retrieval |
| Keyword Retrieval | `rank_bm25` | BM25 lexical retrieval |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) | Combine dense and BM25 rankings |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Query-document reranking |
| LLM Provider | OpenAI / Groq | Grounded answer generation |
| Generation Framework | OpenAI SDK / Groq SDK | LLM API integration |
| Evaluation | Hit Rate@3 | Retrieval evaluation |
| RAG Evaluation | RAGAS | Faithfulness and answer-quality evaluation |
| Dataset Handling | HuggingFace Datasets | RAGAS dataset formatting |
| Configuration | YAML | Pipeline configuration management |
| Logging | Rich | Colored console logging |
| Environment Variables | python-dotenv | API key management |
| Serialization | JSON / JSONL | Data storage and experiment outputs |
| Version Control | GitHub | Source code management |

# Setup Instructions

## Clone Repository

```bash
git clone https://github.com/<your-username>/research-paper-rag.git
cd research-paper-rag
```

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Add Dataset Files

Place research paper PDFs inside:

```text
data/raw/pdfs/
```

Add benchmark files inside `data/raw/`:

```text
queries.json
answers.json
qrels.json
```

## Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

## Configure Pipeline

Update `config.yaml` based on the experiment and pipeline stage.

Example:

```yaml
pipeline:
  run_ingestion: true
  run_chunking: true
  run_indexing: true
  run_retrieval_eval: true
  run_generation: false
  run_ragas_eval: false
```

Enable required architecture:

```yaml
A4_parent_child_hybrid_rerank:
  enabled: true
```

## Run Pipeline

```bash
python run_pipeline.py
```

## Run Error Analysis

```bash
python run_error_analysis.py
```

## Output Files

Generated outputs are saved under:

```text
outputs/<experiment_name>/
```

Common files:

- `retrieval_results.json`
- `retrieval_metrics.json`
- `generated_answers.json`
- `ragas_metrics.json`
- `error_analysis.json`