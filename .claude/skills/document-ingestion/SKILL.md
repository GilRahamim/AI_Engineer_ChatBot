---
name: document-ingestion
description: Use this skill when adding course documents to the RAG knowledge base, parsing PDFs or Jupyter notebooks, extracting structured metadata, implementing or modifying chunking logic, or debugging ingestion issues. Triggers include "ingest new files", "add PDF to knowledge base", "process notebooks", "update document index", "fix chunking", "improve metadata extraction".
---

# Document Ingestion

This skill governs how raw course materials become searchable chunks. The quality of every retrieval downstream depends on what happens here. Bad chunking is unrecoverable — no reranker can fix a chunk that splits a formula in half.

## Core principles

1. **Type-aware processing** — PDFs and notebooks are *not* the same. Treat them differently.
2. **Preserve structure** — chunk along semantic boundaries (sections, cells), not arbitrary token counts.
3. **Rich metadata** — every chunk gets enough metadata to filter, cite, and debug.
4. **Idempotent ingestion** — re-running ingestion on existing files must be a no-op (via content hashing).

## Document loaders

### PDFs
Use `pymupdf4llm.to_markdown()` rather than raw text extraction. It preserves heading hierarchy, code blocks, and tables.

```python
import pymupdf4llm
md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
# Returns list of dicts with 'text', 'metadata' (page number, etc.)
```

Why not plain PyMuPDF text: loses structure, returns wall of text with broken section boundaries.

### Jupyter notebooks
Parse with `nbformat`. **Never** convert the whole notebook to a single string — you lose the cell-type signal.

```python
import nbformat
nb = nbformat.read(path, as_version=4)
for idx, cell in enumerate(nb.cells):
    if cell.cell_type in ("markdown", "code"):
        yield {
            "text": cell.source,
            "cell_type": cell.cell_type,  # CRITICAL — used for filtering later
            "cell_index": idx,
        }
```

A markdown cell explaining backprop and a code cell implementing it should never end up in the same chunk.

## Chunking strategy

### Hierarchical approach
Try in order, stop at the first that fits:

1. **Semantic boundary** — section in PDF, cell in notebook. If under `max_chunk_tokens`, use as-is.
2. **Paragraph split** — within a long section, split on `\n\n` and group paragraphs greedily up to `max_chunk_tokens`.
3. **Recursive character split** — last resort. Use `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,         # tokens, not characters — use tiktoken length_function
    chunk_overlap=150,
    length_function=lambda t: len(tiktoken.get_encoding("cl100k_base").encode(t)),
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)
```

Note the leading `\n## ` and `\n### ` — these are markdown heading markers from `pymupdf4llm`. We split on those *first* to preserve sections.

### Code cells
Code cells need different chunking. Don't split mid-function.

- Try AST-based splitting first (split on top-level `def`/`class`).
- Fallback: keep entire cell as one chunk if reasonably sized (up to ~1500 tokens).
- If too large: split on blank lines between functions, never inside a function body.

### Don't
- Don't chunk by fixed character count. 500 chars cuts mid-word.
- Don't chunk by sentence in code. `df.groupby('x').agg({'y': 'sum'})` is one logical unit.
- Don't strip whitespace aggressively in code chunks. Indentation is meaning.

## Metadata schema

Every chunk MUST have:

```python
class ChunkMetadata(BaseModel):
    chunk_id: str              # deterministic hash of (source + position + content)
    source: str                # filename, e.g. "module2_neural_networks.pdf"
    source_type: Literal["pdf", "notebook"]
    module: str | None         # e.g. "Module 2: Deep Learning & NLP" — inferred from path or content
    section: str | None        # e.g. "2.2 Fundamentals of Neural Networks"
    page: int | None           # PDFs only
    cell_index: int | None     # notebooks only
    cell_type: Literal["markdown", "code"] | None  # notebooks only
    content_type: Literal["text", "code", "formula", "table"]
    token_count: int           # for budgeting later
    doc_hash: str              # hash of full source file — for manifest
```

Module inference: parse filename or path. Pattern: `data/docs/module_2_dl_nlp/file.pdf` → `Module 2`. If unsure, leave `None`.

ChromaDB requires metadata values to be `str | int | float | bool` — flatten unions and Enums to `str` before storing.

## Manifest and deduplication

Track every ingested file in `data/manifest.json`:

```json
{
  "files": {
    "data/docs/module2/neural_networks.pdf": {
      "doc_hash": "sha256-...",
      "ingested_at": "2026-05-07T10:00:00Z",
      "chunk_count": 47,
      "embedder_version": "BAAI/bge-m3@v1"
    }
  }
}
```

On `add_files.py` run:
1. Compute `doc_hash` for every file in `data/docs/`.
2. Skip files where `doc_hash` matches the manifest entry.
3. For new/changed files: ingest, then update manifest.
4. **Never partially update a file** — if a file changed, delete *all* its old chunks first (filter by `source` in metadata) before re-ingesting.

If the embedder model changes, `embedder_version` mismatches force a full rebuild — don't try to mix vectors from different models.

## Common pitfalls

- **Mixed embedding models in the same DB** — silent retrieval quality collapse. Hard guard against it via `embedder_version`.
- **Forgetting `cell_type` filter** — when answering "show me how to implement X", you want code cells. The metadata enables this; use it.
- **PDF page numbers off by 1** — PyMuPDF is 0-indexed, the printed page number can be different. Store both `page_index` (0-based, internal) and `page_label` (printed, for citations) when possible.
- **Hebrew text in PDFs** — RTL can come out reversed character-by-character. Test with `pymupdf4llm` — it usually handles this, but verify on a Hebrew section before bulk ingestion.

## Sanity checks

After every ingestion run, log:
- Number of new chunks created
- Distribution of `content_type` (should not be 100% text — code chunks should appear from notebooks)
- Average tokens per chunk (should be near `chunk_size`, not wildly varying)
- Any chunks above `max_chunk_tokens` (= bug, should be 0)
- Any duplicate `chunk_id` (= bug, should be 0)
