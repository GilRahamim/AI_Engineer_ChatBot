from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional

import fitz  # PyMuPDF
import nbformat
import pymupdf4llm
from pydantic import BaseModel

from src.utils.config import get_config
from src.utils.paths import source_key

logger = logging.getLogger(__name__)

# Minimum average characters per page — below this we treat the PDF as scanned
_MIN_CHARS_PER_PAGE = 50

# Inline images pasted into notebook markdown as base64 data URIs
_DATA_URI = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        try:
            logger.info("Loading EasyOCR (en + he) on GPU...")
            _ocr_reader = easyocr.Reader(["en", "he"], gpu=True)
        except Exception:
            logger.warning("Hebrew OCR not available, falling back to English-only")
            _ocr_reader = easyocr.Reader(["en"], gpu=True)
    return _ocr_reader


def _is_scanned(text: str, page_count: int) -> bool:
    if page_count == 0:
        return True
    return len(text.strip()) / page_count < _MIN_CHARS_PER_PAGE


class Segment(BaseModel):
    start: int  # offset in Document.content where this page/cell begins
    number: int  # 1-based page or cell number


class Document(BaseModel):
    content: str
    source: str
    file_type: str
    metadata: dict = {}
    # Where each page (PDF) or cell (notebook) starts in ``content``, so chunks
    # can be mapped back to the pages they came from for citations.
    segment_kind: Optional[Literal["page", "cell"]] = None
    segments: list[Segment] = []


def _join(parts: list[tuple[int, str]], sep: str) -> tuple[str, list[Segment]]:
    """Join (number, text) parts with ``sep``, recording where each part starts."""
    pieces: list[str] = []
    segments: list[Segment] = []
    offset = 0
    for i, (number, text) in enumerate(parts):
        if i:
            pieces.append(sep)
            offset += len(sep)
        if not segments or segments[-1].number != number:
            segments.append(Segment(start=offset, number=number))
        pieces.append(text)
        offset += len(text)
    return "".join(pieces), segments


def _ocr_pdf(path: Path) -> tuple[str, list[Segment]]:
    reader = _get_ocr_reader()
    doc = fitz.open(str(path))
    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        results = reader.readtext(img_bytes, detail=0)
        page_text = " ".join(results).strip()
        if page_text:
            pages.append((page_num + 1, f"[Page {page_num + 1}]\n{page_text}"))
    doc.close()
    return _join(pages, "\n\n")


def load_pdf(path: Path) -> Document:
    # Per-page output joined with "" is exactly what to_markdown() returns for the
    # whole document, so chunk text is unchanged — chunks only gain page numbers.
    page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    md_text, segments = _join(
        [
            (p["metadata"].get("page_number") or p["metadata"].get("page") or i, p["text"])
            for i, p in enumerate(page_chunks, start=1)
        ],
        "",
    )

    doc = fitz.open(str(path))
    page_count = doc.page_count
    doc.close()

    if _is_scanned(md_text, page_count):
        logger.info("Scanned PDF detected, running OCR: %s", path.name)
        md_text, segments = _ocr_pdf(path)
        extraction_method = "ocr"
    else:
        extraction_method = "text"

    return Document(
        content=md_text,
        source=source_key(path),
        file_type="pdf",
        metadata={"filename": path.name, "extraction_method": extraction_method},
        segment_kind="page",
        segments=segments,
    )


def _truncate_output(text: str, max_lines: int, max_chars: int) -> str:
    """Keep the head and tail of a long cell output.

    Training loops print thousands of near-identical log lines; left whole they
    flood the index with chunks that match every query about the topic.
    """
    lines = text.splitlines()
    if len(lines) > max_lines:
        head = (max_lines + 1) // 2
        tail = max_lines - head
        omitted = len(lines) - max_lines
        lines = lines[:head] + [f"... [{omitted} lines omitted] ..."] + lines[len(lines) - tail:]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n... [output truncated]"
    return out


def load_notebook(path: Path) -> Document:
    cfg = get_config().ingestion
    nb = nbformat.read(str(path), as_version=4)
    parts: list[tuple[int, str]] = []  # (1-based cell number, text)
    for cell_no, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "markdown":
            parts.append((cell_no, _DATA_URI.sub("[embedded image]", cell.source)))
        elif cell.cell_type == "code":
            code = cell.source.strip()
            if code:
                parts.append((cell_no, f"```python\n{code}\n```"))
            for output in cell.get("outputs", []):
                otype = output.get("output_type", "")
                if otype in ("stream", "display_data", "execute_result"):
                    text = output.get("text") or "".join(
                        output.get("data", {}).get("text/plain", [])
                    )
                    if isinstance(text, list):
                        text = "".join(text)
                    if text.strip():
                        text = _truncate_output(
                            text.strip(),
                            cfg.notebook_output_max_lines,
                            cfg.notebook_output_max_chars,
                        )
                        parts.append((cell_no, f"Output:\n{text}"))
    content, segments = _join(parts, "\n\n")
    return Document(
        content=content,
        source=source_key(path),
        file_type="notebook",
        metadata={"filename": path.name},
        segment_kind="cell",
        segments=segments,
    )


def load_document(path: Path) -> Optional[Document]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return load_pdf(path)
        elif suffix == ".ipynb":
            return load_notebook(path)
        else:
            logger.warning("Unsupported file type: %s", path)
            return None
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None
