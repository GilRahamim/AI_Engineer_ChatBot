from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import nbformat
import pymupdf4llm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Minimum average characters per page — below this we treat the PDF as scanned
_MIN_CHARS_PER_PAGE = 50

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


def _ocr_pdf(path: Path) -> str:
    reader = _get_ocr_reader()
    doc = fitz.open(str(path))
    pages: list[str] = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        results = reader.readtext(img_bytes, detail=0)
        page_text = " ".join(results).strip()
        if page_text:
            pages.append(f"[Page {page_num + 1}]\n{page_text}")
    doc.close()
    return "\n\n".join(pages)


class Document(BaseModel):
    content: str
    source: str
    file_type: str
    metadata: dict = {}


def load_pdf(path: Path) -> Document:
    md_text: str = pymupdf4llm.to_markdown(str(path))

    doc = fitz.open(str(path))
    page_count = doc.page_count
    doc.close()

    if _is_scanned(md_text, page_count):
        logger.info("Scanned PDF detected, running OCR: %s", path.name)
        md_text = _ocr_pdf(path)
        extraction_method = "ocr"
    else:
        extraction_method = "text"

    return Document(
        content=md_text,
        source=str(path),
        file_type="pdf",
        metadata={"filename": path.name, "extraction_method": extraction_method},
    )


def load_notebook(path: Path) -> Document:
    nb = nbformat.read(str(path), as_version=4)
    parts: list[str] = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source)
        elif cell.cell_type == "code":
            code = cell.source.strip()
            if code:
                parts.append(f"```python\n{code}\n```")
            for output in cell.get("outputs", []):
                otype = output.get("output_type", "")
                if otype in ("stream", "display_data", "execute_result"):
                    text = output.get("text") or "".join(
                        output.get("data", {}).get("text/plain", [])
                    )
                    if isinstance(text, list):
                        text = "".join(text)
                    if text.strip():
                        parts.append(f"Output:\n{text.strip()}")
    return Document(
        content="\n\n".join(parts),
        source=str(path),
        file_type="notebook",
        metadata={"filename": path.name},
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
