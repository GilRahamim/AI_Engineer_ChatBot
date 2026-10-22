from __future__ import annotations

# Must be imported before pymupdf4llm/chromadb to avoid native-library load-order segfault
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bisect import bisect_right

from pydantic import BaseModel

from src.ingestion.loaders import Document, Segment
from src.utils.config import get_config


class Chunk(BaseModel):
    text: str
    source: str
    file_type: str
    chunk_index: int
    metadata: dict = {}


_MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


def _span(segments: list[Segment], start: int, end: int) -> tuple[int, int]:
    """First and last segment number covering content[start:end]."""
    starts = [s.start for s in segments]
    first = max(bisect_right(starts, start) - 1, 0)
    last = max(bisect_right(starts, max(start, end - 1)) - 1, 0)
    return segments[first].number, segments[last].number


def chunk_document(doc: Document) -> list[Chunk]:

    cfg = get_config().ingestion
    splitter = RecursiveCharacterTextSplitter(
        separators=_MARKDOWN_SEPARATORS,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    for i, piece in enumerate(splitter.create_documents([doc.content])):
        text = piece.page_content
        if len(text.strip()) < cfg.min_chunk_size:
            continue
        metadata = {**doc.metadata, "chunk_index": i}
        start = piece.metadata.get("start_index", -1)
        if doc.segment_kind and doc.segments and start >= 0:
            first, last = _span(doc.segments, start, start + len(text))
            metadata[doc.segment_kind] = first
            metadata[f"{doc.segment_kind}_end"] = last
        chunks.append(
            Chunk(
                text=text,
                source=doc.source,
                file_type=doc.file_type,
                chunk_index=i,
                metadata=metadata,
            )
        )
    return chunks
