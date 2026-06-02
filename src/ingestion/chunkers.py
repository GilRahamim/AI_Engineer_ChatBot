from __future__ import annotations

# Must be imported before pymupdf4llm/chromadb to avoid native-library load-order segfault
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel

from src.ingestion.loaders import Document
from src.utils.config import get_config


class Chunk(BaseModel):
    text: str
    source: str
    file_type: str
    chunk_index: int
    metadata: dict = {}


_MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


def chunk_document(doc: Document) -> list[Chunk]:

    cfg = get_config().ingestion
    splitter = RecursiveCharacterTextSplitter(
        separators=_MARKDOWN_SEPARATORS,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
    )
    texts = splitter.split_text(doc.content)
    return [
        Chunk(
            text=text,
            source=doc.source,
            file_type=doc.file_type,
            chunk_index=i,
            metadata={**doc.metadata, "chunk_index": i},
        )
        for i, text in enumerate(texts)
        if len(text.strip()) >= cfg.min_chunk_size
    ]
