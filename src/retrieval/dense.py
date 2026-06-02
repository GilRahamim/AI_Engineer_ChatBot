from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from pydantic import BaseModel

from src.ingestion.chunkers import Chunk
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    text: str
    source: str
    file_type: str
    chunk_index: int
    score: float
    metadata: dict = {}


class DenseRetriever:
    def __init__(self) -> None:
        cfg = get_config()
        db_path = str(Path(cfg.paths.chroma_db))
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection_name = cfg.retrieval.collection_name
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._top_k = cfg.retrieval.dense_top_k
        logger.info("DenseRetriever ready: %d docs in collection", self._collection.count())

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "file_type": c.file_type, "chunk_index": c.chunk_index, **c.metadata}
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: Optional[int] = None) -> list[RetrievalResult]:
        k = top_k or self._top_k
        results = self._collection.query(query_embeddings=[embedding], n_results=k)
        return [
            RetrievalResult(
                text=doc,
                source=meta.get("source", ""),
                file_type=meta.get("file_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                score=1.0 - dist,  # cosine distance → similarity
                metadata=meta,
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset")
