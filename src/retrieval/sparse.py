from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import bm25s

from src.ingestion.chunkers import Chunk
from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config

logger = logging.getLogger(__name__)

_sparse_retriever: Optional[SparseRetriever] = None


def get_sparse_retriever() -> SparseRetriever:
    global _sparse_retriever
    if _sparse_retriever is None:
        _sparse_retriever = SparseRetriever()
    return _sparse_retriever


class SparseRetriever:
    def __init__(self) -> None:
        cfg = get_config()
        self._base_path = Path(cfg.paths.bm25_index)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._corpus: list[dict] = []
        self._retriever: Optional[bm25s.BM25] = None
        self._dirty = False
        self._load()

    @property
    def _corpus_file(self) -> Path:
        return self._base_path / "corpus.jsonl"

    @property
    def _index_dir(self) -> str:
        return str(self._base_path / "index")

    @property
    def _meta_file(self) -> Path:
        return self._base_path / "index_meta.json"

    def _load(self) -> None:
        if self._corpus_file.exists():
            with open(self._corpus_file, encoding="utf-8") as f:
                self._corpus = [json.loads(line) for line in f if line.strip()]

        index_path = Path(self._index_dir)
        if not (index_path.exists() and any(index_path.iterdir())):
            if self._corpus:
                logger.warning("BM25 index missing — rebuilding from corpus")
                self._rebuild()
            return
        if self._meta_file.exists():
            indexed = json.loads(self._meta_file.read_text(encoding="utf-8")).get("num_docs")
            if indexed != len(self._corpus):
                # An interrupted ingestion appended to the corpus without re-indexing.
                logger.warning(
                    "BM25 index covers %s docs but corpus has %d — rebuilding",
                    indexed, len(self._corpus),
                )
                self._rebuild()
                return
        try:
            self._retriever = bm25s.BM25.load(self._index_dir, load_corpus=False)
            logger.info("BM25 index loaded: %d docs", len(self._corpus))
        except Exception as e:
            logger.warning("BM25 index load failed (%s) — rebuilding from corpus", e)
            self._rebuild()

    def add(self, chunks: list[Chunk]) -> None:
        """Append chunks to the corpus. The index is rebuilt once, by ``build()``."""
        records = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "file_type": chunk.file_type,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        with open(self._corpus_file, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        self._corpus.extend(records)
        self._dirty = True

    def build(self) -> None:
        """Re-index the corpus if chunks were added since the last build."""
        if self._dirty:
            self._rebuild()

    def _rebuild(self) -> None:
        self._dirty = False
        if not self._corpus:
            return
        texts = [c["text"] for c in self._corpus]
        self._retriever = bm25s.BM25()
        self._retriever.index(bm25s.tokenize(texts, stopwords="en"))
        index_path = Path(self._index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(self._index_dir)
        self._meta_file.write_text(json.dumps({"num_docs": len(texts)}), encoding="utf-8")
        logger.info("BM25 index rebuilt: %d docs", len(texts))

    def query(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        if self._retriever is None or not self._corpus:
            logger.warning("BM25 index empty — skipping sparse retrieval")
            return []
        cfg = get_config()
        k = min(top_k or cfg.retrieval.sparse_top_k, len(self._corpus))
        results, scores = self._retriever.retrieve(
            bm25s.tokenize(query, stopwords="en"), k=k
        )
        return [
            RetrievalResult(
                text=self._corpus[idx]["text"],
                source=self._corpus[idx]["source"],
                file_type=self._corpus[idx]["file_type"],
                chunk_index=self._corpus[idx]["chunk_index"],
                score=float(score),
                metadata=self._corpus[idx].get("metadata", {}),
            )
            for idx, score in zip(results[0], scores[0])
        ]

    def reset(self) -> None:
        if self._corpus_file.exists():
            self._corpus_file.unlink()
        index_path = Path(self._index_dir)
        if index_path.exists():
            shutil.rmtree(index_path)
        if self._meta_file.exists():
            self._meta_file.unlink()
        self._corpus = []
        self._retriever = None
        self._dirty = False
        logger.info("BM25 index reset")
