from __future__ import annotations

import logging
from pathlib import Path

from src.embedding.embedder import get_embedder
from src.ingestion.chunkers import chunk_document
from src.ingestion.loaders import load_document
from src.ingestion.manifest import Manifest, load_manifest, save_manifest
from src.retrieval.dense import DenseRetriever
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def ingest_files(paths: list[Path], rebuild: bool = False) -> int:
    cfg = get_config()
    manifest = load_manifest()
    embedder = get_embedder()
    retriever = DenseRetriever()

    if rebuild:
        retriever.reset()
        manifest = Manifest()
        logger.info("Rebuild mode: cleared existing index and manifest")

    total_chunks = 0
    for path in paths:
        if not rebuild and manifest.is_ingested(path):
            logger.debug("Skipping (already ingested): %s", path.name)
            continue

        logger.info("Processing: %s", path.name)
        doc = load_document(path)
        if doc is None:
            continue

        chunks = chunk_document(doc)
        if not chunks:
            logger.warning("No chunks produced for %s", path.name)
            continue

        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        retriever.add(chunks, embeddings)
        manifest.record(path, len(chunks))
        total_chunks += len(chunks)
        logger.info("  -> %d chunks from %s", len(chunks), path.name)

    save_manifest(manifest)
    logger.info("Ingestion complete. Total new chunks: %d", total_chunks)
    return total_chunks


def scan_docs(docs_path: Path, extensions: list[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(docs_path.rglob(f"*{ext}"))
    return sorted(files)
