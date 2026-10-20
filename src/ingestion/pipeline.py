from __future__ import annotations

import logging
from pathlib import Path

from src.embedding.embedder import get_embedder
from src.ingestion.chunkers import chunk_document
from src.ingestion.loaders import load_document
from src.ingestion.manifest import Manifest, file_hash, load_manifest, save_manifest
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.utils.config import get_config
from src.utils.paths import source_key

logger = logging.getLogger(__name__)


def ingest_files(paths: list[Path], rebuild: bool = False) -> int:
    cfg = get_config()
    manifest = load_manifest()
    embedder = get_embedder()
    dense = DenseRetriever()
    sparse = SparseRetriever()

    if rebuild:
        dense.reset()
        sparse.reset()
        manifest = Manifest()
        logger.info("Rebuild mode: cleared existing index and manifest")

    total_chunks = 0
    try:
        for path in paths:
            key = source_key(path)
            content_hash = file_hash(path)
            status = manifest.status(key, content_hash)
            if status == "unchanged":
                logger.debug("Skipping (already ingested): %s", key)
                continue
            if status == "changed":
                # Hard rule: chunks in the DB are never modified in place.
                logger.warning(
                    "Skipping %s: it changed since it was ingested. Existing chunks are "
                    "never modified — run `python scripts/ingest.py --rebuild` to pick it up.",
                    key,
                )
                continue

            original = manifest.find_by_hash(content_hash)
            if original is not None:
                logger.info("Skipping %s: identical content already ingested as %s", key, original)
                manifest.record(key, content_hash, chunk_count=0, duplicate_of=original)
                save_manifest(manifest)
                continue

            logger.info("Processing: %s", key)
            doc = load_document(path)
            if doc is None:
                continue

            chunks = chunk_document(doc)
            if not chunks:
                logger.warning("No chunks produced for %s", key)
                continue
            if len(chunks) > cfg.ingestion.warn_chunks_per_file:
                logger.warning(
                    "%s produced %d chunks — check it for noise (logs, data dumps)",
                    key, len(chunks),
                )

            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            dense.add(chunks, embeddings)
            sparse.add(chunks)
            # Saved per file so a crash never leaves indexed chunks the manifest doesn't know about.
            manifest.record(key, content_hash, chunk_count=len(chunks))
            save_manifest(manifest)
            total_chunks += len(chunks)
            logger.info("  -> %d chunks from %s", len(chunks), key)
    finally:
        sparse.build()
        save_manifest(manifest)

    logger.info("Ingestion complete. Total new chunks: %d", total_chunks)
    return total_chunks


def scan_docs(docs_path: Path, extensions: list[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(docs_path.rglob(f"*{ext}"))
    return sorted(files)
