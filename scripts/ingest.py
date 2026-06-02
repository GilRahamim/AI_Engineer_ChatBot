#!/usr/bin/env python
"""Bulk ingest all course documents from data/docs/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import ingest_files, scan_docs
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingest course documents")
    parser.add_argument("--rebuild", action="store_true", help="Clear existing index and re-ingest everything")
    parser.add_argument("--docs", default=None, help="Override docs directory path")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()

    docs_path = Path(args.docs) if args.docs else Path(cfg.paths.docs)
    if not docs_path.exists():
        print(f"Docs directory not found: {docs_path}")
        print("Copy your course files to data/docs/ first, or pass --docs <path>")
        sys.exit(1)

    files = scan_docs(docs_path, cfg.ingestion.supported_extensions)
    print(f"Found {len(files)} files in {docs_path}")
    if not files:
        print("No .pdf or .ipynb files found.")
        sys.exit(0)

    ingest_files(files, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
