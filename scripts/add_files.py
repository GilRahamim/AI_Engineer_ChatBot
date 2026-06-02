#!/usr/bin/env python
"""Incrementally ingest new or changed course files (skips already-ingested unchanged files)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import ingest_files, scan_docs
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally ingest new/changed course documents")
    parser.add_argument("--docs", default=None, help="Override docs directory path")
    parser.add_argument("paths", nargs="*", help="Specific files to ingest (optional)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        docs_path = Path(args.docs) if args.docs else Path(cfg.paths.docs)
        if not docs_path.exists():
            print(f"Docs directory not found: {docs_path}")
            sys.exit(1)
        files = scan_docs(docs_path, cfg.ingestion.supported_extensions)
        print(f"Found {len(files)} files, checking for new/changed...")

    ingest_files(files, rebuild=False)


if __name__ == "__main__":
    main()
