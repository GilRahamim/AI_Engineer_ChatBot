#!/usr/bin/env python
"""Generate a concept card — a fixed-structure study sheet — as Markdown, JSON and .docx."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be imported before pymupdf4llm/chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

from src.generation.card import make_card
from src.generation.card_render import save_card, to_markdown
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.mathtext import latex_to_unicode

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a concept card from the course materials")
    parser.add_argument("concept", help='Concept name, e.g. "Convolution" or "קונבולוציה"')
    parser.add_argument("--language", default=None, help="Prose language: hebrew | english (default: config)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: paths.cards in config")
    parser.add_argument("--no-docx", action="store_true", help="Skip the .docx export")
    args = parser.parse_args()

    setup_logging()
    load_config()

    print(f"Building card for: {args.concept}")
    result = make_card(args.concept, language=args.language)
    print(latex_to_unicode(to_markdown(result, args.language)))
    if result.dropped_items:
        print(f"({result.dropped_items} uncited items were removed)")

    paths = save_card(result, language=args.language, out_dir=args.output_dir,
                      docx=False if args.no_docx else None)
    for kind, path in paths.items():
        print(f"Saved {kind}: {path}")


if __name__ == "__main__":
    main()
