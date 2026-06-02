#!/usr/bin/env python
"""Interactive CLI chat with course materials via RAG."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.pipeline import answer
from src.retrieval.pipeline import retrieve
from src.utils.config import load_config
from src.utils.logging import setup_logging

TASK_CHOICES = ["default", "concept", "summary", "interview"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your course materials")
    parser.add_argument(
        "--task",
        default="default",
        choices=TASK_CHOICES,
        help="Prompt task profile (default: default)",
    )
    parser.add_argument("--topic", default=None, help="Topic context (useful for summary/interview mode)")
    args = parser.parse_args()

    setup_logging()
    load_config()

    header = f"Task: {args.task}" + (f"  |  Topic: {args.topic}" if args.topic else "")
    print("=" * 60)
    print("  Course RAG — Chat Mode")
    print(f"  {header}")
    print("  Type 'quit' or Ctrl+C to exit")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        effective_query = f"{args.topic}: {query}" if args.topic else query

        results = retrieve(effective_query)
        if not results:
            print("Assistant: No relevant documents found in the index.\n")
            continue

        sources = ", ".join({Path(r.source).name for r in results})
        print(f"\n[{len(results)} chunks retrieved from: {sources}]\n")

        response = answer(effective_query, results, task=args.task)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
