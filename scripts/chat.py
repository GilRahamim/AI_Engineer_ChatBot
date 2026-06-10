#!/usr/bin/env python
"""Interactive CLI chat with course materials via RAG."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.pipeline import answer_stream, format_sources
from src.retrieval.pipeline import retrieve
from src.retrieval.transforms import condense_query
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.mathtext import latex_to_unicode

# Windows consoles default to cp1252, which raises on Unicode math (√, Σ, …).
# Force UTF-8 so rendered formulas print instead of crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

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
    cfg = load_config()
    chat_cfg = cfg.chat

    header = f"Task: {args.task}" + (f"  |  Topic: {args.topic}" if args.topic else "")
    print("=" * 60)
    print("  AI Engineer - Course  ")
    print(f"  {header}")
    print("  Type 'quit' or Ctrl+C to exit")
    print("=" * 60)
    print()

    history: list[dict] = []  # rolling [{"role", "content"}, ...]

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # Rewrite follow-ups ("explain that more") into standalone retrieval queries.
        retrieval_query = query
        if chat_cfg.condense_followups and history:
            retrieval_query = condense_query(query, history)
        if args.topic:
            retrieval_query = f"{args.topic}: {retrieval_query}"

        results = retrieve(retrieval_query)
        if not results:
            print("Assistant: No relevant documents found in the index.\n")
            continue

        sources = ", ".join({Path(r.source).name for r in results})
        print(f"\n[{len(results)} chunks retrieved from: {sources}]\n")

        # Buffer the full answer before rendering: LaTeX formulas span multiple
        # stream tokens, so math can only be converted once the text is complete.
        print("Assistant: ", end="", flush=True)
        full = ""
        for delta in answer_stream(retrieval_query, results, task=args.task, history=history):
            full += delta
        print(latex_to_unicode(full), end="")
        footer = format_sources(results, full)  # citations parse from raw [n]
        if footer:
            print(footer, end="")
        print("\n")

        # Store the clean answer (without footer) so condensing/context stay tidy.
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": full})
        max_msgs = chat_cfg.history_turns * 2
        if len(history) > max_msgs:
            history[:] = history[-max_msgs:]


if __name__ == "__main__":
    main()
