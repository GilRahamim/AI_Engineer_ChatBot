"""Run RAGAS evaluation against the golden test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Must be imported before pymupdf4llm/chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

from src.evaluation.ragas_eval import run_eval
from src.utils.config import get_config
from src.utils.logging import setup_logging


def print_results(results: dict) -> None:
    print(f"\nConfig hash: {results['config_hash']}")
    print(f"Timestamp:   {results['timestamp']}\n")
    metrics = results["metrics"]
    print(f"{'Metric':<22} {'Score':>7}")
    print("-" * 31)
    for name, value in metrics.items():
        bar = "█" * int(value * 20)
        print(f"{name:<22} {value:>6.3f}  {bar}")
    print()


def print_comparison(baseline: dict, current: dict) -> None:
    b = baseline["metrics"]
    c = current["metrics"]
    print(f"\n{'Metric':<22} {'Baseline':>9} {'New':>7} {'Δ':>8}  Verdict")
    print("-" * 60)
    for name in b:
        delta = c[name] - b[name]
        if delta > 0.02:
            verdict = "improved"
        elif delta < -0.02:
            verdict = "REGRESSION"
        else:
            verdict = "neutral"
        print(f"{name:<22} {b[name]:>9.3f} {c[name]:>7.3f} {delta:>+8.3f}  {verdict}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save results JSON (e.g. data/eval/results/baseline.json)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/eval/golden_qa.jsonl"),
        help="Path to golden QA file",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated question IDs to run (e.g. q001,q002). Runs all if omitted.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Path to a previous results JSON to compare against",
    )
    args = parser.parse_args()

    setup_logging()

    if not args.golden.exists():
        print(f"Golden QA file not found: {args.golden}", file=sys.stderr)
        sys.exit(1)

    question_ids = [q.strip() for q in args.questions.split(",")] if args.questions else None

    print(f"Running eval on: {args.golden}")
    if question_ids:
        print(f"Questions: {question_ids}")

    results = run_eval(args.golden, question_ids=question_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")
    print_results(results)

    if args.compare:
        if not args.compare.exists():
            print(f"Comparison file not found: {args.compare}", file=sys.stderr)
        else:
            with open(args.compare, encoding="utf-8") as f:
                baseline = json.load(f)
            print(f"Comparison vs: {args.compare}")
            print_comparison(baseline, results)


if __name__ == "__main__":
    main()
