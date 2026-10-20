"""Run RAGAS evaluation against the golden test set."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be imported before pymupdf4llm/chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

from src.evaluation.retrieval_eval import LOWER_IS_BETTER, run_retrieval_eval
from src.utils.config import get_config
from src.utils.logging import setup_logging


def print_results(results: dict) -> None:
    print(f"\nConfig hash: {results['config_hash']}")
    print(f"Timestamp:   {results['timestamp']}\n")
    if "metrics" in results:
        print_ragas(results["metrics"])
    if "retrieval_metrics" in results:
        print_retrieval(results)


def print_ragas(metrics: dict) -> None:
    print(f"{'RAGAS metric':<22} {'Score':>7}")
    print("-" * 31)
    for name, value in metrics.items():
        if math.isnan(value):
            print(f"{name:<22}    N/A")
        else:
            bar = "█" * int(value * 20)
            print(f"{name:<22} {value:>6.3f}  {bar}")
    print()


def print_retrieval(results: dict) -> None:
    stages = results["retrieval_metrics"]
    labelled = results.get("n_questions_labelled", 0)
    print(f"Retrieval metrics — {labelled}/{results['n_questions_total']} questions labelled")
    if not labelled:
        print("  No question has expected_sources yet: run scripts/label_sources.py to label them.")
    names = list(dict.fromkeys(n for m in stages.values() for n in m))
    print(f"{'Metric':<16}" + "".join(f"{stage:>9}" for stage in stages))
    print("-" * (16 + 9 * len(stages)))
    for name in names:
        row = "".join(
            f"{stages[stage][name]:>9.3f}" if name in stages[stage] else f"{'—':>9}"
            for stage in stages
        )
        print(f"{name:<16}{row}")
    print()


def _flatten(results: dict) -> dict[str, float]:
    flat = {f"ragas.{n}": v for n, v in results.get("metrics", {}).items()}
    for stage, metrics in results.get("retrieval_metrics", {}).items():
        flat.update({f"{stage}.{n}": v for n, v in metrics.items()})
    return flat


def print_comparison(baseline: dict, current: dict) -> None:
    b = _flatten(baseline)
    c = _flatten(current)
    b_ids = {q["id"] for q in baseline.get("per_question", [])}
    c_ids = {q["id"] for q in current.get("per_question", [])}
    if b_ids != c_ids:
        print("WARNING: the two runs scored different question sets — deltas are not comparable.")
    if baseline.get("n_questions_labelled") != current.get("n_questions_labelled"):
        print("WARNING: the number of labelled questions differs — retrieval deltas are not comparable.")
    print(f"\n{'Metric':<24} {'Baseline':>9} {'New':>7} {'Δ':>8}  Verdict")
    print("-" * 62)
    for name in (n for n in b if n in c):
        delta = c[name] - b[name]
        if math.isnan(delta):
            print(f"{name:<24} {b[name]:>9.3f} {c[name]:>7.3f} {'N/A':>8}")
            continue
        gain = -delta if name.rsplit(".", 1)[-1] in LOWER_IS_BETTER else delta
        if gain > 0.02:
            verdict = "improved"
        elif gain < -0.02:
            verdict = "REGRESSION"
        else:
            verdict = "neutral"
        print(f"{name:<24} {b[name]:>9.3f} {c[name]:>7.3f} {delta:>+8.3f}  {verdict}")
    print()


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Run RAGAS and/or retrieval evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save results JSON (e.g. data/eval/results/baseline.json)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=cfg.evaluation.golden_qa,
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
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval against expected_sources only — no generation, no RAGAS judge",
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

    if args.retrieval_only:
        results = run_retrieval_eval(args.golden, question_ids=question_ids)
    else:
        from src.evaluation.ragas_eval import run_eval

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
