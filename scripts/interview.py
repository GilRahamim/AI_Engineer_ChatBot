"""Interactive interview simulator — ask questions, evaluate answers, track progress."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

from src.generation.pipeline import answer
from src.retrieval.pipeline import retrieve
from src.utils.config import get_config
from src.utils.logging import setup_logging

PROGRESS_FILE = Path("data/progress.json")


def _load_questions(path: Path, topic: str | None, difficulty: str | None) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            if q.get("skip_ragas"):
                continue
            if topic and q.get("topic", "").lower() != topic.lower():
                continue
            if difficulty and q.get("difficulty", "").lower() != difficulty.lower():
                continue
            questions.append(q)
    return questions


def _evaluate_answer(question: str, user_answer: str, results: list) -> dict:
    prompt_path = Path(get_config().paths.prompts) / "eval_answer.txt"
    system = prompt_path.read_text(encoding="utf-8")

    from src.generation.pipeline import _format_context, _get_client
    context = _format_context(results)
    user_msg = f"{context}\n\nQuestion: {question}\n\nStudent's answer: {user_answer}"
    raw = _get_client().complete(system=system, user=user_msg)

    # Extract JSON from response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"score": 0, "strengths": "", "gaps": "Could not parse evaluation.", "model_answer": ""}


def _save_session(session: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    else:
        data = {"sessions": []}
    data["sessions"].append(session)
    PROGRESS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_separator() -> None:
    print("\n" + "─" * 60 + "\n")


def run_interview(questions: list[dict], n: int) -> None:
    selected = random.sample(questions, min(n, len(questions)))
    session_results = []

    print(f"\nStarting interview — {len(selected)} question(s). Type your answer and press Enter twice to submit.\n")

    for i, q in enumerate(selected, 1):
        _print_separator()
        print(f"Question {i}/{len(selected)}  [{q.get('topic', '')} | {q.get('difficulty', '')}]")
        print(f"\n{q['question']}\n")

        # Collect multi-line answer
        lines = []
        print("Your answer (press Enter twice to submit):")
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        user_answer = "\n".join(lines).strip()

        if not user_answer:
            print("Skipped.")
            continue

        print("\nEvaluating...")
        results = retrieve(q["question"])
        evaluation = _evaluate_answer(q["question"], user_answer, results)

        score = evaluation.get("score", 0)
        stars = "★" * score + "☆" * (5 - score)

        _print_separator()
        print(f"Score: {stars}  ({score}/5)\n")
        if evaluation.get("strengths"):
            print(f"Strengths:  {evaluation['strengths']}\n")
        if evaluation.get("gaps"):
            print(f"Gaps:       {evaluation['gaps']}\n")
        print(f"Model answer:\n{evaluation.get('model_answer', '')}")

        session_results.append({
            "id": q["id"],
            "question": q["question"],
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", ""),
            "score": score,
            "max_score": 5,
        })

        if i < len(selected):
            cont = input("\nNext question? [Enter to continue / q to quit]: ").strip().lower()
            if cont == "q":
                break

    if not session_results:
        return

    total = sum(r["score"] for r in session_results)
    avg = total / len(session_results)
    _print_separator()
    print(f"Session complete — {len(session_results)} question(s) answered")
    print(f"Average score: {avg:.1f}/5\n")

    _save_session({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(session_results),
        "avg_score": round(avg, 2),
        "questions": session_results,
    })
    print(f"Session saved to {PROGRESS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive interview simulator")
    parser.add_argument("--topic", default=None, help="Filter by topic (e.g. 'NLP', 'Deep Learning', 'RAG')")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"], help="Filter by difficulty")
    parser.add_argument("--n", type=int, default=5, help="Number of questions (default: 5)")
    parser.add_argument("--questions", default="data/eval/golden_qa.jsonl", help="Path to questions file")
    args = parser.parse_args()

    setup_logging()
    questions = _load_questions(Path(args.questions), args.topic, args.difficulty)

    if not questions:
        print(f"No questions found for topic={args.topic!r} difficulty={args.difficulty!r}")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions", end="")
    if args.topic:
        print(f" (topic: {args.topic})", end="")
    if args.difficulty:
        print(f" (difficulty: {args.difficulty})", end="")
    print()

    run_interview(questions, args.n)


if __name__ == "__main__":
    main()
