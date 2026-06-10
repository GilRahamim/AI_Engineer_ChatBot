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


def _question_stats() -> dict[str, dict]:
    """Per-question history from saved sessions: latest score, last_seen, count."""
    stats: dict[str, dict] = {}
    if not PROGRESS_FILE.exists():
        return stats
    data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    for session in data.get("sessions", []):
        ts = session.get("timestamp", "")
        try:
            seen = datetime.fromisoformat(ts)
        except ValueError:
            continue
        for q in session.get("questions", []):
            qid = q.get("id")
            if qid is None:
                continue
            entry = stats.setdefault(qid, {"last_score": None, "last_seen": seen, "count": 0})
            entry["count"] += 1
            # Sessions are appended in order, so the last one wins for "latest".
            if seen >= entry["last_seen"]:
                entry["last_seen"] = seen
                entry["last_score"] = q.get("score")
    return stats


def _days_since(seen: datetime, now: datetime) -> float:
    return (now - seen).total_seconds() / 86400.0


def _is_due(q: dict, stats: dict, cfg, now: datetime) -> bool:
    s = stats.get(q["id"])
    if s is None or s["last_score"] is None:
        return True  # never seen (or unscored) -> always due
    score = min(max(int(s["last_score"]), 1), 5)
    interval = cfg.interval_days[score - 1]
    return _days_since(s["last_seen"], now) >= interval


def _weight(q: dict, stats: dict, cfg, now: datetime) -> float:
    s = stats.get(q["id"])
    if s is None or s["last_score"] is None:
        return cfg.unseen_weight
    score_factor = max(1, 6 - int(s["last_score"]))  # weaker answers weigh more
    recency = 2 ** (_days_since(s["last_seen"], now) / cfg.recency_half_life_days)
    return score_factor * recency


def _weighted_sample(items: list[dict], weights: list[float], k: int) -> list[dict]:
    """Weighted sampling without replacement."""
    items, weights = list(items), list(weights)
    chosen: list[dict] = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        if total <= 0:
            idx = random.randrange(len(items))
        else:
            r, upto, idx = random.uniform(0, total), 0.0, len(items) - 1
            for i, w in enumerate(weights):
                upto += w
                if upto >= r:
                    idx = i
                    break
        chosen.append(items.pop(idx))
        weights.pop(idx)
    return chosen


def _select_questions(questions: list[dict], n: int) -> list[dict]:
    cfg = get_config().interview
    if cfg.selection == "random":
        return random.sample(questions, min(n, len(questions)))

    now = datetime.now(timezone.utc)
    stats = _question_stats()
    due = [q for q in questions if _is_due(q, stats, cfg, now)]
    not_due = [q for q in questions if not _is_due(q, stats, cfg, now)]

    selected = _weighted_sample(due, [_weight(q, stats, cfg, now) for q in due], n)
    if len(selected) < n:
        remaining = n - len(selected)
        selected += _weighted_sample(
            not_due, [_weight(q, stats, cfg, now) for q in not_due], remaining
        )
    random.shuffle(selected)  # don't always lead with the weakest topic
    return selected


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
    return {"score": None, "strengths": "", "gaps": "Could not parse evaluation.", "model_answer": ""}


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
    selected = _select_questions(questions, n)
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
        # Retrieve on question + answer so correct points the student raised that
        # aren't keyed to the question's wording still surface as reference context.
        results = retrieve(f"{q['question']}\n\n{user_answer}")
        evaluation = _evaluate_answer(q["question"], user_answer, results)

        score = evaluation.get("score")
        if isinstance(score, int) and 1 <= score <= 5:
            stars = "★" * score + "☆" * (5 - score)
            score_label = f"{stars}  ({score}/5)"
        else:
            score = None
            score_label = "unscored (evaluation could not be parsed)"

        _print_separator()
        print(f"Score: {score_label}\n")
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
            "gaps": evaluation.get("gaps", ""),
            "model_answer": evaluation.get("model_answer", ""),
        })

        if i < len(selected):
            cont = input("\nNext question? [Enter to continue / q to quit]: ").strip().lower()
            if cont == "q":
                break

    if not session_results:
        return

    scored = [r["score"] for r in session_results if isinstance(r["score"], int)]
    avg = sum(scored) / len(scored) if scored else None
    _print_separator()
    print(f"Session complete — {len(session_results)} question(s) answered")
    if avg is not None:
        print(f"Average score: {avg:.1f}/5", end="")
        unscored = len(session_results) - len(scored)
        print(f"  ({unscored} unscored)\n" if unscored else "\n")
    else:
        print("Average score: n/a (no questions could be scored)\n")

    _save_session({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(session_results),
        "avg_score": round(avg, 2) if avg is not None else None,
        "questions": session_results,
    })
    print(f"Session saved to {PROGRESS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive interview simulator")
    parser.add_argument("--topic", default=None, help="Filter by topic (e.g. 'NLP', 'Deep Learning', 'RAG')")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"], help="Filter by difficulty")
    parser.add_argument("--n", type=int, default=None, help="Number of questions (default: from config)")
    parser.add_argument("--mode", default=None, choices=["adaptive", "random"],
                        help="Question selection mode (default: from config)")
    parser.add_argument("--questions", default="data/eval/golden_qa.jsonl", help="Path to questions file")
    args = parser.parse_args()

    setup_logging()
    cfg = get_config().interview
    if args.mode:
        cfg.selection = args.mode  # in-memory override for this run
    n = args.n if args.n is not None else cfg.default_n

    questions = _load_questions(Path(args.questions), args.topic, args.difficulty)

    if not questions:
        print(f"No questions found for topic={args.topic!r} difficulty={args.difficulty!r}")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions", end="")
    if args.topic:
        print(f" (topic: {args.topic})", end="")
    if args.difficulty:
        print(f" (difficulty: {args.difficulty})", end="")
    print(f"  [selection: {cfg.selection}]")

    if cfg.selection == "adaptive":
        now = datetime.now(timezone.utc)
        stats = _question_stats()
        due = sum(1 for q in questions if _is_due(q, stats, cfg, now))
        print(f"{due} question(s) due for review.")

    run_interview(questions, n)


if __name__ == "__main__":
    main()
