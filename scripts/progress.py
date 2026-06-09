"""Display study progress summary from saved session history."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROGRESS_FILE = Path("data/progress.json")


def _stars(score: float, max_score: float = 5.0) -> str:
    filled = round(score)
    return "★" * filled + "☆" * (5 - filled)


def main() -> None:
    if not PROGRESS_FILE.exists():
        print("No progress data found. Complete an interview session first.")
        print(f"  Run: python scripts/interview.py")
        sys.exit(0)

    data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])

    if not sessions:
        print("No sessions recorded yet.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  STUDY PROGRESS  —  {len(sessions)} session(s)")
    print(f"{'='*60}\n")

    # Overall stats
    all_questions = [q for s in sessions for q in s.get("questions", [])]
    if all_questions:
        overall_avg = sum(q["score"] for q in all_questions) / len(all_questions)
        print(f"Total questions answered : {len(all_questions)}")
        print(f"Overall average score    : {overall_avg:.2f}/5  {_stars(overall_avg)}\n")

    # Per-topic breakdown
    topic_scores: dict[str, list[float]] = defaultdict(list)
    for q in all_questions:
        topic = q.get("topic") or "Unknown"
        topic_scores[topic].append(q["score"])

    if topic_scores:
        print(f"{'Topic':<25} {'Questions':>9} {'Avg Score':>10}  Rating")
        print("─" * 60)
        for topic, scores in sorted(topic_scores.items()):
            avg = sum(scores) / len(scores)
            print(f"{topic:<25} {len(scores):>9} {avg:>9.2f}  {_stars(avg)}")
        print()

    # Session history (last 10)
    print(f"Recent sessions (last {min(10, len(sessions))}):")
    print(f"{'Timestamp':<30} {'Questions':>9} {'Avg Score':>10}")
    print("─" * 55)
    for s in sessions[-10:]:
        ts = s["timestamp"][:19].replace("T", " ")
        n = s.get("n_questions", len(s.get("questions", [])))
        avg = s.get("avg_score", 0)
        print(f"{ts:<30} {n:>9} {avg:>9.2f}  {_stars(avg)}")
    print()


if __name__ == "__main__":
    main()
