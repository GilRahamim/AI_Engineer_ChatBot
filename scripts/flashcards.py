#!/usr/bin/env python
"""Build a flashcard review deck from missed/weak interview questions.

Reads data/progress.json (written by scripts/interview.py), keeps the latest
attempt per question whose score is at or below the review threshold, and writes
a study deck (Markdown + JSONL) with the model answer and noted gaps on the back.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config

PROGRESS_FILE = Path("data/progress.json")


def _latest_attempts(sessions: list[dict]) -> dict[str, dict]:
    """Latest attempt per question id, carrying score/gaps/model_answer."""
    latest: dict[str, dict] = {}
    for session in sessions:
        ts = session.get("timestamp", "")
        try:
            seen = datetime.fromisoformat(ts)
        except ValueError:
            continue
        for q in session.get("questions", []):
            qid = q.get("id")
            if qid is None:
                continue
            prev = latest.get(qid)
            if prev is None or seen >= prev["_seen"]:
                latest[qid] = {**q, "_seen": seen}
    return latest


def _eligible(attempt: dict, threshold: int) -> bool:
    score = attempt.get("score")
    # Unscored attempts (parse failures) and low scores both warrant review.
    return score is None or (isinstance(score, int) and score <= threshold)


def build_deck(topic: str | None, threshold: int) -> list[dict]:
    if not PROGRESS_FILE.exists():
        return []
    data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    latest = _latest_attempts(data.get("sessions", []))

    cards = []
    for qid, attempt in latest.items():
        if not _eligible(attempt, threshold):
            continue
        if topic and attempt.get("topic", "").lower() != topic.lower():
            continue
        cards.append({
            "id": qid,
            "topic": attempt.get("topic", ""),
            "difficulty": attempt.get("difficulty", ""),
            "score": attempt.get("score"),
            "front": attempt.get("question", ""),
            "model_answer": attempt.get("model_answer", ""),
            "gaps": attempt.get("gaps", ""),
        })
    # Weakest first (None treated as lowest), then by topic for grouping.
    cards.sort(key=lambda c: (c["score"] if isinstance(c["score"], int) else -1, c["topic"]))
    return cards


def _write_markdown(cards: list[dict], path: Path) -> None:
    lines = ["# Flashcard Review Deck", "", f"{len(cards)} card(s) to review.", ""]
    for i, c in enumerate(cards, 1):
        score = c["score"] if c["score"] is not None else "unscored"
        lines.append(f"## {i}. {c['topic']} — {c['difficulty']} (last score: {score})")
        lines.append("")
        lines.append(f"**Q:** {c['front']}")
        lines.append("")
        if c["model_answer"]:
            lines.append(f"**A:** {c['model_answer']}")
            lines.append("")
        if c["gaps"]:
            lines.append(f"**You missed:** {c['gaps']}")
            lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_jsonl(cards: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a flashcard deck from weak interview answers")
    parser.add_argument("--topic", default=None, help="Filter by topic")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Max score to include (default: from config review_threshold)")
    parser.add_argument("--output-dir", default="summaries", help="Output directory (default: summaries/)")
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else get_config().interview.review_threshold
    cards = build_deck(args.topic, threshold)

    if not cards:
        print(f"No cards to review (score <= {threshold}). Run an interview session first.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "flashcards.md"
    jsonl_path = out_dir / "flashcards.jsonl"
    _write_markdown(cards, md_path)
    _write_jsonl(cards, jsonl_path)

    print(f"Built {len(cards)} flashcard(s) (score <= {threshold}).")
    print(f"  Markdown : {md_path}")
    print(f"  JSONL    : {jsonl_path}")


if __name__ == "__main__":
    main()
