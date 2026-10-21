#!/usr/bin/env python
"""Label golden questions with the course files that answer them (expected_sources).

For each question, candidate files are pooled from the raw dense and sparse
rankings — before fusion and reranking, so labels are not biased toward what
the current pipeline already ranks first — and listed with a snippet of their
best-matching chunk. Pick every file that genuinely covers the answer.

Progress is saved after each question, so you can quit and resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be imported before pymupdf4llm/chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

from src.embedding.embedder import get_embedder
from src.evaluation.retrieval_eval import load_golden_qa
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.utils.config import get_config
from src.utils.logging import setup_logging
from src.utils.paths import normalize_key


def candidate_files(question, dense, sparse, embedder, pool):
    """Files hit by the raw dense/sparse rankings, best rank first."""
    files = {}
    rankings = [
        dense.query(embedder.embed_query(question), top_k=pool),
        sparse.query(question, top_k=pool),
    ]
    for ranking in rankings:
        for rank, r in enumerate(ranking, start=1):
            key = normalize_key(r.source)
            entry = files.setdefault(key, {"hits": 0, "best_rank": rank, "snippet": r.text})
            entry["hits"] += 1
            if rank < entry["best_rank"]:
                entry["best_rank"], entry["snippet"] = rank, r.text
    return sorted(files.items(), key=lambda kv: (kv[1]["best_rank"], -kv[1]["hits"]))


def save(path: Path, questions: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def ask(candidates) -> list[str] | None:
    """Return chosen file names, [] to skip, or None to quit."""
    while True:
        raw = input("Files (numbers like 1,3 and/or file names; Enter = skip, q = quit): ").strip()
        if raw.lower() == "q":
            return None
        if not raw:
            return []
        chosen, bad = [], []
        for token in (t.strip() for t in raw.split(",") if t.strip()):
            if token.isdigit():
                n = int(token)
                if 1 <= n <= len(candidates):
                    chosen.append(candidates[n - 1][0].rsplit("/", 1)[-1])
                else:
                    bad.append(token)
            else:
                chosen.append(token)
        if bad:
            print(f"  Out of range: {', '.join(bad)}")
            continue
        return list(dict.fromkeys(chosen))


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Label golden questions with expected source files")
    parser.add_argument("--golden", type=Path, default=cfg.evaluation.golden_qa)
    parser.add_argument("--questions", default=None, help="Comma-separated question IDs (default: all unlabelled)")
    parser.add_argument("--relabel", action="store_true", help="Also revisit questions that already have labels")
    args = parser.parse_args()

    setup_logging()
    questions = load_golden_qa(args.golden)
    wanted = {q.strip() for q in args.questions.split(",")} if args.questions else None

    dense, sparse, embedder = DenseRetriever(), SparseRetriever(), get_embedder()
    pool, max_files = cfg.evaluation.label_pool, cfg.evaluation.label_max_files

    for item in questions:
        if wanted is not None and item["id"] not in wanted:
            continue
        if item.get("expected_sources") and not args.relabel:
            continue

        print("\n" + "=" * 78)
        print(f"[{item['id']}] ({item.get('topic', '')}) {item['question']}")
        print(f"Reference: {item['ground_truth'][:240]}")
        if item.get("expected_sources"):
            print(f"Current labels: {item['expected_sources']}")
        candidates = candidate_files(item["question"], dense, sparse, embedder, pool)[:max_files]
        for n, (key, info) in enumerate(candidates, start=1):
            snippet = " ".join(info["snippet"].split())[:160]
            print(f"{n:>3}. {key.removeprefix('data/docs/')}")
            print(f"     best rank {info['best_rank']}, hits {info['hits']} | {snippet}")

        chosen = ask(candidates)
        if chosen is None:
            break
        if chosen:
            item["expected_sources"] = chosen
            save(args.golden, questions)
            print(f"  Saved: {chosen}")

    labelled = sum(1 for q in questions if q.get("expected_sources"))
    print(f"\n{labelled}/{len(questions)} questions labelled in {args.golden}")


if __name__ == "__main__":
    main()
