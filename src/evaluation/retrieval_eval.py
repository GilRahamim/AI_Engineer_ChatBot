"""Deterministic retrieval metrics against hand-labelled source files — no LLM judge.

Each golden question may carry ``expected_sources``: file names (e.g.
``"5. Transformers.pdf"``) whose content answers it. A retrieved chunk is
relevant when it comes from one of those files. Scores are computed for every
pipeline stage, so a drop can be traced to retrieval, fusion or reranking.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.ingestion.manifest import Manifest, load_manifest
from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config
from src.utils.paths import normalize_key

logger = logging.getLogger(__name__)

# Metrics where a lower value is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"dup_rate"}


def load_golden_qa(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def config_summary() -> dict:
    cfg = get_config()
    return {
        "embedder": cfg.embedding.model,
        "reranker_enabled": cfg.retrieval.reranker.enabled,
        "reranker_model": cfg.retrieval.reranker.model,
        "hyde_enabled": cfg.retrieval.hyde.enabled,
        "dense_top_k": cfg.retrieval.dense_top_k,
        "sparse_top_k": cfg.retrieval.sparse_top_k,
        "fusion_k": cfg.retrieval.fusion.k,
        "fusion_top_n": cfg.retrieval.fusion.top_n,
        "reranker_top_n": cfg.retrieval.reranker.top_n,
        "chunk_size": cfg.ingestion.chunk_size,
        "chunk_overlap": cfg.ingestion.chunk_overlap,
        "notebook_output_max_lines": cfg.ingestion.notebook_output_max_lines,
        "notebook_output_max_chars": cfg.ingestion.notebook_output_max_chars,
    }


def config_hash(summary: dict) -> str:
    s = json.dumps(summary, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _basename(source: str) -> str:
    return source.replace("\\", "/").rsplit("/", 1)[-1].lower()


class SourceMatcher:
    """Maps chunk sources and file-name labels onto a shared file identity.

    Identity is the file's content hash from the manifest, so a label matches
    a byte-identical copy under another name ("Copy of X.ipynb") and survives
    a change of path format. Sources missing from the manifest fall back to
    their file name.
    """

    def __init__(self, manifest: Manifest) -> None:
        self._hash_by_key = {key: rec.hash for key, rec in manifest.files.items()}
        self._hashes_by_name: dict[str, set[str]] = defaultdict(set)
        for key, rec in manifest.files.items():
            self._hashes_by_name[_basename(key)].add(rec.hash)

    def file_id(self, source: str) -> str:
        return self._hash_by_key.get(normalize_key(source)) or _basename(source)

    def label_ids(self, label: str) -> set[str]:
        name = _basename(label)
        if name not in self._hashes_by_name:
            logger.warning("expected source %r is not in the manifest — check the spelling", label)
            return {name}
        return self._hashes_by_name[name]


def score_ranking(
    ranked_ids: list[str],
    expected: list[set[str]],
    ks: list[int],
) -> dict[str, float]:
    """Hit@k, recall@k, precision@k and MRR for one ranked list of file ids.

    ``expected`` holds one id-set per labelled file. Recall counts labelled files
    found in the top k; precision divides by min(k, len(ranked_ids)) so a stage
    that returns 8 results is not penalised at k=20.
    """
    relevant = [any(fid in ids for ids in expected) for fid in ranked_ids]
    scores: dict[str, float] = {}
    for k in ks:
        top = ranked_ids[:k]
        scores[f"hit@{k}"] = float(any(relevant[:k]))
        found = sum(1 for ids in expected if any(fid in ids for fid in top))
        scores[f"recall@{k}"] = found / len(expected)
        scores[f"precision@{k}"] = sum(relevant[:k]) / max(1, min(k, len(ranked_ids)))
    first = next((rank for rank, rel in enumerate(relevant, start=1) if rel), None)
    scores["mrr"] = 1.0 / first if first else 0.0
    return scores


def dup_rate(results: list[RetrievalResult]) -> float:
    """Share of results whose text repeats an earlier result in the same list."""
    if not results:
        return 0.0
    return 1.0 - len({r.text for r in results}) / len(results)


def score_question(
    item: dict,
    stages: dict[str, list[RetrievalResult]],
    matcher: SourceMatcher,
    ks: list[int],
) -> dict[str, dict[str, float]]:
    labels = item.get("expected_sources") or []
    expected = [matcher.label_ids(label) for label in labels]
    scored: dict[str, dict[str, float]] = {}
    for stage, results in stages.items():
        metrics = {"dup_rate": dup_rate(results)}
        if expected:
            metrics.update(score_ranking([matcher.file_id(r.source) for r in results], expected, ks))
        scored[stage] = metrics
    return scored


def summarize(per_question: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Mean of each stage metric over the questions that report it."""
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for scored in per_question:
        for stage, metrics in scored.items():
            for name, value in metrics.items():
                sums[stage][name].append(value)
    return {
        stage: {name: sum(vals) / len(vals) for name, vals in metrics.items()}
        for stage, metrics in sums.items()
    }


def run_retrieval_eval(
    golden_path: Path,
    question_ids: list[str] | None = None,
    retrieve_fn: Callable[[str], dict[str, list[RetrievalResult]]] | None = None,
) -> dict[str, Any]:
    """Retrieval-only eval: no generation, no RAGAS. Runs in a few seconds per question."""
    if retrieve_fn is None:
        from src.retrieval.pipeline import retrieve_stages as retrieve_fn

    cfg = get_config()
    questions = load_golden_qa(golden_path)
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    matcher = SourceMatcher(load_manifest())
    ks = cfg.evaluation.retrieval_ks

    per_question = []
    scored_all = []
    for item in questions:
        logger.info("Retrieving %s: %s", item["id"], item["question"][:60])
        stages = retrieve_fn(item["question"])
        scored = score_question(item, stages, matcher, ks)
        scored_all.append(scored)
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "topic": item.get("topic", ""),
            "expected_sources": item.get("expected_sources", []),
            "retrieved_sources": [r.source for r in stages["final"]],
            "retrieval": scored,
        })

    summary = config_summary()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(summary),
        "config_summary": summary,
        "n_questions_total": len(per_question),
        "n_questions_labelled": sum(1 for q in questions if q.get("expected_sources")),
        "retrieval_metrics": summarize(scored_all),
        "per_question": per_question,
    }
