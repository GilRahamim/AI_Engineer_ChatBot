from __future__ import annotations

import json

import pytest

from src.evaluation.retrieval_eval import (
    SourceMatcher,
    dup_rate,
    run_retrieval_eval,
    score_ranking,
    summarize,
)
from src.ingestion.manifest import Manifest
from src.retrieval.dense import RetrievalResult


def _r(source: str, text: str = "t") -> RetrievalResult:
    return RetrievalResult(text=text, source=source, file_type="pdf", chunk_index=0, score=0.0)


def test_score_ranking():
    scores = score_ranking(["x", "a", "y", "b"], expected=[{"a"}, {"b"}, {"c"}], ks=[1, 2, 4])
    assert scores["hit@1"] == 0.0 and scores["hit@2"] == 1.0
    assert scores["recall@2"] == pytest.approx(1 / 3)
    assert scores["recall@4"] == pytest.approx(2 / 3)
    assert scores["precision@4"] == 0.5
    assert scores["mrr"] == 0.5


def test_precision_uses_list_length_when_shorter_than_k():
    assert score_ranking(["a"], [{"a"}], ks=[20])["precision@20"] == 1.0


def test_no_relevant_result_scores_zero_mrr():
    assert score_ranking(["x"], [{"a"}], ks=[1])["mrr"] == 0.0


def test_dup_rate():
    assert dup_rate([_r("s", "a"), _r("s2", "a"), _r("s", "b"), _r("s", "c")]) == 0.25
    assert dup_rate([]) == 0.0


def test_matcher_matches_copies_and_legacy_paths():
    manifest = Manifest()
    manifest.record("data/docs/T/a.ipynb", "h1", 3)
    manifest.record("data/docs/T/Copy of a.ipynb", "h1", 0, duplicate_of="data/docs/T/a.ipynb")
    manifest.record("data/docs/T/b.pdf", "h2", 3)
    matcher = SourceMatcher(manifest)
    # chunk indexed under a legacy Windows-style source, labelled by the copy's name
    assert matcher.file_id("data\\docs\\T\\a.ipynb") in matcher.label_ids("Copy of a.ipynb")
    assert matcher.file_id("data/docs/T/b.pdf") not in matcher.label_ids("a.ipynb")


def test_summarize_averages_only_questions_that_report_a_metric():
    out = summarize([
        {"final": {"dup_rate": 0.5, "mrr": 1.0}},
        {"final": {"dup_rate": 0.0}},  # unlabelled question
    ])
    assert out == {"final": {"dup_rate": 0.25, "mrr": 1.0}}


def test_run_retrieval_eval_end_to_end(cfg, tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(q) for q in [
        {"id": "q1", "question": "kmeans?", "ground_truth": "-", "expected_sources": ["a.pdf"]},
        {"id": "q2", "question": "unanswerable?", "ground_truth": "-"},
    ]))
    stages = {
        "dense": [_r("data/docs/b.pdf"), _r("data/docs/a.pdf")],
        "final": [_r("data/docs/a.pdf", "x"), _r("data/docs/a.pdf", "x")],
    }
    out = run_retrieval_eval(golden, retrieve_fn=lambda q: stages)
    assert out["n_questions_labelled"] == 1
    assert out["retrieval_metrics"]["dense"]["mrr"] == 0.5
    assert out["retrieval_metrics"]["final"]["hit@1"] == 1.0
    assert out["retrieval_metrics"]["final"]["dup_rate"] == 0.5
