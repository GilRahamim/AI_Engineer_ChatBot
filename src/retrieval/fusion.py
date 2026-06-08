from __future__ import annotations

from collections import defaultdict

from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_n: int = 20,
) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(float)
    all_results: dict[str, RetrievalResult] = {}

    for ranked_list in result_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = _chunk_id(result)
            scores[cid] += 1.0 / (k + rank)
            if cid not in all_results:
                all_results[cid] = result

    merged = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [all_results[cid] for cid, _ in merged]


def fuse(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
) -> list[RetrievalResult]:
    cfg = get_config()
    return reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=cfg.retrieval.fusion.k,
        top_n=cfg.retrieval.fusion.top_n,
    )


def _chunk_id(result: RetrievalResult) -> str:
    return f"{result.source}::{result.chunk_index}"
