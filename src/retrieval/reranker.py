from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config

logger = logging.getLogger(__name__)

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        cfg = get_config()
        logger.info("Loading reranker: %s", cfg.retrieval.reranker.model)
        _reranker = CrossEncoder(cfg.retrieval.reranker.model, device="cuda")
    return _reranker


def rerank(query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
    if not candidates:
        return []
    cfg = get_config()
    reranker = get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    top_n = cfg.retrieval.reranker.top_n
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_n]
    return [r for r, _ in ranked]
