from __future__ import annotations

from src.embedding.embedder import get_embedder
from src.retrieval.dense import DenseRetriever, RetrievalResult
from src.retrieval.fusion import fuse
from src.retrieval.reranker import rerank
from src.retrieval.sparse import SparseRetriever
from src.retrieval.transforms import hyde_transform
from src.utils.config import get_config

_dense: DenseRetriever | None = None
_sparse: SparseRetriever | None = None


def _get_dense() -> DenseRetriever:
    global _dense
    if _dense is None:
        _dense = DenseRetriever()
    return _dense


def _get_sparse() -> SparseRetriever:
    global _sparse
    if _sparse is None:
        _sparse = SparseRetriever()
    return _sparse


def retrieve(query: str) -> list[RetrievalResult]:
    cfg = get_config()
    embedder = get_embedder()

    embed_text = hyde_transform(query) if cfg.retrieval.hyde.enabled else query
    query_embedding = embedder.embed_query(embed_text)
    dense_results = _get_dense().query(query_embedding)
    sparse_results = _get_sparse().query(query)

    fused = fuse(dense_results, sparse_results)

    if cfg.retrieval.reranker.enabled:
        return rerank(query, fused)
    return fused[: cfg.retrieval.reranker.top_n]
