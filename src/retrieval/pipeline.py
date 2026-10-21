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


def retrieve_stages(query: str) -> dict[str, list[RetrievalResult]]:
    """Run the pipeline and return every stage's ranking: dense, sparse, fused, final.

    Eval uses the intermediate stages to tell whether a relevant chunk was never
    retrieved or was retrieved and then dropped by fusion/reranking.
    """
    cfg = get_config()
    embedder = get_embedder()

    embed_text = hyde_transform(query) if cfg.retrieval.hyde.enabled else query
    query_embedding = embedder.embed_query(embed_text)
    dense_results = _get_dense().query(query_embedding)
    sparse_results = _get_sparse().query(query)

    fused = fuse(dense_results, sparse_results)

    if cfg.retrieval.reranker.enabled:
        final = rerank(query, fused)
    else:
        final = fused[: cfg.retrieval.reranker.top_n]
    return {"dense": dense_results, "sparse": sparse_results, "fused": fused, "final": final}


def retrieve(query: str) -> list[RetrievalResult]:
    return retrieve_stages(query)["final"]
