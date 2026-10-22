from __future__ import annotations

from typing import Optional

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


def retrieve_stages(
    query: str,
    use_hyde: Optional[bool] = None,
    top_n: Optional[int] = None,
    file_type: Optional[str] = None,
) -> dict[str, list[RetrievalResult]]:
    """Run the pipeline and return every stage's ranking: dense, sparse, fused, final.

    Eval uses the intermediate stages to tell whether a relevant chunk was never
    retrieved or was retrieved and then dropped by fusion/reranking. The optional
    arguments override config for one call (None keeps the configured behaviour);
    ``file_type`` restricts results to one kind of source, e.g. "notebook".
    """
    cfg = get_config()
    embedder = get_embedder()
    if use_hyde is None:
        use_hyde = cfg.retrieval.hyde.enabled
    top_n = top_n or cfg.retrieval.reranker.top_n

    embed_text = hyde_transform(query) if use_hyde else query
    query_embedding = embedder.embed_query(embed_text)
    where = {"file_type": file_type} if file_type else None
    dense_results = _get_dense().query(query_embedding, where=where)
    sparse_results = _get_sparse().query(query)
    if file_type:
        sparse_results = [r for r in sparse_results if r.file_type == file_type]

    fused = fuse(dense_results, sparse_results)

    if cfg.retrieval.reranker.enabled:
        final = rerank(query, fused, top_n=top_n)
    else:
        final = fused[:top_n]
    return {"dense": dense_results, "sparse": sparse_results, "fused": fused, "final": final}


def retrieve(query: str) -> list[RetrievalResult]:
    return retrieve_stages(query)["final"]
