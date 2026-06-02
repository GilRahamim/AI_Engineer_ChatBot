from __future__ import annotations

from src.embedding.embedder import get_embedder
from src.retrieval.dense import DenseRetriever, RetrievalResult
from src.utils.config import get_config


def retrieve(query: str) -> list[RetrievalResult]:
    cfg = get_config()
    embedder = get_embedder()
    retriever = DenseRetriever()
    query_embedding = embedder.embed_query(query)
    return retriever.query(query_embedding, top_k=cfg.retrieval.final_top_k)
