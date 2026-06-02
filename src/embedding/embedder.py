from __future__ import annotations

import logging
from typing import Optional

from src.utils.config import get_config

logger = logging.getLogger(__name__)

_embedder: Optional["Embedder"] = None


class Embedder:
    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel

        cfg = get_config().embedding
        logger.info("Loading embedder: %s on %s", cfg.model, cfg.device)
        self._model = BGEM3FlagModel(cfg.model, use_fp16=True, device=cfg.device)
        self._batch_size = cfg.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=512,
            return_dense=True,
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
