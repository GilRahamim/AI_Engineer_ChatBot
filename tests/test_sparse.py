from __future__ import annotations

import json

from src.ingestion.chunkers import Chunk
from src.retrieval.sparse import SparseRetriever


def _chunks(*texts):
    return [Chunk(text=t, source=f"s{i}", file_type="pdf", chunk_index=i) for i, t in enumerate(texts)]


def test_add_defers_indexing_until_build(cfg):
    sparse = SparseRetriever()
    sparse.add(_chunks("gradient descent optimizer"))
    sparse.add(_chunks("kmeans clustering centroid"))
    assert sparse.query("clustering") == []  # not indexed yet
    sparse.build()
    assert sparse.query("clustering")[0].text == "kmeans clustering centroid"


def test_load_rebuilds_when_corpus_outgrew_index(cfg):
    sparse = SparseRetriever()
    sparse.add(_chunks("gradient descent optimizer"))
    sparse.build()
    # Simulate a crash: chunks appended to the corpus but never indexed.
    with open(cfg.paths.bm25_index / "corpus.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "kmeans clustering centroid", "source": "s9",
                            "file_type": "pdf", "chunk_index": 0, "metadata": {}}) + "\n")

    reloaded = SparseRetriever()
    assert reloaded.query("clustering")[0].source == "s9"
