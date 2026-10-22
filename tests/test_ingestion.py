from __future__ import annotations

import json
import shutil

import nbformat
import pytest

from src.ingestion import pipeline
from src.ingestion.loaders import _truncate_output, load_notebook
from src.ingestion.manifest import load_manifest
from src.utils.config import PROJECT_ROOT
from src.utils.paths import normalize_key, source_key


# --- paths -------------------------------------------------------------------

def test_legacy_relative_and_absolute_keys_collapse():
    root = str(PROJECT_ROOT).replace("/", "\\")
    rel = "data\\docs\\Topic 1\\a.pdf"
    assert normalize_key(rel) == "data/docs/Topic 1/a.pdf"
    assert normalize_key(f"{root}\\{rel}") == "data/docs/Topic 1/a.pdf"


def test_source_key_same_for_relative_and_absolute_path():
    rel = "data/docs/x.pdf"
    assert source_key(rel) == source_key(PROJECT_ROOT / rel) == rel


def test_source_key_outside_root_stays_absolute(tmp_path):
    assert source_key(tmp_path / "x.pdf") == (tmp_path / "x.pdf").as_posix()


# --- manifest ----------------------------------------------------------------

def test_load_manifest_merges_legacy_duplicate_keys(cfg):
    root = str(PROJECT_ROOT)
    record = {"hash": "h", "ingested_at": "t", "chunk_count": 3}
    cfg.paths.manifest.write_text(json.dumps({"files": {
        "data\\docs\\a.pdf": record,
        f"{root}/data/docs/a.pdf": record,
        "data\\docs\\b.pdf": {**record, "hash": "h2"},
    }}))
    manifest = load_manifest()
    assert set(manifest.files) == {"data/docs/a.pdf", "data/docs/b.pdf"}
    assert manifest.legacy_duplicates == 1


def test_status_new_unchanged_changed(cfg):
    manifest = load_manifest()
    assert manifest.status("k", "h") == "new"
    manifest.record("k", "h", chunk_count=2)
    assert manifest.status("k", "h") == "unchanged"
    assert manifest.status("k", "other") == "changed"
    manifest.record("dup", "h", chunk_count=0, duplicate_of="k")
    assert manifest.status("dup", "other") == "new"  # owns no chunks
    assert manifest.find_by_hash("h") == "k"


# --- notebook loader ---------------------------------------------------------

def test_truncate_output_keeps_head_and_tail():
    text = "\n".join(f"epoch {i}" for i in range(1000))
    out = _truncate_output(text, max_lines=4, max_chars=10_000).splitlines()
    assert out == ["epoch 0", "epoch 1", "... [996 lines omitted] ...", "epoch 998", "epoch 999"]


def test_truncate_output_caps_chars_and_leaves_short_output_alone():
    assert _truncate_output("short", 20, 2000) == "short"
    out = _truncate_output("x" * 5000, 20, 100)
    assert out.startswith("x" * 100) and out.endswith("[output truncated]")


def test_load_notebook_truncates_logs_and_strips_inline_images(cfg, tmp_path):
    cfg.ingestion.notebook_output_max_lines = 4
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell('Perceptron <img src="data:image/png;base64,iVBORw0KGgo=="> rule'),
        nbformat.v4.new_code_cell(
            "train()",
            outputs=[nbformat.v4.new_output("stream", name="stdout",
                                            text="\n".join(f"loss {i}" for i in range(500)))],
        ),
    ]
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, str(path))

    doc = load_notebook(path)
    assert "base64" not in doc.content
    assert "[embedded image]\"> rule" in doc.content
    assert "loss 0" in doc.content and "loss 499" in doc.content
    assert "loss 250" not in doc.content
    assert doc.source == path.as_posix()


# --- pipeline ----------------------------------------------------------------

class _FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class _FakeStore:
    added: list = []

    def __init__(self):
        pass

    def add(self, chunks, embeddings=None):
        _FakeStore.added.extend(c.source for c in chunks)

    def reset(self):
        _FakeStore.added.clear()

    def build(self):
        pass


@pytest.fixture
def fake_pipeline(cfg, monkeypatch):
    _FakeStore.added = []
    monkeypatch.setattr(pipeline, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(pipeline, "DenseRetriever", _FakeStore)
    monkeypatch.setattr(pipeline, "SparseRetriever", _FakeStore)
    cfg.ingestion.min_chunk_size = 1
    cfg.paths.docs.mkdir()
    return cfg


def _write_nb(path, text):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(text)]
    nbformat.write(nb, str(path))


def test_identical_files_are_ingested_once(fake_pipeline):
    docs = fake_pipeline.paths.docs
    _write_nb(docs / "a.ipynb", "same content")
    shutil.copy(docs / "a.ipynb", docs / "Copy of a.ipynb")

    pipeline.ingest_files(pipeline.scan_docs(docs, [".ipynb"]))

    manifest = load_manifest()
    # Which copy is ingested first depends on the platform's path sort order
    # (WindowsPath sorts case-insensitively), so only the pairing is checked.
    a, copy = source_key(docs / "a.ipynb"), source_key(docs / "Copy of a.ipynb")
    dups = {k: r for k, r in manifest.files.items() if r.duplicate_of}
    assert len(dups) == 1
    (dup_key, dup), = dups.items()
    assert {dup_key, dup.duplicate_of} == {a, copy}
    assert dup.chunk_count == 0
    # one store instance for dense + one for sparse, one file each
    assert len(_FakeStore.added) == 2


def test_rerun_skips_unchanged_and_refuses_changed(fake_pipeline):
    docs = fake_pipeline.paths.docs
    _write_nb(docs / "a.ipynb", "v1")
    pipeline.ingest_files([docs / "a.ipynb"])
    _FakeStore.added.clear()

    assert pipeline.ingest_files([docs / "a.ipynb"]) == 0
    _write_nb(docs / "a.ipynb", "v2 — edited")
    assert pipeline.ingest_files([docs / "a.ipynb"]) == 0
    assert _FakeStore.added == []
