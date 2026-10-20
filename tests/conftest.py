from __future__ import annotations

import pytest

import src.utils.config as config_module
from src.utils.config import Config, PathsConfig


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A Config whose data paths all live under tmp_path."""
    config = Config(
        paths=PathsConfig(
            docs=tmp_path / "docs",
            chroma_db=tmp_path / "chroma_db",
            bm25_index=tmp_path / "bm25_index",
            manifest=tmp_path / "manifest.json",
            prompts=tmp_path / "prompts",
            eval_results=tmp_path / "results",
        )
    )
    monkeypatch.setattr(config_module, "_config", config)
    return config
