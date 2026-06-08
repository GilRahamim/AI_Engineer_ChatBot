from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    docs: Path = Path("data/docs")
    chroma_db: Path = Path("data/chroma_db")
    bm25_index: Path = Path("data/bm25_index")
    manifest: Path = Path("data/manifest.json")
    prompts: Path = Path("config/prompts")
    eval_results: Path = Path("data/eval/results")


class IngestionConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    supported_extensions: list[str] = [".pdf", ".ipynb"]


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    device: str = "cuda"
    normalize: bool = True


class FusionConfig(BaseModel):
    k: int = 60
    top_n: int = 20


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5


class RetrievalConfig(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    collection_name: str = "course_docs"
    fusion: FusionConfig = FusionConfig()
    reranker: RerankerConfig = RerankerConfig()


class GenerationConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    model: str = "openai/gpt-oss-20b"
    api_key: str = "lm-studio"
    temperature: float = 0.1
    max_tokens: int = 2048
    reasoning_effort: str = "medium"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    ingestion: IngestionConfig = IngestionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    logging: LoggingConfig = LoggingConfig()


_config: Optional[Config] = None


def load_config(path: str = "config/config.yaml") -> Config:
    global _config
    if _config is not None:
        return _config
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _config = Config(**data)
    else:
        _config = Config()
    return _config


def get_config() -> Config:
    return load_config()
