from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from src.utils.config import get_config


class FileRecord(BaseModel):
    hash: str
    ingested_at: str
    chunk_count: int


class Manifest(BaseModel):
    files: dict[str, FileRecord] = {}

    def is_ingested(self, path: Path) -> bool:
        key = str(path)
        return key in self.files and self.files[key].hash == _file_hash(path)

    def record(self, path: Path, chunk_count: int) -> None:
        self.files[str(path)] = FileRecord(
            hash=_file_hash(path),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=chunk_count,
        )


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> Manifest:
    path = Path(get_config().paths.manifest)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(**data)
    return Manifest()


def save_manifest(manifest: Manifest) -> None:
    path = Path(get_config().paths.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
