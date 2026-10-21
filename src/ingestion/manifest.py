from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from src.utils.config import get_config
from src.utils.paths import normalize_key

logger = logging.getLogger(__name__)

FileStatus = Literal["new", "unchanged", "changed"]


class FileRecord(BaseModel):
    hash: str
    ingested_at: str
    chunk_count: int
    # Set when the file was skipped because identical content was already
    # ingested under another key; such a record owns no chunks.
    duplicate_of: Optional[str] = None


class Manifest(BaseModel):
    files: dict[str, FileRecord] = {}
    # Number of files that earlier manifests recorded under two different path
    # spellings — each such file has two copies of its chunks in the index.
    legacy_duplicates: int = 0

    def status(self, key: str, file_hash: str) -> FileStatus:
        record = self.files.get(key)
        if record is None:
            return "new"
        if record.hash == file_hash:
            return "unchanged"
        # A changed duplicate has no chunks of its own, so ingesting it modifies nothing.
        return "new" if record.duplicate_of else "changed"

    def find_by_hash(self, file_hash: str) -> Optional[str]:
        """Key of the file whose chunks hold this content, if it was already ingested."""
        for key, record in self.files.items():
            if record.hash == file_hash and record.duplicate_of is None:
                return key
        return None

    def record(
        self,
        key: str,
        file_hash: str,
        chunk_count: int,
        duplicate_of: Optional[str] = None,
    ) -> None:
        self.files[key] = FileRecord(
            hash=file_hash,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=chunk_count,
            duplicate_of=duplicate_of,
        )


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> Manifest:
    path = Path(get_config().paths.manifest)
    if not path.exists():
        return Manifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, FileRecord] = {}
    collisions = 0
    for raw_key, raw_record in data.get("files", {}).items():
        key = normalize_key(raw_key)
        if key in files:
            collisions += 1
        files[key] = FileRecord(**raw_record)
    if collisions:
        logger.warning(
            "Manifest recorded %d files under two different paths — the index holds "
            "duplicate chunks for them. Run `python scripts/ingest.py --rebuild`.",
            collisions,
        )
    return Manifest(files=files, legacy_duplicates=collisions)


def save_manifest(manifest: Manifest) -> None:
    path = Path(get_config().paths.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2, exclude={"legacy_duplicates"}), encoding="utf-8")
