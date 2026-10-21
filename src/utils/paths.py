from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import PROJECT_ROOT


def normalize_key(path_str: str) -> str:
    """Map a path string from any earlier format onto the canonical source key.

    Accepts Windows or POSIX separators, relative or absolute. Paths under the
    project root become root-relative; anything else is kept absolute. Earlier
    manifests and indexed chunks stored keys like ``data\\docs\\x.pdf`` and
    ``C:\\...\\data\\docs\\x.pdf`` for the same file — both map to ``data/docs/x.pdf``.
    """
    posix = path_str.replace("\\", "/")
    root = PROJECT_ROOT.as_posix().rstrip("/") + "/"
    if posix.lower().startswith(root.lower()):
        return posix[len(root):]
    return posix


def source_key(path: Path) -> str:
    """Canonical identifier for a course file, independent of OS and working directory.

    Used as the manifest key and as the ``source`` of every chunk, so the same
    file always gets the same key no matter how its path was spelled.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return normalize_key(os.path.abspath(p))
