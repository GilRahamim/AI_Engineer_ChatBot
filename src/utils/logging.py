from __future__ import annotations

import logging
import sys

from src.utils.config import get_config


def setup_logging() -> None:
    cfg = get_config()
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
