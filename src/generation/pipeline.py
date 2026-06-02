from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.generation.llm_client import LLMClient
from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config

_client: Optional[LLMClient] = None


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _load_prompt(name: str) -> str:
    path = Path(get_config().paths.prompts) / f"{name}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _format_context(results: list[RetrievalResult]) -> str:
    parts = [
        f"[{i}] {Path(r.source).name}\n{r.text}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n---\n\n".join(parts)


def answer(query: str, results: list[RetrievalResult], task: str = "default") -> str:
    system = _load_prompt(task) or _load_prompt("system")
    context = _format_context(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    return _get_client().complete(system=system, user=user_message)
