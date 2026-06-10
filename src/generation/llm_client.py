from __future__ import annotations

import logging
from typing import Iterator, Optional

from openai import OpenAI

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        cfg = get_config().generation
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._reasoning_effort = cfg.reasoning_effort

    def _build_messages(
        self, system: str, user: str, history: Optional[list[dict]]
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages

    def complete(
        self, system: str, user: str, history: Optional[list[dict]] = None
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(system, user, history),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"reasoning_effort": self._reasoning_effort},
        )
        return response.choices[0].message.content or ""

    def stream(
        self, system: str, user: str, history: Optional[list[dict]] = None
    ) -> Iterator[str]:
        """Yield answer text deltas as they arrive. Reasoning deltas are ignored."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(system, user, history),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"reasoning_effort": self._reasoning_effort},
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
