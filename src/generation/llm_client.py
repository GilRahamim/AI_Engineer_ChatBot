from __future__ import annotations

import logging

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

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"reasoning_effort": self._reasoning_effort},
        )
        return response.choices[0].message.content or ""
