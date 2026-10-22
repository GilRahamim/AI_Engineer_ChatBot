from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from openai import BadRequestError, OpenAI

from src.generation.thinking import ThinkStreamFilter, apply_soft_switch, strip_think, thinking_body
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client. Every request goes through ``_request`` so sampling,
    thinking control and <think> stripping behave the same for all calls."""

    def __init__(self) -> None:
<<<<<<< HEAD
        cfg = get_config().generation
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._reasoning_effort = cfg.reasoning_effort
=======
        self._cfg = get_config().generation
        self._client = OpenAI(base_url=self._cfg.base_url, api_key=self._cfg.api_key)

    def max_tokens_for(self, task: str) -> int:
        return self._cfg.task_max_tokens.get(task, self._cfg.max_tokens)
>>>>>>> 025436f995d34a8e26537aec2830c857ab7c2118

    def _build_messages(
        self, system: str, user: str, history: Optional[list[dict]]
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages

    def _request(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        thinking: bool,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        cfg = self._cfg
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": apply_soft_switch(messages, cfg, thinking),
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.presence_penalty is not None:
            kwargs["presence_penalty"] = cfg.presence_penalty
        # Not part of the OpenAI schema; local servers read them from the body.
        body = {name: getattr(cfg, name) for name in ("top_k", "min_p") if getattr(cfg, name) is not None}
        body.update(thinking_body(cfg, thinking))
        if body:
            kwargs["extra_body"] = body
        return kwargs

    def _clean(self, text: Optional[str]) -> str:
        text = text or ""
        return strip_think(text) if self._cfg.thinking.strip_tags else text

    def complete(
        self,
        system: str,
        user: str,
        history: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self._client.chat.completions.create(
            **self._request(
                self._build_messages(system, user, history),
                max_tokens=max_tokens or self._cfg.max_tokens,
                thinking=True,
            )
        )
        return self._clean(response.choices[0].message.content)

    def helper(self, system: str, user: str, max_tokens: int) -> str:
        """Short auxiliary call (HyDE, follow-up condensing, term translation).

        Thinking is off unless ``thinking.helper_thinking`` — with a small
        max_tokens, reasoning would use up the budget and leave no answer.
        """
        response = self._client.chat.completions.create(
            **self._request(
                self._build_messages(system, user, None),
                max_tokens=max_tokens,
                thinking=self._cfg.thinking.helper_thinking,
                temperature=self._cfg.helper_temperature,
            )
        )
        return self._clean(response.choices[0].message.content)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
        structured: bool = True,
    ) -> str:
        """Complete with the output constrained to ``schema`` when the server supports it.

        Falls back to an unconstrained call if the server rejects ``response_format``;
        the caller parses and validates the text either way.
        """
        kwargs = self._request(
            self._build_messages(system, user, None), max_tokens=max_tokens, thinking=True
        )
        if structured:
            try:
                response = self._client.chat.completions.create(
                    **kwargs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "output", "schema": schema, "strict": True},
                    },
                )
                return self._clean(response.choices[0].message.content)
            except BadRequestError as exc:
                logger.warning("Server rejected response_format (%s) — retrying unconstrained", exc)
        response = self._client.chat.completions.create(**kwargs)
        return self._clean(response.choices[0].message.content)

    def stream(
        self,
        system: str,
        user: str,
        history: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield answer text deltas as they arrive. Reasoning deltas are ignored."""
        response = self._client.chat.completions.create(
            **self._request(
                self._build_messages(system, user, history),
                max_tokens=max_tokens or self._cfg.max_tokens,
                thinking=True,
            ),
            stream=True,
        )
        think_filter = ThinkStreamFilter() if self._cfg.thinking.strip_tags else None
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta and think_filter:
                delta = think_filter.feed(delta)
            if delta:
                yield delta
        if think_filter:
            tail = think_filter.flush()
            if tail:
                yield tail
