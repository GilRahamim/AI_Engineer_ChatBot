"""Model-agnostic control of reasoning ("thinking") and removal of leaked reasoning text.

Reasoning models differ in how thinking is switched and where it ends up:
GPT-OSS takes a ``reasoning_effort`` field, Qwen-style models take a chat
template flag or a /think | /no_think switch in the prompt, and some server
setups return the reasoning inline as <think>...</think> in the answer text.
"""
from __future__ import annotations

import re

from src.utils.config import GenerationConfig

_OPEN, _CLOSE = "<think>", "</think>"
_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def thinking_body(cfg: GenerationConfig, enabled: bool) -> dict:
    """Request-body fields that switch thinking on or off for the configured style."""
    style = cfg.thinking.style
    if style == "reasoning_effort":
        # GPT-OSS cannot turn reasoning off; "off" means the lowest effort.
        return {"reasoning_effort": cfg.reasoning_effort if enabled else cfg.thinking.helper_effort}
    if style == "chat_template_kwargs":
        return {"chat_template_kwargs": {"enable_thinking": enabled}}
    return {}


def apply_soft_switch(messages: list[dict], cfg: GenerationConfig, enabled: bool) -> list[dict]:
    """Append the /think | /no_think switch to the last user message (style "soft_switch")."""
    if cfg.thinking.style != "soft_switch":
        return messages
    switch = cfg.thinking.soft_switch_on if enabled else cfg.thinking.soft_switch_off
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] == "user":
            m["content"] = f"{m['content']} {switch}"
            break
    return out


def strip_think(text: str) -> str:
    """Remove reasoning that leaked into the answer text. Text without tags is returned as is.

    Handles complete <think>…</think> blocks, an orphan </think> (the chat template
    put the opening tag in the prompt) and an unterminated <think> (cut off by
    max_tokens, so there is no answer after it).
    """
    if _OPEN not in text and _CLOSE not in text:
        return text
    text = _BLOCK.sub("", text)
    if _CLOSE in text:
        text = text.rsplit(_CLOSE, 1)[1]
    if _OPEN in text:
        text = text.split(_OPEN, 1)[0]
    return text.strip()


def _partial_tag(buf: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ``buf`` ends with."""
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class ThinkStreamFilter:
    """Drop <think>…</think> spans from a token stream whose tags may split across deltas.

    An orphan </think> cannot be detected while streaming without holding back
    the whole answer; callers that buffer the full text should also run
    ``strip_think`` on it.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._started = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            tag = _CLOSE if self._in_think else _OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                keep = _partial_tag(self._buf, tag)
                if not self._in_think:
                    out.append(self._buf[: len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep :]
                break
            if not self._in_think:
                out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(tag) :]
            self._in_think = not self._in_think
        return self._emit("".join(out))

    def flush(self) -> str:
        rest = "" if self._in_think else self._buf
        self._buf = ""
        return self._emit(rest)

    def _emit(self, text: str) -> str:
        # Drop the blank lines a model leaves between its reasoning and the answer.
        if not self._started:
            text = text.lstrip()
            self._started = bool(text)
        return text
