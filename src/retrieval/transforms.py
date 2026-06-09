from __future__ import annotations

import logging

from openai import OpenAI

from src.utils.config import get_config

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are an AI/ML expert. Write a short, factual paragraph (3-5 sentences) "
    "that directly answers the question below. Write as if it were an excerpt from "
    "a course textbook. Do not add headings, lists, or citations."
)


def hyde_transform(query: str) -> str:
    cfg = get_config().generation
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=256,
        extra_body={"reasoning_effort": "low"},
    )
    hypothesis = response.choices[0].message.content or query
    logger.debug("HyDE hypothesis: %s", hypothesis[:120])
    return hypothesis
