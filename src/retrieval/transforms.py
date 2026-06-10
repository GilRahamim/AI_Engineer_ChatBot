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


_CONDENSE_SYSTEM = (
    "Given a conversation and a follow-up question, rewrite the follow-up as a "
    "standalone question that contains all the context needed to answer it on its "
    "own (resolve pronouns and references like 'it', 'that', 'this'). "
    "Output only the rewritten question — no preamble, no explanation. "
    "If the question is already self-contained, output it unchanged. "
    "Keep the original language (Hebrew or English)."
)


def condense_query(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone retrieval query using recent turns.

    ``history`` is a list of {"role": "user"|"assistant", "content": str}.
    Returns ``query`` unchanged when there is no history.
    """
    if not history:
        return query
    cfg = get_config().generation
    convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
    user = f"Conversation:\n{convo}\n\nFollow-up question: {query}\n\nStandalone question:"
    try:
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": _CONDENSE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=128,
            extra_body={"reasoning_effort": "low"},
        )
        rewritten = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - never let condensing break a chat turn
        logger.warning("condense_query failed, using original query: %s", exc)
        return query
    if not rewritten:
        return query
    logger.debug("Condensed query: %s -> %s", query, rewritten)
    return rewritten


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
