from __future__ import annotations

import logging
import re
from typing import Optional

from src.generation.llm_client import LLMClient
from src.utils.config import get_config

logger = logging.getLogger(__name__)

_client: Optional[LLMClient] = None


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client

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


_TRANSLATE_SYSTEM = (
    "Translate the AI/ML concept name below into the standard English technical "
    "term used in courses and papers. Output only the term — no explanation."
)

_HEBREW = re.compile(r"[\u0590-\u05FF]")


def english_term(concept: str) -> str:
    """English technical term for a concept typed in Hebrew; other input is returned as is.

    BM25 cannot match a Hebrew word against the English course material, so card
    retrieval runs on the English term.
    """
    if not _HEBREW.search(concept):
        return concept
    max_tokens = get_config().generation.helper_max_tokens.translate
    try:
        term = _get_client().helper(_TRANSLATE_SYSTEM, concept, max_tokens).strip().strip('"')
    except Exception as exc:  # noqa: BLE001 - fall back to the original wording
        logger.warning("english_term failed, using original concept: %s", exc)
        return concept
    logger.debug("Translated concept: %s -> %s", concept, term)
    return term or concept


def condense_query(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone retrieval query using recent turns.

    ``history`` is a list of {"role": "user"|"assistant", "content": str}.
    Returns ``query`` unchanged when there is no history.
    """
    if not history:
        return query
    max_tokens = get_config().generation.helper_max_tokens.condense
    convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
    user = f"Conversation:\n{convo}\n\nFollow-up question: {query}\n\nStandalone question:"
    try:
        rewritten = _get_client().helper(_CONDENSE_SYSTEM, user, max_tokens).strip()
    except Exception as exc:  # noqa: BLE001 - never let condensing break a chat turn
        logger.warning("condense_query failed, using original query: %s", exc)
        return query
    if not rewritten:
        return query
    logger.debug("Condensed query: %s -> %s", query, rewritten)
    return rewritten


def hyde_transform(query: str) -> str:
    max_tokens = get_config().generation.helper_max_tokens.hyde
    hypothesis = _get_client().helper(_HYDE_SYSTEM, query, max_tokens) or query
    logger.debug("HyDE hypothesis: %s", hypothesis[:120])
    return hypothesis
