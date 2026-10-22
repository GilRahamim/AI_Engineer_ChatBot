"""Concept cards: a fixed-structure study sheet for one concept.

Context is gathered with one retrieval per card section, the LLM fills a JSON
schema, and every item without a valid citation is dropped before rendering —
a fixed template otherwise invites the model to fill empty slots from memory.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.generation.llm_client import LLMClient
from src.generation.thinking import strip_think
from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class _Cited(BaseModel):
    citations: list[int] = Field(default=[], description="Excerpt numbers [n] that support this item")

    @field_validator("citations", mode="before")
    @classmethod
    def _parse_citations(cls, value: object) -> list[int]:
        # Models sometimes emit "[2]" or "2, 3" instead of integers.
        if isinstance(value, (str, int)):
            value = [value]
        out: list[int] = []
        for item in value or []:
            out.extend(int(n) for n in re.findall(r"\d+", str(item)))
        return out


class CitedText(_Cited):
    text: str


class Formula(_Cited):
    name: str
    latex: str = Field(description="LaTeX body without $ delimiters")
    explanation: str = ""
    standard: bool = Field(
        default=False,
        description="True only for a standard textbook formula the excerpts confirm is taught but do not show",
    )


class MetricRow(_Cited):
    task: str
    metrics: str


class ComparisonRow(_Cited):
    concept: str
    difference: str
    when_to_choose: str


class CodeExample(_Cited):
    language: str = "python"
    code: str
    caption: str = ""


class InterviewQA(_Cited):
    question: str
    answer: str


class ConceptCard(BaseModel):
    title: str = Field(description='English term and its translation, e.g. "Convolution — קונבולוציה"')
    covered: bool = Field(default=True, description="False when the excerpts do not cover the concept")
    tldr: Optional[CitedText] = None
    definition: Optional[CitedText] = None
    steps: list[CitedText] = []
    formulas: list[Formula] = []
    applications: list[CitedText] = []
    advantages: list[CitedText] = []
    disadvantages: list[CitedText] = []
    pitfalls: list[CitedText] = []
    metrics: list[MetricRow] = []
    comparisons: list[ComparisonRow] = []
    code: Optional[CodeExample] = None
    interview: list[InterviewQA] = []
    related: list[str] = Field(default=[], description="Names of related course concepts")


class CardResult(BaseModel):
    concept: str
    retrieval_concept: str
    card: ConceptCard
    context: list[RetrievalResult]
    dropped_items: int


# --- context -----------------------------------------------------------------

def gather_context(
    concept: str,
    retrieve_fn: Optional[Callable[..., dict[str, list[RetrievalResult]]]] = None,
) -> list[RetrievalResult]:
    """One retrieval per card section, merged in section order and deduplicated.

    Each chunk's metadata gets ``card_sections`` — the sections it was retrieved
    for — so the prompt can tell the model where each excerpt is likely to help.
    """
    if retrieve_fn is None:
        from src.retrieval.pipeline import retrieve_stages as retrieve_fn

    cfg = get_config().card
    merged: dict[tuple[str, int], RetrievalResult] = {}
    seen_text: dict[str, tuple[str, int]] = {}
    for section in cfg.sections:
        results = retrieve_fn(
            section.query.replace("{concept}", concept),
            use_hyde=cfg.use_hyde,
            top_n=cfg.chunks_per_section,
            file_type=section.file_type,
        )["final"]
        for r in results:
            key = seen_text.get(r.text, (r.source, r.chunk_index))
            if key in merged:
                sections = merged[key].metadata["card_sections"]
                if section.name not in sections:
                    sections.append(section.name)
                continue
            if len(merged) >= cfg.max_chunks:
                continue
            seen_text[r.text] = key
            merged[key] = r.model_copy(
                update={"metadata": {**r.metadata, "card_sections": [section.name]}}
            )
    return list(merged.values())


def format_card_context(results: list[RetrievalResult]) -> str:
    parts = [
        f"[{i}] SOURCE: {Path(r.source).name} | retrieved for: "
        f"{', '.join(r.metadata.get('card_sections', []))}\n{r.text}"
        for i, r in enumerate(results, 1)
    ]
    body = "\n\n---\n\n".join(parts)
    return f"=== COURSE MATERIAL EXCERPTS (reference only) ===\n\n{body}\n\n=== END OF EXCERPTS ==="


# --- parsing & grounding -----------------------------------------------------

def parse_card(raw: str) -> ConceptCard:
    """Parse model output into a card, tolerating think blocks and code fences."""
    text = strip_think(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return ConceptCard.model_validate(json.loads(text[start : end + 1]))


def ground(card: ConceptCard, n_excerpts: int) -> tuple[ConceptCard, int]:
    """Keep only citations to real excerpts and drop items left without one.

    Exempt: standard formulas (flagged as such in the output) and related-concept
    names. Returns the grounded card and the number of items dropped.
    """
    dropped = 0

    def valid(item: _Cited) -> bool:
        item.citations = sorted({n for n in item.citations if 1 <= n <= n_excerpts})
        return bool(item.citations)

    def keep_list(items: list) -> list:
        nonlocal dropped
        kept = [i for i in items if valid(i) or (isinstance(i, Formula) and i.standard)]
        dropped += len(items) - len(kept)
        return kept

    def keep_one(item: Optional[_Cited]) -> Optional[_Cited]:
        nonlocal dropped
        if item is None or valid(item):
            return item
        dropped += 1
        return None

    card = card.model_copy(deep=True)
    for name in ("steps", "formulas", "applications", "advantages", "disadvantages",
                 "pitfalls", "metrics", "comparisons", "interview"):
        setattr(card, name, keep_list(getattr(card, name)))
    card.tldr = keep_one(card.tldr)
    card.definition = keep_one(card.definition)
    card.code = keep_one(card.code)

    has_content = any([card.tldr, card.definition, card.steps, card.formulas, card.code,
                       card.applications, card.comparisons, card.interview])
    if not has_content:
        card.covered = False
    return card, dropped


# --- generation --------------------------------------------------------------

def _load_prompt() -> str:
    path = Path(get_config().paths.prompts) / "card.txt"
    return path.read_text(encoding="utf-8")


def make_card(
    concept: str,
    language: Optional[str] = None,
    retrieve_fn: Optional[Callable[..., dict[str, list[RetrievalResult]]]] = None,
    client: Optional[LLMClient] = None,
) -> CardResult:
    from src.retrieval.transforms import english_term

    cfg = get_config().card
    language = language or cfg.language
    retrieval_concept = english_term(concept) if cfg.translate_concept else concept
    context = gather_context(retrieval_concept, retrieve_fn)

    if not context:
        card = ConceptCard(title=concept, covered=False)
        return CardResult(concept=concept, retrieval_concept=retrieval_concept,
                          card=card, context=[], dropped_items=0)

    schema = ConceptCard.model_json_schema()
    system = _load_prompt().replace("{schema}", json.dumps(schema, ensure_ascii=False))
    user = (
        f"{format_card_context(context)}\n\n"
        f"Concept: {concept}"
        + (f" (English term: {retrieval_concept})" if retrieval_concept != concept else "")
        + f"\nWrite all prose in {language}."
    )
    client = client or LLMClient()

    card: Optional[ConceptCard] = None
    error = ""
    for attempt in range(cfg.max_retries + 1):
        prompt = user if not error else (
            f"{user}\n\nYour previous output was rejected: {error}\n"
            "Output only one JSON object that matches the schema."
        )
        raw = client.complete_json(system, prompt, schema, cfg.max_tokens, cfg.structured_output)
        try:
            card = parse_card(raw)
            break
        except (ValueError, ValidationError) as exc:
            error = str(exc)[:500]
            logger.warning("Card JSON rejected (attempt %d): %s", attempt + 1, error)
    if card is None:
        raise RuntimeError(f"model did not return a valid card: {error}")

    card, dropped = ground(card, len(context))
    if dropped:
        logger.info("Dropped %d uncited card items", dropped)
    return CardResult(concept=concept, retrieval_concept=retrieval_concept,
                      card=card, context=context, dropped_items=dropped)
