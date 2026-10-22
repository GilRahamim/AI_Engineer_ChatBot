from __future__ import annotations

import json

import pytest

from src.generation import card as card_module
from src.generation.card import ConceptCard, gather_context, ground, make_card, parse_card
from src.generation.card_render import to_markdown
from src.retrieval.dense import RetrievalResult


def _r(source: str, idx: int, text: str | None = None, file_type: str = "pdf") -> RetrievalResult:
    return RetrievalResult(text=text or f"{source}#{idx}", source=source, file_type=file_type,
                           chunk_index=idx, score=0.0)


# --- parsing & grounding -----------------------------------------------------

def test_parse_card_tolerates_think_blocks_fences_and_string_citations():
    raw = '<think>{"not": "this"}</think>```json\n{"title": "X", "definition": {"text": "d", "citations": ["[2]", "3, 4"]}}\n```'
    card = parse_card(raw)
    assert card.definition.citations == [2, 3, 4]


def test_parse_card_rejects_output_without_json():
    with pytest.raises(ValueError):
        parse_card("I cannot help with that.")


def test_ground_drops_uncited_and_out_of_range_items():
    card = ConceptCard.model_validate({
        "title": "X",
        "definition": {"text": "d", "citations": [1, 7]},
        "steps": [{"text": "ok", "citations": [2]}, {"text": "bad", "citations": [9]}, {"text": "none"}],
        "formulas": [{"name": "std", "latex": "a", "standard": True},
                     {"name": "made up", "latex": "b"}],
        "code": {"code": "x = 1"},
        "related": ["Y"],
    })
    grounded, dropped = ground(card, n_excerpts=3)
    assert grounded.definition.citations == [1]
    assert [s.text for s in grounded.steps] == ["ok"]
    assert [f.name for f in grounded.formulas] == ["std"]
    assert grounded.code is None
    assert grounded.related == ["Y"]
    assert dropped == 4
    assert card.steps[1].citations == [9]  # input card is not mutated


def test_ground_marks_card_uncovered_when_nothing_survives():
    card = ConceptCard.model_validate({"title": "X", "definition": {"text": "d"}, "related": ["Y"]})
    grounded, _ = ground(card, n_excerpts=5)
    assert grounded.covered is False


# --- context -----------------------------------------------------------------

def test_gather_context_dedupes_and_records_sections(cfg):
    cfg.card.max_chunks = 3
    calls = []

    def fake_retrieve(query, use_hyde, top_n, file_type):
        calls.append((query, use_hyde, top_n, file_type))
        if file_type == "notebook":
            return {"final": [_r("nb.ipynb", 0, file_type="notebook")]}
        # every text section returns the same chunk plus a copy under another path
        return {"final": [_r("a.pdf", 0, text="shared"), _r("copy/a.pdf", 0, text="shared"),
                          _r("b.pdf", len(calls))]}

    results = gather_context("Convolution", retrieve_fn=fake_retrieve)

    assert calls[0] == ("what is Convolution", False, cfg.card.chunks_per_section, None)
    assert ("Convolution python code example", False, cfg.card.chunks_per_section, "notebook") in calls
    assert len(results) == 3  # capped
    shared = results[0]
    assert shared.source == "a.pdf"
    assert shared.metadata["card_sections"][:3] == ["definition", "mechanism", "formula"]


# --- generation --------------------------------------------------------------

class _FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete_json(self, system, user, schema, max_tokens, structured=True):
        self.prompts.append(user)
        return self.outputs.pop(0)


def _prompt_file(cfg, tmp_path):
    cfg.paths.prompts.mkdir(parents=True, exist_ok=True)
    (cfg.paths.prompts / "card.txt").write_text("Schema: {schema}", encoding="utf-8")


def test_make_card_retries_invalid_json_then_grounds(cfg, tmp_path):
    _prompt_file(cfg, tmp_path)
    good = json.dumps({"title": "Convolution", "definition": {"text": "d", "citations": [1]},
                       "steps": [{"text": "uncited"}]})
    client = _FakeClient(["not json", good])
    result = make_card("Convolution", retrieve_fn=lambda q, **kw: {"final": [_r("a.pdf", 0)]},
                       client=client)
    assert len(client.prompts) == 2 and "rejected" in client.prompts[1]
    assert result.card.definition.text == "d"
    assert result.card.steps == [] and result.dropped_items == 1


def test_make_card_raises_after_retries(cfg, tmp_path):
    _prompt_file(cfg, tmp_path)
    cfg.card.max_retries = 1
    with pytest.raises(RuntimeError):
        make_card("X", retrieve_fn=lambda q, **kw: {"final": [_r("a.pdf", 0)]},
                  client=_FakeClient(["nope", "still nope"]))


def test_make_card_without_context_skips_the_llm(cfg):
    client = _FakeClient([])
    result = make_card("X", retrieve_fn=lambda q, **kw: {"final": []}, client=client)
    assert result.card.covered is False and client.prompts == []


def test_hebrew_concept_is_retrieved_by_english_term(cfg, tmp_path, monkeypatch):
    _prompt_file(cfg, tmp_path)
    import src.retrieval.transforms as transforms
    monkeypatch.setattr(transforms, "english_term", lambda c: "Convolution")
    queries = []

    def fake_retrieve(query, **kw):
        queries.append(query)
        return {"final": [_r("a.pdf", 0)]}

    client = _FakeClient([json.dumps({"title": "t", "definition": {"text": "d", "citations": [1]}})])
    result = make_card("קונבולוציה", retrieve_fn=fake_retrieve, client=client)
    assert all("Convolution" in q for q in queries)
    assert "English term: Convolution" in client.prompts[0]
    assert result.retrieval_concept == "Convolution"


# --- rendering ---------------------------------------------------------------

def _result(card_dict, n_ctx=2):
    card, dropped = ground(ConceptCard.model_validate(card_dict), n_ctx)
    return card_module.CardResult(concept="X", retrieval_concept="X", card=card,
                                  context=[_r(f"data/docs/f{i}.pdf", i) for i in range(1, n_ctx + 1)],
                                  dropped_items=dropped)


def test_markdown_sections_tables_and_sources(cfg):
    md = to_markdown(_result({
        "title": "Convolution — קונבולוציה",
        "definition": {"text": "הגדרה", "citations": [1]},
        "formulas": [{"name": "f", "latex": "$$a+b$$", "citations": [2]}],
        "comparisons": [{"concept": "Pooling", "difference": "a | b", "when_to_choose": "c", "citations": [2]}],
    }), language="hebrew")
    assert "## 📋 הגדרה" in md
    assert "‏[1]" in md  # RTL mark before citations
    assert "$$a+b$$" in md
    assert "a \\| b" in md
    assert '::: {dir="ltr"}' in md and "f2.pdf · chunk 2" in md


def test_markdown_english_has_no_rtl_mark(cfg):
    md = to_markdown(_result({"title": "X", "definition": {"text": "d", "citations": [1]}}),
                     language="english")
    assert "## 📋 Definition" in md and "‏" not in md


def test_markdown_for_uncovered_concept(cfg):
    md = to_markdown(_result({"title": "X", "covered": False}), language="english")
    assert "do not cover" in md


def test_docx_is_rtl_with_ltr_code_and_native_math(cfg, tmp_path):
    pytest.importorskip("pypandoc")
    docx = pytest.importorskip("docx")
    from docx.oxml.ns import qn

    from src.generation.card_render import write_docx

    md = to_markdown(_result({
        "title": "Convolution — קונבולוציה",
        "definition": {"text": "הגדרה", "citations": [1]},
        "formulas": [{"name": "f", "latex": "a^2", "citations": [1]}],
        "comparisons": [{"concept": "P", "difference": "d", "when_to_choose": "w", "citations": [1]}],
        "code": {"code": "x = 1  # הערה", "citations": [2]},
    }), language="hebrew")
    path = write_docx(md, tmp_path / "card.docx", language="hebrew")

    body = docx.Document(str(path)).element.body
    xml = body.xml
    assert "bidiVisual" in xml and "oMath" in xml
    for p in body.iter(qn("w:p")):
        style = p.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
        if style is not None and style.get(qn("w:val")) == "SourceCode":
            assert p.find(f"{qn('w:pPr')}/{qn('w:bidi')}") is None
