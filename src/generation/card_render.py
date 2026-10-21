"""Render a ConceptCard to Markdown (chat) and .docx (study sheet)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.generation.card import CardResult, ConceptCard, _Cited
from src.generation.pipeline import source_label
from src.retrieval.dense import RetrievalResult
from src.utils.config import get_config

logger = logging.getLogger(__name__)

_LABELS = {
    "hebrew": {
        "tldr": "💡 בקצרה",
        "definition": "📋 הגדרה",
        "steps": "⚙️ איך זה עובד",
        "formulas": "📐 נוסחאות",
        "applications": "🎯 שימושים עיקריים",
        "advantages": "✅ יתרונות",
        "disadvantages": "❌ חסרונות / מגבלות",
        "pitfalls": "⚠️ טעויות נפוצות",
        "metrics": "📊 מדדי הערכה מומלצים",
        "comparisons": "🔄 השוואה למושגים קרובים",
        "code": "💻 קוד לדוגמה",
        "interview": "🧠 שאלות ראיון נפוצות",
        "related": "🔗 קשורים ל",
        "sources": "📚 מקורות",
        "metric_cols": ("סוג משימה", "מדדים מומלצים"),
        "comparison_cols": ("מושג", "הבדל מרכזי", "מתי לבחור"),
        "q": "ש:",
        "a": "ת:",
        "standard": "נוסחה סטנדרטית — לא מופיעה בחומר",
        "not_covered": "חומרי הקורס אינם מכסים את המושג הזה.",
    },
    "english": {
        "tldr": "💡 TL;DR",
        "definition": "📋 Definition",
        "steps": "⚙️ How it works",
        "formulas": "📐 Formulas",
        "applications": "🎯 Main applications",
        "advantages": "✅ Advantages",
        "disadvantages": "❌ Disadvantages / limitations",
        "pitfalls": "⚠️ Common pitfalls",
        "metrics": "📊 Recommended evaluation metrics",
        "comparisons": "🔄 Comparison with related concepts",
        "code": "💻 Code example",
        "interview": "🧠 Common interview questions",
        "related": "🔗 Related",
        "sources": "📚 Sources",
        "metric_cols": ("Task type", "Recommended metrics"),
        "comparison_cols": ("Concept", "Key difference", "When to choose"),
        "q": "Q:",
        "a": "A:",
        "standard": "standard formula — not in the course material",
        "not_covered": "The course materials do not cover this concept.",
    },
}

_RTL_LANGUAGES = {"hebrew": "he", "arabic": "ar"}


def _labels(language: str) -> dict:
    return _LABELS.get(language.lower(), _LABELS["english"])


def _citer(rtl: bool):
    # A right-to-left mark keeps "[n]" at the end of a Hebrew sentence that ends
    # in an English term; without it the bidi algorithm moves it mid-line.
    mark = "\u200f" if rtl else ""

    def cite(item: _Cited) -> str:
        return f" {mark}[{', '.join(map(str, item.citations))}]" if item.citations else ""

    return cite


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _latex_body(latex: str) -> str:
    body = latex.strip()
    for open_, close in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if body.startswith(open_) and body.endswith(close) and len(body) > len(open_) + len(close):
            return body[len(open_) : -len(close)].strip()
    return body


def _fence(code: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


def _cited_sources(card: ConceptCard, context: list[RetrievalResult]) -> list[int]:
    cited: set[int] = set()
    for value in card.__dict__.values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, _Cited):
                cited.update(item.citations)
    return sorted(n for n in cited if 1 <= n <= len(context))


def to_markdown(result: CardResult, language: Optional[str] = None) -> str:
    card = result.card
    language = (language or get_config().card.language).lower()
    lb = _labels(language)
    _cite = _citer(language in _RTL_LANGUAGES)
    out: list[str] = [f"# {card.title}", ""]
    if not card.covered:
        out.append(lb["not_covered"])
        return "\n".join(out) + "\n"

    def section(key: str) -> None:
        out.extend([f"## {lb[key]}", ""])

    def bullets(key: str, items: list) -> None:
        if items:
            section(key)
            out.extend(f"- {i.text}{_cite(i)}" for i in items)
            out.append("")

    if card.tldr:
        out.extend([f"> **{lb['tldr']}:** {card.tldr.text}{_cite(card.tldr)}", ""])
    if card.definition:
        section("definition")
        out.extend([f"{card.definition.text}{_cite(card.definition)}", ""])
    if card.steps:
        section("steps")
        out.extend(f"{n}. {s.text}{_cite(s)}" for n, s in enumerate(card.steps, 1))
        out.append("")
    if card.formulas:
        section("formulas")
        for f in card.formulas:
            note = f" *({lb['standard']})*" if f.standard and not f.citations else ""
            desc = f" — {f.explanation}" if f.explanation else ""
            out.extend([f"$${_latex_body(f.latex)}$$", "", f"**{f.name}**{desc}{_cite(f)}{note}", ""])
    bullets("applications", card.applications)
    bullets("advantages", card.advantages)
    bullets("disadvantages", card.disadvantages)
    bullets("pitfalls", card.pitfalls)
    if card.metrics:
        section("metrics")
        a, b = lb["metric_cols"]
        out.extend([f"| {a} | {b} |", "|---|---|"])
        out.extend(f"| {_cell(m.task)} | {_cell(m.metrics)}{_cite(m)} |" for m in card.metrics)
        out.append("")
    if card.comparisons:
        section("comparisons")
        a, b, c = lb["comparison_cols"]
        out.extend([f"| {a} | {b} | {c} |", "|---|---|---|"])
        out.extend(
            f"| {_cell(r.concept)} | {_cell(r.difference)}{_cite(r)} | {_cell(r.when_to_choose)} |"
            for r in card.comparisons
        )
        out.append("")
    if card.code:
        section("code")
        if card.code.caption:
            out.extend([f"{card.code.caption}{_cite(card.code)}", ""])
        fence = _fence(card.code.code)
        out.extend([f"{fence}{card.code.language}", card.code.code.rstrip(), fence, ""])
    if card.interview:
        section("interview")
        for qa in card.interview:
            out.extend([f"**{lb['q']}** {qa.question}", "", f"**{lb['a']}** {qa.answer}{_cite(qa)}", ""])
    if card.related:
        section("related")
        out.extend(f"- {name}" for name in card.related)
        out.append("")

    cited = _cited_sources(card, result.context)
    if cited:
        section("sources")
        # File names are English: keep the list left-to-right inside an RTL document.
        out.append('::: {dir="ltr"}')
        out.extend(f"- [{n}] {source_label(result.context[n - 1])}" for n in cited)
        out.extend([":::", ""])
    return "\n".join(out)


# --- docx --------------------------------------------------------------------

def _w(tag: str) -> str:
    from docx.oxml.ns import qn

    return qn(tag)


# Child order inside w:tblPr / w:tcPr required by the OOXML schema.
_TBLPR_ORDER = ["w:tblStyle", "w:tblpPr", "w:tblOverlap", "w:bidiVisual", "w:tblStyleRowBandSize",
                "w:tblStyleColBandSize", "w:tblW", "w:jc", "w:tblCellSpacing", "w:tblInd",
                "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook"]
_TCPR_ORDER = ["w:cnfStyle", "w:tcW", "w:gridSpan", "w:hMerge", "w:vMerge", "w:tcBorders",
               "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark"]


def _set_child(parent, child, order: list[str]) -> None:
    """Insert ``child`` into ``parent`` at its schema position, replacing any same-tag child."""
    for old in parent.findall(child.tag):
        parent.remove(old)
    rank = {_w(tag): i for i, tag in enumerate(order)}
    pos = rank[child.tag]
    for i, existing in enumerate(parent):
        if rank.get(existing.tag, -1) > pos:
            parent.insert(i, child)
            return
    parent.append(child)


def _make_ltr(paragraph) -> None:
    for el in list(paragraph.iter(_w("w:bidi"))):
        el.getparent().remove(el)
    for el in list(paragraph.iter(_w("w:rtl"))):
        el.getparent().remove(el)


def _style_tables(body, rtl: bool) -> None:
    from docx.oxml import OxmlElement

    for tbl in body.iter(_w("w:tbl")):
        tbl_pr = tbl.find(_w("w:tblPr"))
        if rtl:
            _set_child(tbl_pr, OxmlElement("w:bidiVisual"), _TBLPR_ORDER)
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{side}")
            el.set(_w("w:val"), "single")
            el.set(_w("w:sz"), "4")
            el.set(_w("w:color"), "8EAADB")
            borders.append(el)
        _set_child(tbl_pr, borders, _TBLPR_ORDER)
        first_row = tbl.find(_w("w:tr"))
        for tc in first_row.findall(_w("w:tc")) if first_row is not None else []:
            tc_pr = tc.find(_w("w:tcPr"))
            if tc_pr is None:
                tc_pr = OxmlElement("w:tcPr")
                tc.insert(0, tc_pr)
            shd = OxmlElement("w:shd")
            shd.set(_w("w:val"), "clear")
            shd.set(_w("w:color"), "auto")
            shd.set(_w("w:fill"), "D9E2F3")
            _set_child(tc_pr, shd, _TCPR_ORDER)


def _set_fonts(styles, font: str) -> None:
    """Point every non-code style at ``font`` for Latin and complex scripts (Hebrew)."""
    code_styles = {"SourceCode", "VerbatimChar"}
    for rfonts in styles.iter(_w("w:rFonts")):
        style = rfonts.getparent().getparent()
        if style.tag == _w("w:style") and style.get(_w("w:styleId")) in code_styles:
            continue
        for attr in list(rfonts.attrib):
            if attr.endswith("Theme"):
                del rfonts.attrib[attr]
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(_w(attr), font)


def write_docx(markdown: str, path: Path, language: Optional[str] = None) -> Path:
    import docx
    import pypandoc

    cfg = get_config().card
    language = (language or cfg.language).lower()
    rtl = language in _RTL_LANGUAGES
    args: list[str] = []
    if rtl:
        args += ["-M", "dir=rtl", "-M", f"lang={_RTL_LANGUAGES[language]}"]
    use_reference = cfg.reference_docx is not None and Path(cfg.reference_docx).exists()
    if use_reference:
        args.append(f"--reference-doc={cfg.reference_docx}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_text(markdown, "docx", format="markdown", outputfile=str(path), extra_args=args)

    # Pandoc marks every paragraph RTL; code and display math must stay LTR,
    # and tables need bidiVisual so the first column sits on the right.
    document = docx.Document(str(path))
    body = document.element.body
    for p in body.iter(_w("w:p")):
        style = p.find(f"{_w('w:pPr')}/{_w('w:pStyle')}")
        is_code = style is not None and style.get(_w("w:val")) == "SourceCode"
        is_math = p.find(".//" + _w("m:oMathPara")) is not None
        if is_code or is_math:
            _make_ltr(p)
    _style_tables(body, rtl)
    if not use_reference:
        _set_fonts(document.styles.element, cfg.docx_font)
    document.save(str(path))
    return path


# --- saving ------------------------------------------------------------------

def _slug(concept: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", concept.strip()).strip("_") or "card"


def save_card(
    result: CardResult,
    language: Optional[str] = None,
    out_dir: Optional[Path] = None,
    docx: Optional[bool] = None,
) -> dict[str, Path]:
    """Write the card as .md, .json (card + sources) and, if enabled, .docx."""
    cfg = get_config()
    language = language or cfg.card.language
    out_dir = Path(out_dir or cfg.paths.cards)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / _slug(result.concept)

    markdown = to_markdown(result, language)
    paths = {"md": stem.with_suffix(".md"), "json": stem.with_suffix(".json")}
    paths["md"].write_text(markdown, encoding="utf-8")
    payload = {
        "concept": result.concept,
        "retrieval_concept": result.retrieval_concept,
        "language": language,
        "dropped_items": result.dropped_items,
        "card": result.card.model_dump(),
        "sources": [
            {"n": n, "source": r.source, "chunk_index": r.chunk_index,
             "sections": r.metadata.get("card_sections", [])}
            for n, r in enumerate(result.context, 1)
        ],
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if cfg.card.docx if docx is None else docx:
        paths["docx"] = write_docx(markdown, stem.with_suffix(".docx"), language)
    return paths
