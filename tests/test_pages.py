from __future__ import annotations

import fitz
import nbformat
import pymupdf4llm
import pytest

from src.generation.pipeline import source_label
from src.ingestion import loaders
from src.ingestion.chunkers import chunk_document
from src.ingestion.loaders import Document, Segment, _join, load_notebook, load_pdf
from src.retrieval.dense import RetrievalResult

_PAGES = [
    "Convolutional Networks\n\n" + "A convolution slides a kernel over the input image. " * 12,
    "Pooling\n\n" + "Max pooling keeps the maximum value of each window. " * 12,
    "",  # blank page
    "Backpropagation\n\n" + "Gradients flow backwards through the chain rule. " * 12,
]


@pytest.fixture
def deck(tmp_path):
    doc = fitz.open()
    for text in _PAGES:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=11)
    path = tmp_path / "deck.pdf"
    doc.save(str(path))
    return path


def test_join_records_part_offsets_and_merges_same_number():
    content, segments = _join([(1, "ab"), (1, "cd"), (3, "ef")], "\n\n")
    assert content == "ab\n\ncd\n\nef"
    assert segments == [Segment(start=0, number=1), Segment(start=8, number=3)]


def test_pdf_text_is_unchanged_by_page_tracking(deck):
    # Guards the claim that per-page output joined with "" equals the whole-document
    # markdown — if a pymupdf4llm upgrade breaks it, chunk text would silently change.
    assert load_pdf(deck).content == pymupdf4llm.to_markdown(str(deck))


def test_pdf_chunks_carry_their_pages(cfg, deck):
    cfg.ingestion.chunk_size = 300
    cfg.ingestion.chunk_overlap = 0
    cfg.ingestion.min_chunk_size = 1
    doc = load_pdf(deck)
    assert doc.segment_kind == "page" and [s.number for s in doc.segments] == [1, 2, 3, 4]

    chunks = chunk_document(doc)
    by_page = {}
    for c in chunks:
        assert 1 <= c.metadata["page"] <= c.metadata["page_end"] <= 4
        by_page.setdefault(c.metadata["page"], []).append(c.text)
    assert any("convolution" in t for t in by_page[1])
    assert any("Max pooling" in t for t in by_page[2])
    assert any("chain rule" in t for t in by_page[4])
    assert 3 not in by_page  # blank page owns no chunk start


def test_chunk_spanning_a_page_break_gets_a_range(cfg):
    cfg.ingestion.chunk_size = 1000
    cfg.ingestion.min_chunk_size = 1
    doc = Document(content="first page text. second page text.", source="s", file_type="pdf",
                   segment_kind="page",
                   segments=[Segment(start=0, number=1), Segment(start=17, number=2)])
    (chunk,) = chunk_document(doc)
    assert (chunk.metadata["page"], chunk.metadata["page_end"]) == (1, 2)


def test_notebook_chunks_carry_cell_numbers(cfg, tmp_path):
    cfg.ingestion.chunk_size = 120
    cfg.ingestion.chunk_overlap = 0
    cfg.ingestion.min_chunk_size = 1
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Intro\n\n" + "Perceptrons learn a linear boundary. " * 3),
        nbformat.v4.new_raw_cell("raw cell is skipped"),
        nbformat.v4.new_code_cell("model.fit(X, y)", outputs=[
            nbformat.v4.new_output("stream", name="stdout", text="accuracy 0.93")]),
    ]
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, str(path))

    doc = load_notebook(path)
    assert doc.segment_kind == "cell"
    assert [s.number for s in doc.segments] == [1, 3]  # raw cell 2 has no text; code+output share 3
    cells = {c.metadata["cell"] for c in chunk_document(doc)}
    assert cells == {1, 3}


def test_ocr_pages_are_tracked(monkeypatch, deck):
    class FakeReader:
        def readtext(self, img, detail=0):
            return ["scanned", "words"]

    monkeypatch.setattr(loaders, "_get_ocr_reader", lambda: FakeReader())
    content, segments = loaders._ocr_pdf(deck)
    assert content.startswith("[Page 1]\nscanned words")
    assert [s.number for s in segments] == [1, 2, 3, 4]


@pytest.mark.parametrize("metadata, expected", [
    ({"page": 3, "page_end": 3}, "deck.pdf · p.3"),
    ({"page": 3, "page_end": 4}, "deck.pdf · pp.3–4"),
    ({"cell": 12, "cell_end": 14}, "deck.pdf · cells 12–14"),
    ({"cell": 5}, "deck.pdf · cell 5"),
    ({}, "deck.pdf · chunk 7"),  # chunk indexed before page metadata
])
def test_source_label(metadata, expected):
    r = RetrievalResult(text="t", source="data/docs/deck.pdf", file_type="pdf", chunk_index=7,
                        score=0.0, metadata=metadata)
    assert source_label(r) == expected
