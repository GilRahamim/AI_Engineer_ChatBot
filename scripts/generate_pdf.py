"""Generate a structured study PDF for a given topic."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

from src.generation.pipeline import answer
from src.retrieval.pipeline import retrieve
from src.utils.logging import setup_logging


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text)


def _to_latin1(text: str) -> str:
    """Strip unsupported characters for Helvetica (latin-1 range)."""
    # Remove LaTeX math expressions
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text)
    # Remove markdown formatting
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _render_pdf(topic: str, content: str, output_path: Path) -> None:
    from fpdf import FPDF

    L, R, PAGE_W = 22, 22, 210
    TEXT_W = PAGE_W - L - R

    class PDF(FPDF):
        def header(self) -> None:
            pass

        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8, f"Page {self.page_no()}", align="C")
            self.set_text_color(0, 0, 0)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_left_margin(L)
    pdf.set_right_margin(R)

    # ── Title block ──────────────────────────────────────────────
    pdf.set_fill_color(30, 30, 30)
    pdf.rect(0, 0, PAGE_W, 38, style="F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(L, 10)
    pdf.cell(TEXT_W, 10, _to_latin1(f"Study Summary: {topic}"), align="L")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(L, 24)
    pdf.cell(TEXT_W, 6, "AI Engineer Course", align="L")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(46)

    # ── Body ─────────────────────────────────────────────────────
    for line in content.split("\n"):
        line = _strip_citations(line).rstrip()
        pdf.set_x(L)

        if not line:
            pdf.ln(2)

        elif line.startswith("## "):
            # Section header with colored background
            pdf.ln(4)
            title = _to_latin1(line[3:])
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_x(L)
            pdf.multi_cell(TEXT_W, 8, f"  {title}", fill=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)

        elif line.startswith("- "):
            text = _to_latin1(line[2:])
            # Bullet marker
            pdf.set_x(L + 3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(5, 6, "-")
            # Bullet text with hanging indent
            pdf.set_font("Helvetica", "", 11)
            pdf.set_x(L + 8)
            pdf.multi_cell(TEXT_W - 8, 6, text)

        elif re.match(r"^\d+\.", line):
            # Numbered list
            match = re.match(r"^(\d+\.)\s+(.*)", line)
            if match:
                num, text = match.group(1), _to_latin1(match.group(2))
                pdf.set_x(L + 3)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(7, 6, num)
                pdf.set_font("Helvetica", "", 11)
                pdf.set_x(L + 10)
                pdf.multi_cell(TEXT_W - 10, 6, text)

        elif line.startswith("|"):
            # Skip markdown table rows
            pass

        else:
            text = _to_latin1(line)
            if text.strip():
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(TEXT_W, 6, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def generate_pdf(topic: str, output_dir: Path) -> Path:
    print(f"Retrieving course material for: {topic}")
    results = retrieve(f"comprehensive overview of {topic} key concepts algorithms")
    print(f"  {len(results)} chunks retrieved")

    print("Generating summary...")
    summary = answer(f"Create a study summary for the topic: {topic}", results, task="summary")

    output_path = output_dir / f"{topic.replace(' ', '_')}.pdf"
    print(f"Rendering PDF...")
    _render_pdf(topic, summary, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a study PDF for a topic")
    parser.add_argument("topic", help="Topic name (e.g. 'Transformers', 'K-Means')")
    parser.add_argument("--output-dir", default="summaries", help="Output directory (default: summaries/)")
    args = parser.parse_args()

    setup_logging()
    output_path = generate_pdf(args.topic, Path(args.output_dir))
    print(f"\nPDF saved to: {output_path}")


if __name__ == "__main__":
    main()
