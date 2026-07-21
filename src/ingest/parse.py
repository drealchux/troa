"""
PDF parser for the TROA RAG corpus.

Extracts text blocks with font-size metadata using pypdf's visitor API.
Font sizes are used downstream by the chunker to detect section headings.
Falls back to line-based extraction when the visitor API fails (some PDFs
use unusual encoding that trips the visitor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from pypdf import PdfReader


@dataclass
class TextBlock:
    text: str
    font_size: float
    page_num: int


@dataclass
class ParsedPage:
    page_num: int
    blocks: list[TextBlock]

    @property
    def text(self) -> str:
        return " ".join(b.text for b in self.blocks if b.text.strip())


@dataclass
class ParsedDocument:
    path: Path
    doc_name: str
    pages: list[ParsedPage]

    @property
    def all_blocks(self) -> list[TextBlock]:
        return [b for page in self.pages for b in page.blocks]


# Heading detection patterns for RRC documents (font-size is primary signal;
# these patterns catch headings when font-size data is unavailable).
_HEADING_RE = [
    re.compile(r'^\d+(\.\d+)*\s{1,4}\S'),    # "1.2 Section title"
    re.compile(r'^[A-Z][A-Z0-9\-\.]*(?:\s+[A-Z0-9\-\.]+){1,10}$'),  # "ALL CAPS HEADING" or "RULE 37" (≥2 words required)
    re.compile(r'^[A-Z][a-zA-Z\s]{2,60}:$'),      # "Some heading:"
]


def _is_likely_heading(text: str, font_size: float, median_size: float) -> bool:
    text = text.strip()
    if not text or len(text) > 120:
        return False
    # Primary signal: font size significantly above the page median
    if font_size > 0 and median_size > 0 and font_size >= median_size * 1.15:
        return True
    # Fallback: structural text patterns
    return any(p.match(text) for p in _HEADING_RE)


def _extract_blocks(pdf_page, page_num: int) -> list[TextBlock]:
    """Extract text blocks with font sizes via pypdf visitor API."""
    collected: list[tuple[str, float]] = []

    def visitor(text: str, cm, tm, font_dict, font_size: float) -> None:  # noqa: ANN001
        text = text.strip()
        if text:
            collected.append((text, float(font_size or 0.0)))

    try:
        pdf_page.extract_text(visitor_text=visitor)
    except Exception:
        # Visitor fails on some PDFs (encrypted content, unusual encodings).
        # Fall back to plain text extraction; font sizes will all be 0.
        plain = pdf_page.extract_text() or ""
        for line in plain.splitlines():
            if line.strip():
                collected.append((line.strip(), 0.0))

    return [TextBlock(text=t, font_size=s, page_num=page_num) for t, s in collected]


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse a PDF into pages of TextBlocks. Raises on unreadable files."""
    reader = PdfReader(str(path))
    pages = []
    for i, pdf_page in enumerate(reader.pages):
        blocks = _extract_blocks(pdf_page, page_num=i + 1)
        pages.append(ParsedPage(page_num=i + 1, blocks=blocks))
    return ParsedDocument(path=path, doc_name=path.stem, pages=pages)
