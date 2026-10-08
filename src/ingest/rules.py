"""
Chunker for the RRC Oil and Gas Statewide Rules (16 TAC Chapter 3).

The rules PDF is one long run of rules, each headed "§3.N Title" and divided
into lettered subsections "(a)", "(b)", ... with deeper (1), (A), (i) levels.
The generic font-size parser finds no reliable headings in it (it mistakes
lines like "500 psi." for headings), so this module splits the text itself:

  rule  (§3.N)  -> section_path[1], e.g. "§3.37 Statewide Spacing Rule (Statewide Rule 37)"
  top-level subsection (a), (b) ... -> section_path[2]

Each chunk's text starts with a one-line header naming the rule and
subsection, so both BM25 and the embedding see "Statewide Rule 37" even when
the passage body never says it, and every citation names the exact rule.

Chunks never cross rule boundaries, and small subsections are merged only
with neighbours from the same rule.

Rule headings are told apart from wrapped cross-references ("§3.16 of this
title (relating to ...)") by requiring an increasing rule number and the
absence of cross-reference wording. Figures printed in the appendix at the end
of the PDF are attached back to the rule that cites them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .chunk import Chunk, ChunkingConfig, _count_tokens, _make_id, _split_with_overlap

RULES_FILE_PREFIX = "statewide_rules"

_PAGE_HEADER_RE = re.compile(r"^\s*As in effect on \d{1,2}/\d{1,2}/\d{4}\.?\s*$")
_RULE_HEAD_RE = re.compile(r"^§\s*3\.(\d+)([A-Z]?)\.?\s+(\S.*)$")   # "§3.66." occurs too
_XREF_RE = re.compile(r"of this title|relating to|\(\w+\)|^\W")
# Top-level subsections sit at indent 0-3; deeper levels are indented further.
_SUBSECTION_RE = re.compile(r"^ {0,3}\(([a-z]{1,2})\)\s+(.*)$")
_SOURCE_NOTE_RE = re.compile(r"^\s*Source Note:")
# A figure printed in the appendix at the end of the PDF; the in-rule reference
# reads "Figure: 16 TAC §3.66(g)(1)   [See Figure at the end of ...]" instead.
_FIGURE_RE = re.compile(r"^Figure: 16 TAC §\s*3\.(\d+[A-Z]?)(\S*)\s*$")


def is_rules_pdf(path: Path) -> bool:
    return path.stem.lower().startswith(RULES_FILE_PREFIX)


@dataclass
class _Line:
    text: str
    page: int


@dataclass
class _Part:
    label: str                  # "" for the rule's preamble, else "(a) Filing requirements"
    page: int
    lines: list[str] = field(default_factory=list)


@dataclass
class Rule:
    number: str                 # "37", "13", "8A"
    title: str
    page: int
    parts: list[_Part] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"§3.{self.number} {self.title} (Statewide Rule {self.number})"


def read_lines(path: Path) -> list[_Line]:
    """Extract text lines with page numbers, dropping the per-page date header.

    Leading whitespace is kept because indentation encodes subsection depth.
    """
    from pypdf import PdfReader

    out: list[_Line] = []
    for page_num, page in enumerate(PdfReader(str(path)).pages, start=1):
        for raw in (page.extract_text() or "").splitlines():
            line = raw.rstrip()
            if line.strip() and not _PAGE_HEADER_RE.match(line):
                out.append(_Line(line, page_num))
    return out


def _subsection_label(letter: str, rest: str) -> str:
    """'(a) Filing requirements.' -> '(a) Filing requirements'; long openings -> '(a)'."""
    m = re.match(r"([A-Z][^.]{2,60})\.(\s|$)", rest.strip())
    return f"({letter}) {m.group(1)}" if m else f"({letter})"


def split_rules(lines: list[_Line]) -> list[Rule]:
    rules: list[Rule] = []
    by_number: dict[str, Rule] = {}
    current: Optional[Rule] = None
    last_number = 0.0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _RULE_HEAD_RE.match(line.text.strip())
        if m:
            number = float(m.group(1)) + (0.5 if m.group(2) else 0.0)
            title = m.group(3).strip()
            if number > last_number and not _XREF_RE.search(title):
                # Titles wrap: take unindented, non-subsection lines right after the heading.
                j = i + 1
                while (j < len(lines) and j <= i + 2 and not lines[j].text.startswith(" ")
                       and not lines[j].text.startswith("(")
                       and not _RULE_HEAD_RE.match(lines[j].text.strip())):
                    title += " " + lines[j].text.strip()
                    j += 1
                current = Rule(m.group(1) + m.group(2), title.strip(), line.page,
                               parts=[_Part("", line.page)])
                rules.append(current)
                by_number[current.number] = current
                last_number = number
                i = j
                continue
        fig = _FIGURE_RE.match(line.text.strip())
        if fig and fig.group(1) in by_number:
            current = by_number[fig.group(1)]
            current.parts.append(_Part(f"Figure §3.{fig.group(1)}{fig.group(2)}", line.page))
        if current is not None:
            rule = current
            sub = None if fig else _SUBSECTION_RE.match(line.text)
            if sub and not _SOURCE_NOTE_RE.match(line.text):
                rule.parts.append(_Part(_subsection_label(sub.group(1), sub.group(2)), line.page))
            rule.parts[-1].lines.append(line.text.strip())
        i += 1
    return rules


def _join(lines: list[str]) -> str:
    text = " ".join(lines)
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)       # "first -class" -> "first-class"
    return re.sub(r"\s+", " ", text).strip()


def chunk_rules(path: Path, config: Optional[ChunkingConfig] = None,
                doc_name: Optional[str] = None) -> list[Chunk]:
    """Chunk the Statewide Rules PDF into rule- and subsection-aware Chunks."""
    config = config or ChunkingConfig()
    doc_name = doc_name or path.stem
    chunks: list[Chunk] = []

    for rule in split_rules(read_lines(path)):
        # Merge small adjacent subsections of this rule only.
        groups: list[_Part] = []
        for part in rule.parts:
            if not part.lines:
                continue
            if groups and _count_tokens(_join(groups[-1].lines)) < config.min_tokens and \
                    _count_tokens(_join(groups[-1].lines + part.lines)) <= config.max_tokens:
                prev = groups[-1]
                label = prev.label or part.label
                if prev.label and part.label:
                    label = f"{prev.label.split(')')[0]})–({part.label[1:].split(')')[0]})"
                groups[-1] = _Part(label, prev.page, prev.lines + part.lines)
            else:
                groups.append(_Part(part.label, part.page, list(part.lines)))

        for index, part in enumerate(groups):
            section_path = ["root", rule.heading] + ([part.label] if part.label else [])
            header = f"16 TAC {rule.heading}" + (f", subsection {part.label}" if part.label else "")
            # The word-based splitter underestimates tokens for long legal words, so
            # shrink its budget until every piece plus header fits the embedder window.
            body, budget = _join(part.lines), config.max_tokens - _count_tokens(header) - 4
            pieces = _split_with_overlap(body, budget, config.overlap_tokens)
            while max(map(_count_tokens, pieces)) > budget and budget > 64:
                budget = int(budget * 0.85)
                pieces = _split_with_overlap(body, budget, config.overlap_tokens)
            for k, piece in enumerate(pieces):
                text = f"[{header}]\n{piece}"
                chunks.append(Chunk(
                    chunk_id=_make_id(doc_name, section_path, index * 1000 + k),
                    doc_name=doc_name,
                    section_path=section_path,
                    page_num=part.page,
                    text=text,
                    token_count=_count_tokens(text),
                ))
    return chunks
