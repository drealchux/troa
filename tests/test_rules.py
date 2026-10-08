"""
Tests for the Statewide Rules chunker (src/ingest/rules.py).

Self-contained: rule text is built from synthetic lines, so no PDF is needed.

Run with: pytest tests/test_rules.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest import rules
from src.ingest.chunk import ChunkingConfig, _count_tokens, _split_with_overlap
from src.ingest.rules import _Line, chunk_rules, is_rules_pdf, split_by_tokens, split_rules


def lines(*texts: str, page: int = 1) -> list[_Line]:
    return [_Line(t, page) for t in texts]


SAMPLE = lines(
    "§3.13 Casing, Cementing, Drilling, Well Control, and Completion Requirements",
    "    (a) General. Operators shall case and cement wells as required.",
    "    (b) Surface casing. Surface casing shall be set below usable water.",
    "§3.14 Plugging",
    "(a) Definitions. The following words have these meanings.",
    "     (1) Nested text referring to §3.13 of this title (relating to casing).",
    "(b) Plugging deadline. Wells shall be plugged within one year.",
    "§3.13 of this title (relating to Casing) applies here.",     # cross-reference, not a heading
    "§3.37 Statewide Spacing Rule",
    "(a) Distance requirements. No well shall be drilled nearer than 1,200 feet.",
)


def test_is_rules_pdf_matches_filename_prefix():
    assert is_rules_pdf(Path("statewide_rules_16tac_ch3.pdf"))
    assert not is_rules_pdf(Path("oga049_drilling_permit_master.pdf"))


def test_split_rules_finds_rules_and_rejects_cross_references():
    found = split_rules(SAMPLE)
    assert [r.number for r in found] == ["13", "14", "37"]
    assert found[2].heading == "§3.37 Statewide Spacing Rule (Statewide Rule 37)"
    # The cross-reference line stays inside Rule 14 instead of starting a rule.
    rule14_text = " ".join(line for p in found[1].parts for line in p.lines)
    assert "of this title (relating to Casing)" in rule14_text


def test_split_rules_labels_top_level_subsections_only():
    rule14 = split_rules(SAMPLE)[1]
    labels = [p.label for p in rule14.parts if p.label]
    assert labels == ["(a) Definitions", "(b) Plugging deadline"]


def test_split_by_tokens_fits_budget_and_overlaps():
    text = " ".join(f"word{i:04d}" for i in range(2000))    # 8-char words: long for the word heuristic
    pieces = split_by_tokens(text, max_tokens=100, overlap_tokens=20)
    assert all(_count_tokens(p) <= 100 for p in pieces)
    assert set(pieces[0].split()) & set(pieces[1].split())  # adjacent pieces share words
    assert " ".join(pieces).count("word1999") >= 1          # nothing dropped at the end


def test_split_by_tokens_never_degenerates_when_overlap_exceeds_budget():
    # The 2026-10-08 failure mode: overlap >= window. Each step must still
    # advance by at least half a piece, so the piece count stays proportional.
    n_words = 3000
    pieces = split_by_tokens(" ".join(["legislative"] * n_words), max_tokens=40, overlap_tokens=64)
    words_per_piece = len(pieces[0].split())
    assert words_per_piece > 1
    assert len(pieces) <= -(-n_words // (words_per_piece // 2)) + 1   # ceil(n / half a piece) + 1
    assert len(pieces) < n_words / 4    # vs. ~n_words pieces when the window crept one word at a time


def test_split_with_overlap_caps_overlap_at_half_the_window():
    text = " ".join(["x"] * 1000)
    pieces = _split_with_overlap(text, max_tokens=60, overlap_tokens=64)
    words_per_piece = int(60 * 0.8)
    # Step is at least half a window, so at most ~2x the non-overlapping count.
    assert len(pieces) <= 2 * (1000 // words_per_piece) + 2


def test_chunk_rules_headers_paths_and_budget(monkeypatch):
    monkeypatch.setattr(rules, "read_lines", lambda path: SAMPLE)
    chunks = chunk_rules(Path("statewide_rules_test.pdf"), ChunkingConfig(min_tokens=1))
    assert chunks
    for c in chunks:
        assert c.doc_name == "statewide_rules_test"
        assert c.section_path[0] == "root" and c.section_path[1].startswith("§3.")
        assert c.text.startswith("[16 TAC §3.")
        assert c.token_count <= 512
    spacing = [c for c in chunks if "Statewide Rule 37" in c.section_path[1]]
    assert spacing and "1,200 feet" in spacing[0].text
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_rules_merges_small_subsections_within_a_rule_only(monkeypatch):
    monkeypatch.setattr(rules, "read_lines", lambda path: SAMPLE)
    chunks = chunk_rules(Path("statewide_rules_test.pdf"))   # default min_tokens=100
    # Every subsection here is tiny, so each rule collapses to one chunk,
    # but rules are never merged with each other.
    assert sorted(c.section_path[1].split()[0] for c in chunks) == ["§3.13", "§3.14", "§3.37"]
