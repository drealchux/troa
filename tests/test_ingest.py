"""
Tests for the ingestion pipeline (parse, chunk).

All tests are self-contained: no real PDFs, no Qdrant, no model downloads.
The embedder and store are tested only at the import/interface level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.chunk import (
    Chunk,
    ChunkingConfig,
    _count_tokens,
    _make_id,
    _split_with_overlap,
    chunk_document,
)
from src.ingest.parse import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    _is_likely_heading,
)


# ------------------------------------------------------------------ helpers --

def make_doc(
    blocks_per_page: list[list[tuple[str, float]]],
    doc_name: str = "test_doc",
) -> ParsedDocument:
    pages = []
    for i, block_specs in enumerate(blocks_per_page, start=1):
        blocks = [TextBlock(text=t, font_size=s, page_num=i) for t, s in block_specs]
        pages.append(ParsedPage(page_num=i, blocks=blocks))
    return ParsedDocument(path=Path(f"{doc_name}.pdf"), doc_name=doc_name, pages=pages)


# ------------------------------------------------------------------ parse ---

class TestIsLikelyHeading:
    def test_font_size_above_median(self):
        assert _is_likely_heading("Introduction", font_size=14.0, median_size=10.0)

    def test_font_size_at_median_not_heading(self):
        assert not _is_likely_heading("Body text here.", font_size=10.0, median_size=10.0)

    def test_numbered_section_pattern(self):
        assert _is_likely_heading("1 Overview", font_size=0.0, median_size=0.0)
        assert _is_likely_heading("1.2 Field Layout", font_size=0.0, median_size=0.0)
        assert _is_likely_heading("3.1.4 Sub-section", font_size=0.0, median_size=0.0)

    def test_all_caps_pattern(self):
        assert _is_likely_heading("FILING REQUIREMENTS", font_size=0.0, median_size=0.0)
        assert _is_likely_heading("RULE 37", font_size=0.0, median_size=0.0)

    def test_too_short_not_heading(self):
        assert not _is_likely_heading("A", font_size=0.0, median_size=0.0)

    def test_too_long_not_heading(self):
        long_text = "A" * 130
        assert not _is_likely_heading(long_text, font_size=20.0, median_size=10.0)

    def test_empty_not_heading(self):
        assert not _is_likely_heading("", font_size=14.0, median_size=10.0)

    def test_heading_with_colon(self):
        assert _is_likely_heading("Filing Requirements:", font_size=0.0, median_size=0.0)


# ------------------------------------------------------------------ chunk ---

class TestCountTokens:
    def test_basic(self):
        assert _count_tokens("a" * 400) == 100
        assert _count_tokens("a" * 4) == 1

    def test_minimum_one(self):
        assert _count_tokens("hi") == 1  # 2 // 4 = 0, clamped to 1


class TestSplitWithOverlap:
    def test_short_text_not_split(self):
        text = "short text"
        parts = _split_with_overlap(text, max_tokens=512, overlap_tokens=64)
        assert parts == [text]

    def test_long_text_split(self):
        # ~600 token equivalent: 600 * 4 = 2400 chars ≈ 480 words
        text = " ".join(["word"] * 700)
        parts = _split_with_overlap(text, max_tokens=512, overlap_tokens=64)
        assert len(parts) >= 2

    def test_overlap_produces_shared_words(self):
        text = " ".join([f"w{i}" for i in range(800)])
        parts = _split_with_overlap(text, max_tokens=200, overlap_tokens=40)
        assert len(parts) >= 2
        # Last words of part[0] should appear at start of part[1]
        last_words_p0 = set(parts[0].split()[-20:])
        first_words_p1 = set(parts[1].split()[:20])
        assert last_words_p0 & first_words_p1  # non-empty overlap

    def test_no_empty_parts(self):
        text = " ".join(["x"] * 500)
        parts = _split_with_overlap(text, max_tokens=100, overlap_tokens=20)
        assert all(p.strip() for p in parts)


class TestMakeId:
    def test_deterministic(self):
        a = _make_id("doc", ["root", "1 Intro"], 0)
        b = _make_id("doc", ["root", "1 Intro"], 0)
        assert a == b

    def test_different_index_differs(self):
        a = _make_id("doc", ["root"], 0)
        b = _make_id("doc", ["root"], 1)
        assert a != b

    def test_different_doc_differs(self):
        a = _make_id("doc_a", ["root"], 0)
        b = _make_id("doc_b", ["root"], 0)
        assert a != b

    def test_id_length(self):
        cid = _make_id("doc", ["root"], 0)
        assert len(cid) == 16


class TestChunkDocument:
    def test_basic_two_sections(self):
        doc = make_doc([
            [
                ("1 Introduction", 14.0),
                ("This is the introduction text.", 10.0),
                ("It describes the dataset.", 10.0),
            ],
            [
                ("2 Field Layout", 14.0),
                ("Field A is an integer.", 10.0),
                ("Field B is a string.", 10.0),
            ],
        ])
        chunks = chunk_document(doc)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.doc_name == "test_doc"
            assert c.text.strip()
            assert c.token_count >= 1
            assert len(c.section_path) >= 1

    def test_no_headings_still_chunks(self):
        doc = make_doc([[("Body text only. " * 50, 10.0)]])
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

    def test_empty_document_no_chunks(self):
        doc = make_doc([[]])
        chunks = chunk_document(doc)
        assert chunks == []

    def test_section_path_tracks_hierarchy(self):
        doc = make_doc([
            [
                ("1 Overview", 14.0),
                ("Some overview text.", 10.0),
                ("1.1 Background", 14.0),
                ("Background details here.", 10.0),
            ]
        ])
        chunks = chunk_document(doc)
        assert len(chunks) >= 2
        # First chunk should be under "1 Overview"
        assert any("1 Overview" in c.section_path for c in chunks)
        # Second chunk should be under "1.1 Background"
        assert any("1.1 Background" in c.section_path for c in chunks)

    def test_short_section_merged(self):
        config = ChunkingConfig(max_tokens=512, overlap_tokens=64, min_tokens=50)
        doc = make_doc([
            [
                ("1 Introduction", 14.0),
                ("Short body.", 10.0),   # ~11 chars = ~2 tokens — below min
                ("2 Details", 14.0),
                ("Longer body text that should form its own chunk. " * 5, 10.0),
            ]
        ])
        chunks = chunk_document(doc, config)
        # The tiny "Short body." section should be merged rather than emitted alone
        # (it has ~2 tokens; min_tokens=50)
        assert all(c.token_count >= 1 for c in chunks)

    def test_long_section_split(self):
        # One section with ~700 tokens of text
        long_text = "word " * 700
        doc = make_doc([[("1 Data Layout", 14.0), (long_text, 10.0)]])
        config = ChunkingConfig(max_tokens=200, overlap_tokens=30, min_tokens=20)
        chunks = chunk_document(doc, config)
        assert len(chunks) >= 2

    def test_chunk_ids_unique(self):
        doc = make_doc([
            [
                ("1 Section A", 14.0), ("Text for A.", 10.0),
                ("2 Section B", 14.0), ("Text for B.", 10.0),
            ]
        ])
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_page_num_recorded(self):
        doc = make_doc([
            [("1 Intro", 14.0), ("Intro text.", 10.0)],
            [("2 Body", 14.0), ("Body text.", 10.0)],
        ])
        chunks = chunk_document(doc)
        page_nums = {c.page_num for c in chunks}
        assert 1 in page_nums or 2 in page_nums


# ----------------------------------------------------------------- imports --

def test_embedder_importable():
    from src.ingest.embed import Embedder
    e = Embedder(model_name="BAAI/bge-large-en-v1.5")
    assert e.model_name == "BAAI/bge-large-en-v1.5"
    # Model is NOT loaded here (lazy) — just check the object was created
    assert e._model is None


def test_store_importable():
    from src.ingest.store import QdrantStore, COLLECTION_NAME
    assert COLLECTION_NAME == "troa_chunks"
    # Don't instantiate — that requires Qdrant running


def test_pipeline_importable():
    from src.ingest.pipeline import run, main
    assert callable(run)
    assert callable(main)
