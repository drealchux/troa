"""
Semantic chunker for the TROA RAG corpus.

Splits ParsedDocuments into Chunks that respect section boundaries. Strategy:
  - Walk blocks sequentially; a heading triggers a section boundary.
  - Accumulate body text within a section; split at max_tokens with overlap.
  - Merge trailing sub-threshold chunks into the preceding chunk.

Token counting uses a character approximation (4 chars ≈ 1 token), which is
fast and accurate enough for chunking. For exact token counts we'd need to load
the BGE tokenizer here, which adds latency we don't need at ingest time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from statistics import median
from typing import Optional

from .parse import ParsedDocument, _is_likely_heading


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    section_path: list[str]   # e.g. ["root", "1 Overview", "1.2 Field Layout"]
    page_num: int
    text: str
    token_count: int


@dataclass
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 100


def _count_tokens(text: str) -> int:
    """Character-based token approximation: 4 chars ≈ 1 token."""
    return max(1, len(text) // 4)


def _make_id(doc_name: str, section_path: list[str], index: int) -> str:
    key = f"{doc_name}|{'|'.join(section_path)}|{index}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _merge_small_chunks(
    chunks: list[Chunk], max_tokens: int, min_tokens: int
) -> list[Chunk]:
    """Greedily merge consecutive under-threshold chunks until min_tokens or max_tokens is hit.

    Handles dense/tabular PDFs where many sections produce tiny body fragments.
    Preserves the chunk_id and section_path of the first chunk in each merged group.
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    pending = chunks[0]

    for chunk in chunks[1:]:
        combined_tc = _count_tokens(pending.text + " " + chunk.text)
        if pending.token_count < min_tokens and combined_tc <= max_tokens:
            pending = Chunk(
                chunk_id=pending.chunk_id,
                doc_name=pending.doc_name,
                section_path=pending.section_path,
                page_num=pending.page_num,
                text=pending.text + " " + chunk.text,
                token_count=combined_tc,
            )
        else:
            merged.append(pending)
            pending = chunk

    merged.append(pending)
    return merged


def _split_with_overlap(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into token-bounded slices with overlap between adjacent slices."""
    if _count_tokens(text) <= max_tokens:
        return [text]

    words = text.split()
    # 1 word ≈ 5 chars ≈ 1.25 tokens → words_per_chunk ≈ max_tokens * 0.8
    words_per_chunk = max(1, int(max_tokens * 0.8))
    overlap_words = max(0, int(overlap_tokens * 0.8))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words

    return chunks if chunks else [text]


def chunk_document(
    doc: ParsedDocument, config: Optional[ChunkingConfig] = None
) -> list[Chunk]:
    """Split a ParsedDocument into Chunks respecting section boundaries."""
    if config is None:
        config = ChunkingConfig()

    all_blocks = doc.all_blocks
    sizes = [b.font_size for b in all_blocks if b.font_size > 0]
    median_size = median(sizes) if sizes else 0.0

    section_path: list[str] = ["root"]
    section_buffer: list[str] = []
    section_page: int = 1
    chunks: list[Chunk] = []
    chunk_index = 0

    def flush_buffer() -> None:
        nonlocal chunk_index
        text = " ".join(section_buffer).strip()
        if not text:
            return
        parts = _split_with_overlap(text, config.max_tokens, config.overlap_tokens)
        for part in parts:
            tc = _count_tokens(part)
            if tc < config.min_tokens and chunks and chunks[-1].section_path == section_path:
                # Merge short trailing piece into the preceding chunk of the same section
                prev = chunks[-1]
                merged = prev.text + " " + part
                chunks[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    doc_name=prev.doc_name,
                    section_path=prev.section_path,
                    page_num=prev.page_num,
                    text=merged,
                    token_count=_count_tokens(merged),
                )
            else:
                chunks.append(Chunk(
                    chunk_id=_make_id(doc.doc_name, section_path, chunk_index),
                    doc_name=doc.doc_name,
                    section_path=list(section_path),
                    page_num=section_page,
                    text=part,
                    token_count=tc,
                ))
                chunk_index += 1

    for block in all_blocks:
        if _is_likely_heading(block.text, block.font_size, median_size):
            flush_buffer()
            section_buffer = []
            # Track hierarchical depth from numbering (e.g. "1.2" → depth 2)
            heading_text = block.text.strip()
            m = re.match(r'^(\d+(?:\.\d+)*)\s', heading_text)
            if m:
                depth = m.group(1).count('.') + 1
                section_path = section_path[:1] + section_path[1:depth] + [heading_text]
            else:
                section_path = [section_path[0], heading_text]
            section_page = block.page_num
        else:
            section_buffer.append(block.text)

    flush_buffer()
    return _merge_small_chunks(chunks, config.max_tokens, config.min_tokens)
