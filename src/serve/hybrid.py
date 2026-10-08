"""
Keyword search and rank fusion for TROA's hybrid retrieval.

RRC manuals are full of exact tokens that dense embeddings handle poorly:
form numbers (W-10, P-5), field names (WELL-NO), record type codes. BM25 over
the same chunks catches those, and reciprocal rank fusion (RRF) merges the
keyword and vector rankings without having to put their scores on one scale.

Used by src/serve/retrieve.py over the chunk payloads in Qdrant. Pattern adapted from the hybrid-search week of
jamwithai/production-agentic-rag-course; there it runs inside OpenSearch, here
the corpus (a few thousand chunks) is small enough to score in-process.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np

RRF_K = 60  # standard RRF constant: 1 / (k + rank)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    # Also index the parts of hyphenated tokens so "w-10" matches "W 10" and vice versa.
    return tokens + [p for t in tokens if "-" in t for p in t.split("-")]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.lengths = np.array(lengths, dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            idx = np.fromiter((p[0] for p in plist), dtype=np.int64)
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32)
            norm = self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avg_len)
            out[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top(self, query: str, k: int, mask: Optional[np.ndarray] = None) -> list[int]:
        """Indices of the k best-scoring texts with a positive score."""
        scores = self.scores(query)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores)[:k]
        return [int(i) for i in order if scores[i] > 0]


def rrf_fuse(rankings: Iterable[list], k: int = RRF_K) -> dict:
    """Reciprocal rank fusion: {item: sum over rankings of 1 / (k + rank)}.

    Each ranking is a best-first list of hashable items (ids or indices).
    Returns scores for every item that appears in any ranking.
    """
    fused: dict = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return dict(fused)
