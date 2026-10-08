"""
Retriever for TROA: dense search in Qdrant, optionally fused with BM25.

Returns up to `top_k` candidates with their payload. Supports an optional
`doc_names` filter (MatchAny on the doc_name payload field) for doc-scoped
queries identified by the router.

Modes:
    vector  dense cosine search only (the original baseline)
    hybrid  dense top-N and BM25 top-N fused with reciprocal rank fusion
            (src/serve/hybrid.py). The BM25 index is built in memory from the
            chunk payloads already in Qdrant, so no re-ingestion is needed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

from .hybrid import BM25, rrf_fuse

COLLECTION_NAME = "troa_chunks"
SEARCH_MODES = ("vector", "hybrid")
CANDIDATE_POOL = 50     # per-retriever candidates fed into fusion


def _make_client(url: str = "http://localhost:6333", path: Optional[str] = None) -> "QdrantClient":
    if path:
        return QdrantClient(path=path)
    return QdrantClient(url=url, timeout=30)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    section_path: list[str]
    page_num: int
    text: str
    token_count: int
    score: float    # cosine similarity (vector mode) or RRF score (hybrid mode)
    vector_rank: Optional[int] = None   # 1-based rank in the dense list, if present
    bm25_rank: Optional[int] = None     # 1-based rank in the BM25 list (hybrid only)


def _from_payload(payload: dict, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        doc_name=payload["doc_name"],
        section_path=payload["section_path"],
        page_num=payload["page_num"],
        text=payload["text"],
        token_count=payload["token_count"],
        score=score,
    )


class Retriever:
    def __init__(self, url: str = "http://localhost:6333", path: Optional[str] = None):
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is not installed.")
        self.client = _make_client(url=url, path=path)
        self._doc_names: Optional[list[str]] = None
        self._kw_chunks: Optional[list[RetrievedChunk]] = None
        self._kw_docs: Optional[np.ndarray] = None
        self._bm25: Optional[BM25] = None

    def doc_names(self) -> list[str]:
        """Distinct doc_name values in the collection (scanned once, then cached)."""
        if self._doc_names is None:
            names: set[str] = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    with_payload=["doc_name"],
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                names.update(p.payload["doc_name"] for p in points)
                if offset is None:
                    break
            self._doc_names = sorted(names)
        return self._doc_names

    def doc_stats(self) -> dict[str, dict]:
        """Per document: chunk count, highest page number cited, and total tokens."""
        stats: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=["doc_name", "page_num", "token_count"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for p in points:
                s = stats.setdefault(p.payload["doc_name"], {"chunks": 0, "pages": 0, "tokens": 0})
                s["chunks"] += 1
                s["pages"] = max(s["pages"], int(p.payload.get("page_num") or 0))
                s["tokens"] += int(p.payload.get("token_count") or 0)
            if offset is None:
                break
        return stats

    def fingerprint(self) -> str:
        """Short hash of the collection's size and documents, for cache keys."""
        count = self.client.count(collection_name=COLLECTION_NAME).count
        raw = f"{count}|{'|'.join(self.doc_names())}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _keyword_index(self) -> BM25:
        """Build the BM25 index over every chunk payload (once, then cached)."""
        if self._bm25 is None:
            chunks: list[RetrievedChunk] = []
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    with_payload=True,
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                chunks.extend(_from_payload(p.payload) for p in points)
                if offset is None:
                    break
            self._kw_chunks = chunks
            self._kw_docs = np.array([c.doc_name for c in chunks])
            self._bm25 = BM25([c.text for c in chunks])
        return self._bm25

    def _dense(self, query_vector: np.ndarray, limit: int,
               doc_names: Optional[list[str]]) -> list[RetrievedChunk]:
        query_filter: Optional[Filter] = None
        if doc_names:
            query_filter = Filter(
                must=[FieldCondition(key="doc_name", match=MatchAny(any=doc_names))]
            )
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        out = [_from_payload(r.payload, r.score) for r in results]
        for rank, c in enumerate(out, start=1):
            c.vector_rank = rank
        return out

    def _keyword(self, query_text: str, limit: int,
                 doc_names: Optional[list[str]]) -> list[RetrievedChunk]:
        bm25 = self._keyword_index()
        mask = np.isin(self._kw_docs, doc_names) if doc_names else None
        out = []
        for rank, i in enumerate(bm25.top(query_text, limit, mask), start=1):
            c = self._kw_chunks[i]
            out.append(RetrievedChunk(**{**c.__dict__, "bm25_rank": rank}))
        return out

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        doc_names: Optional[list[str]] = None,
        query_text: Optional[str] = None,
        mode: str = "vector",
    ) -> list[RetrievedChunk]:
        """Return the top_k candidates for the query.

        doc_names: if provided, restricts results to chunks whose doc_name
        matches any entry in the list (case-sensitive, exact match). Use
        src.serve.scope.resolve_scope to map router names onto doc_names().
        mode: "vector", or "hybrid" (needs query_text for the BM25 side).
        """
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
        if mode == "vector" or not query_text:
            return self._dense(query_vector, top_k, doc_names)

        pool = max(top_k, CANDIDATE_POOL)
        dense = self._dense(query_vector, pool, doc_names)
        keyword = self._keyword(query_text, pool, doc_names)

        by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in keyword}
        for c in dense:   # dense entries carry the cosine score; keep the BM25 rank too
            if c.chunk_id in by_id:
                c.bm25_rank = by_id[c.chunk_id].bm25_rank
            by_id[c.chunk_id] = c
        fused = rrf_fuse([[c.chunk_id for c in dense], [c.chunk_id for c in keyword]])
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [RetrievedChunk(**{**by_id[cid].__dict__, "score": fused[cid]}) for cid in top]
