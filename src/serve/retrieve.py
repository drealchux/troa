"""
Dense retriever for TROA: queries Qdrant with an embedded query vector.

Returns up to `top_k` candidates with their payload. Supports an optional
`doc_names` filter (MatchAny on the doc_name payload field) for doc-scoped
queries identified by the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

COLLECTION_NAME = "troa_chunks"


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
    score: float    # cosine similarity from Qdrant


class Retriever:
    def __init__(self, url: str = "http://localhost:6333", path: Optional[str] = None):
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is not installed.")
        self.client = _make_client(url=url, path=path)

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        doc_names: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Return top_k chunks nearest to query_vector.

        doc_names: if provided, restricts results to chunks whose doc_name
        matches any entry in the list (case-sensitive, exact match).
        """
        query_filter: Optional[Filter] = None
        if doc_names:
            query_filter = Filter(
                must=[FieldCondition(key="doc_name", match=MatchAny(any=doc_names))]
            )

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                chunk_id=r.payload["chunk_id"],
                doc_name=r.payload["doc_name"],
                section_path=r.payload["section_path"],
                page_num=r.payload["page_num"],
                text=r.payload["text"],
                token_count=r.payload["token_count"],
                score=r.score,
            )
            for r in results
        ]
