"""
Qdrant vector store for the TROA RAG corpus.

One collection ("troa_chunks") with:
  - 1024-dimensional cosine vectors
  - Payload indexes on doc_name and section_path for metadata-filtered retrieval
  - Deterministic point IDs derived from chunk_id (UUID5) so re-ingestion
    is idempotent — re-uploading the same chunk overwrites rather than duplicates.
"""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

from .chunk import Chunk

COLLECTION_NAME = "troa_chunks"
DEFAULT_VECTOR_SIZE = 1024


def _chunk_to_point_id(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id so upserts are idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _make_client(url: str = "http://localhost:6333", path: Optional[str] = None) -> "QdrantClient":
    """Return a QdrantClient in local-file mode (path) or HTTP mode (url)."""
    if path:
        return QdrantClient(path=path)
    return QdrantClient(url=url, timeout=30)


class QdrantStore:
    def __init__(self, url: str = "http://localhost:6333", path: Optional[str] = None):
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            )
        self.client = _make_client(url=url, path=path)

    def ensure_collection(self, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
        """Create the collection and payload indexes if they don't exist yet."""
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            return
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        for field_name in ("doc_name", "section_path"):
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema="keyword",
            )

    def upsert(
        self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 100
    ) -> int:
        """Upsert chunks with their embeddings. Returns number of points written."""
        assert len(chunks) == len(embeddings), "chunks and embeddings length mismatch"

        points = [
            PointStruct(
                id=_chunk_to_point_id(chunk.chunk_id),
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "section_path": chunk.section_path,
                    "page_num": chunk.page_num,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + batch_size],
            )

        return len(points)

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count

    def collection_info(self) -> dict:
        info = self.client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }
