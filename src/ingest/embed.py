"""
Embedder wrapper for the TROA RAG corpus.

Uses BAAI/bge-large-en-v1.5 (1024-dimensional, MIT license). BGE uses a
query-prefix convention for retrieval tasks: passages are embedded as-is;
queries get a short instruction prefix so they align with the passage space.

Model is loaded lazily on first call to avoid paying the load cost at import.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# BGE retrieval query prefix (passages need no prefix)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed corpus passages. No prefix needed for BGE passage encoding."""
        return np.array(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with BGE's instruction prefix."""
        return np.array(
            self.model.encode(
                _QUERY_PREFIX + text,
                normalize_embeddings=True,
            )
        )
