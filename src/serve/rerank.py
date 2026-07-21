"""
Cross-encoder reranker for TROA using BAAI/bge-reranker-large.

Scores each (query, passage) pair and returns the top_k by score.
Model is loaded lazily on first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .retrieve import RetrievedChunk


@dataclass
class RankedChunk:
    chunk: RetrievedChunk
    rerank_score: float


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Score candidates against query and return top_k by rerank score."""
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            RankedChunk(chunk=chunk, rerank_score=float(score))
            for score, chunk in ranked[:top_k]
        ]
