"""
TROA serving pipeline: router → embed → retrieve → rerank → generate → guardrail.

Usage:
    from src.serve.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run("What is the filing deadline for Form W-10?")
    print(result.answer)

Calibration is optional. Without it, the raw generator confidence is used.
When a CalibrationModel is supplied, the pipeline applies Platt scaling to
convert the raw 0–100 integer into a calibrated probability before thresholding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .generate import Generator, GeneratorResult
from .rerank import RankedChunk, Reranker
from .retrieve import RetrievedChunk, Retriever
from .router import Router, RouterResult

log = logging.getLogger(__name__)

_CAVEATS_PATH = Path(__file__).parent / "prompts" / "caveats.yaml"

# Guardrail thresholds on calibrated [0, 1] confidence
THRESHOLD_AUTONOMOUS = 0.85
THRESHOLD_CAVEAT = 0.70


def _load_caveats() -> dict:
    with open(_CAVEATS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class PipelineResponse:
    question: str
    answer: str             # final answer shown to user (may include caveat banner)
    is_ood: bool
    refused: bool           # True when confidence < THRESHOLD_CAVEAT or OOD
    raw_confidence: int     # 0–100 from generator
    calibrated_confidence: float    # after Platt scaling (or raw/100 if no calibrator)
    intent: str
    doc_scope: Optional[list[str]]
    retrieved_chunks: list[RetrievedChunk]      # top-20 from Qdrant
    ranked_chunks: list[RankedChunk]            # top-5 after reranking
    usage: dict             # token counts from the generator call


class Pipeline:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_path: Optional[str] = None,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        rerank_model: str = "BAAI/bge-reranker-large",
        retrieve_top_k: int = 20,
        rerank_top_k: int = 5,
        generate_prompt: str = "generate_v1.yaml",
        router_prompt: str = "router_v1.yaml",
        calibrator=None,    # Optional[CalibrationModel] — avoids circular import
        api_key: Optional[str] = None,
    ):
        self._retriever = Retriever(url=qdrant_url, path=qdrant_path)
        self._reranker = Reranker(model_name=rerank_model)
        self._generator = Generator(prompt_file=generate_prompt, api_key=api_key)
        self._router = Router(prompt_file=router_prompt, api_key=api_key)
        self._calibrator = calibrator
        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_k = rerank_top_k
        self._caveats = _load_caveats()

        # Embedder is lazy-loaded inside retrieve; import here for query embedding
        from src.ingest.embed import Embedder
        self._embedder = Embedder(model_name=embed_model)

    def run(self, question: str) -> PipelineResponse:
        # 1. Route
        route: RouterResult = self._router.route(question)
        log.debug("Router: intent=%s is_ood=%s doc_scope=%s",
                  route.intent, route.is_ood, route.doc_scope)

        if route.is_ood and route.ood_confidence >= 0.85:
            return PipelineResponse(
                question=question,
                answer=self._caveats["ood"],
                is_ood=True,
                refused=True,
                raw_confidence=0,
                calibrated_confidence=0.0,
                intent=route.intent,
                doc_scope=route.doc_scope,
                retrieved_chunks=[],
                ranked_chunks=[],
                usage={},
            )

        # 2. Embed query
        query_vec = self._embedder.embed_query(question)

        # 3. Retrieve top-20
        candidates = self._retriever.retrieve(
            query_vector=query_vec,
            top_k=self._retrieve_top_k,
            doc_names=route.doc_scope,
        )
        log.debug("Retrieved %d candidates", len(candidates))

        # 4. Rerank to top-5
        ranked = self._reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=self._rerank_top_k,
        )
        log.debug("Reranked to %d chunks", len(ranked))

        if not ranked:
            return PipelineResponse(
                question=question,
                answer=self._caveats["escalate"],
                is_ood=False,
                refused=True,
                raw_confidence=0,
                calibrated_confidence=0.0,
                intent=route.intent,
                doc_scope=route.doc_scope,
                retrieved_chunks=candidates,
                ranked_chunks=[],
                usage={},
            )

        # 5. Generate
        gen: GeneratorResult = self._generator.generate(question, ranked)

        # 6. Calibrate confidence
        raw_prob = gen.raw_confidence / 100.0
        if self._calibrator is not None:
            cal_conf = float(self._calibrator.predict([raw_prob])[0])
        else:
            cal_conf = raw_prob

        # 7. Guardrail
        answer = gen.answer
        refused = False

        if cal_conf < THRESHOLD_CAVEAT:
            answer = self._caveats["escalate"]
            refused = True
        elif cal_conf < THRESHOLD_AUTONOMOUS:
            answer = gen.answer.rstrip() + "\n\n" + self._caveats["low_confidence"]

        return PipelineResponse(
            question=question,
            answer=answer,
            is_ood=False,
            refused=refused,
            raw_confidence=gen.raw_confidence,
            calibrated_confidence=cal_conf,
            intent=route.intent,
            doc_scope=route.doc_scope,
            retrieved_chunks=candidates,
            ranked_chunks=ranked,
            usage=gen.usage,
        )
