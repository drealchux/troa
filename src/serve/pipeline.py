"""
TROA serving pipeline:
    router → embed → retrieve (vector or hybrid) → rerank
           → [grade → rewrite → retrieve → rerank → grade]   (optional, once)
           → generate → calibrate → guardrail

Usage:
    from src.serve.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run("What is the filing deadline for Form W-10?")
    print(result.decision, result.answer)

    for event in pipe.stream("..."):       # {"event": "meta" | "token" | "final", ...}
        ...

Calibration is optional. Without it, the raw generator confidence / 100 is used.
When a CalibrationModel (src/eval/calibration.py) is supplied, the pipeline
calls its predict_proba() to map the raw 0–100 integer to a calibrated
probability before thresholding. Thresholds live in src/serve/guardrail.py.

Search mode, the agent loop, the answer cache, and the query log are all off
by default so the baseline stays comparable; turn each on and measure it with
the eval harness (python -m src.eval.harness --help).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import anthropic
import yaml

from .cache import CACHEABLE_DECISIONS, AnswerCache, cache_key, prompts_fingerprint
from .generate import Generator, GeneratorResult, visible_text
from .guardrail import OOD_CUTOFF, THRESHOLD_AUTONOMOUS, THRESHOLD_CAVEAT, decide, is_ood_refusal
from .rerank import PassthroughReranker, RankedChunk, Reranker
from .retrieve import SEARCH_MODES, RetrievedChunk, Retriever
from .router import Router, RouterResult
from .scope import resolve_scope
from .telemetry import Trace, open_query_log

log = logging.getLogger(__name__)

_CAVEATS_PATH = Path(__file__).parent / "prompts" / "caveats.yaml"


def _load_caveats() -> dict:
    with open(_CAVEATS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class PipelineResponse:
    question: str
    answer: str             # final answer shown to user (may include caveat banner)
    is_ood: bool
    refused: bool           # True when escalated (confidence < THRESHOLD_CAVEAT) or OOD
    raw_confidence: int     # 0–100 from generator
    calibrated_confidence: float    # after Platt scaling (or raw/100 if no calibrator)
    intent: str
    doc_scope: Optional[list[str]]      # resolved doc_names searched; None = whole corpus
    retrieved_chunks: list[RetrievedChunk]      # top-20 candidates (pooled across a retry)
    ranked_chunks: list[RankedChunk]            # top-5 after reranking
    usage: dict             # token counts from the generator call
    decision: str = ""      # autonomous | caveat | escalate | refuse_ood
    search_query: str = ""  # query behind the final retrieval (the rewrite, after a retry)
    retrieval_sufficient: Optional[bool] = None     # grader's final verdict; None if not graded
    agent_steps: list[dict] = field(default_factory=list)  # grade / rewrite records
    trace: list[dict] = field(default_factory=list)        # per-stage timings
    latency_s: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineResponse":
        data = dict(data)
        data["retrieved_chunks"] = [RetrievedChunk(**c) for c in data["retrieved_chunks"]]
        data["ranked_chunks"] = [
            RankedChunk(chunk=RetrievedChunk(**rc["chunk"]), rerank_score=rc["rerank_score"])
            for rc in data["ranked_chunks"]
        ]
        return cls(**data)


@dataclass
class _Prepared:
    """Everything before generation. `response` is set when no generation is needed."""
    question: str
    route: RouterResult
    trace: Trace
    scope: Optional[list[str]] = None
    candidates: list[RetrievedChunk] = field(default_factory=list)
    ranked: list[RankedChunk] = field(default_factory=list)
    search_query: str = ""
    retrieval_sufficient: Optional[bool] = None
    agent_steps: list[dict] = field(default_factory=list)
    response: Optional[PipelineResponse] = None


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
        grader_prompt: str = "grader_v1.yaml",
        rewrite_prompt: str = "rewrite_v1.yaml",
        calibrator=None,    # Optional[CalibrationModel]: anything with predict_proba(raw_0_100)
        api_key: Optional[str] = None,
        search_mode: str = "vector",    # "vector" | "hybrid" (BM25 + dense, RRF)
        agentic: bool = False,          # grade → rewrite → retry once
        cache: Optional[AnswerCache] = None,
        query_log: Optional[str | Path] = None,     # JSONL path, one line per question
        rerank: bool = True,            # False: skip the cross-encoder, keep top-k by retrieval score
        *,
        # Component overrides (tests, alternative backends). Built from the
        # arguments above when not given.
        retriever=None,
        reranker=None,
        generator=None,
        router=None,
        embedder=None,
        grader=None,
        rewriter=None,
    ):
        if search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {SEARCH_MODES}, got {search_mode!r}")
        self._retriever = retriever or Retriever(url=qdrant_url, path=qdrant_path)
        if not rerank:
            reranker, rerank_model = PassthroughReranker(), PassthroughReranker.model_name
        self._reranker = reranker or Reranker(model_name=rerank_model)
        self._rerank = rerank
        self._generator = generator or Generator(prompt_file=generate_prompt, api_key=api_key)
        self._router = router or Router(prompt_file=router_prompt, api_key=api_key)
        self._calibrator = calibrator
        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_k = rerank_top_k
        self._search_mode = search_mode
        self._agentic = agentic
        self._cache = cache
        self._query_log = open_query_log(query_log)
        self._caveats = _load_caveats()
        self._prompt_files = [generate_prompt, router_prompt, "caveats.yaml"]
        self._model_names = [embed_model, rerank_model]
        self._versions: Optional[dict] = None

        if agentic:
            from .agent import Grader, Rewriter
            self._grader = grader or Grader(prompt_file=grader_prompt, api_key=api_key)
            self._rewriter = rewriter or Rewriter(prompt_file=rewrite_prompt, api_key=api_key)
            self._prompt_files += [grader_prompt, rewrite_prompt]

        if embedder is None:
            # Imported here so the heavy model stack loads only when needed
            from src.ingest.embed import Embedder
            embedder = Embedder(model_name=embed_model)
        self._embedder = embedder

    # ---------- public API ----------

    @property
    def settings(self) -> dict:
        return {
            "search_mode": self._search_mode,
            "agentic": self._agentic,
            "rerank": self._rerank,
            "calibrated": self._calibrator is not None,
            "cache": self._cache.backend if self._cache else None,
            "query_log": str(self._query_log.path) if self._query_log else None,
        }

    def versions(self) -> dict:
        """Everything that changes an answer. Used for cache keys and query logs.

        Computed once; restart the process after re-ingesting or editing prompts.
        """
        if self._versions is None:
            to_json = getattr(self._calibrator, "to_json", None)
            self._versions = {
                "prompts": prompts_fingerprint(self._prompt_files),
                "thresholds": [THRESHOLD_AUTONOMOUS, THRESHOLD_CAVEAT, OOD_CUTOFF],
                "calibrator": to_json() if to_json else (
                    None if self._calibrator is None else repr(self._calibrator)),
                "models": self._model_names + [getattr(self._generator, "model", "")],
                "search": [self._search_mode, self._retrieve_top_k, self._rerank_top_k,
                           self._agentic],
                "corpus": self._retriever.fingerprint(),
            }
        return self._versions

    def run(self, question: str) -> PipelineResponse:
        t0 = time.perf_counter()
        key, hit = self._cache_lookup(question)
        if hit is not None:
            return self._complete(hit, t0, key=None)

        prep = self.prepare(question)
        if prep.response is not None:
            return self._complete(prep.response, t0, key)

        with prep.trace.stage("generate") as s:
            gen = self._generator.generate(question, prep.ranked)
            s["detail"] = getattr(self._generator, "model", "")
        return self._complete(self._finish(prep, gen), t0, key)

    def stream(self, question: str) -> Iterator[dict]:
        """Yield events while answering.

        {"event": "meta",  ...}    routing, scope, sources (before generation)
        {"event": "token", "text"}  answer text as it is generated
        {"event": "final", "response": {...}}   PipelineResponse.to_dict()

        The tokens are the generator's *draft*. The guardrail runs after
        generation, so `final.response.answer` is authoritative: on escalation
        it replaces the draft, and on a caveat it appends the banner.
        """
        t0 = time.perf_counter()
        key, hit = self._cache_lookup(question)
        if hit is not None:
            yield {"event": "final", "response": self._complete(hit, t0, key=None).to_dict()}
            return

        prep = self.prepare(question)
        if prep.response is not None:
            yield {"event": "final", "response": self._complete(prep.response, t0, key).to_dict()}
            return

        yield {
            "event": "meta",
            "intent": prep.route.intent,
            "doc_scope": prep.scope,
            "search_query": prep.search_query,
            "retrieval_sufficient": prep.retrieval_sufficient,
            "sources": [_source(i, rc) for i, rc in enumerate(prep.ranked, start=1)],
        }
        gen_stream = self._generator.stream(question, prep.ranked)
        sent = 0
        text = ""
        with prep.trace.stage("generate") as s:
            s["detail"] = getattr(self._generator, "model", "") + " (streamed)"
            for delta in gen_stream:
                text += delta
                shown = visible_text(text)
                if len(shown) > sent:
                    yield {"event": "token", "text": shown[sent:]}
                    sent = len(shown)
        resp = self._complete(self._finish(prep, gen_stream.result), t0, key)
        yield {"event": "final", "response": resp.to_dict()}

    # ---------- stages ----------

    def prepare(self, question: str) -> _Prepared:
        """Route, retrieve, rerank, and (optionally) grade and retry."""
        trace = Trace()

        # 1. Route
        with trace.stage("route") as s:
            route: RouterResult = self._router.route(question)
            s["detail"] = (f"intent={route.intent}, ood={route.is_ood} "
                           f"({route.ood_confidence:.2f}), scope={route.doc_scope}")
        prep = _Prepared(question=question, route=route, trace=trace, search_query=question)

        if is_ood_refusal(route.is_ood, route.ood_confidence):
            prep.response = self._early_response(prep, "refuse_ood", self._caveats["ood"],
                                                 is_ood=True, doc_scope=route.doc_scope)
            return prep

        # 2–3. Embed and retrieve, scoped to the documents the router named (if
        # any resolve). An empty scoped search falls back to the whole corpus.
        requested_scope = resolve_scope(route.doc_scope, self._retriever.doc_names()) or None
        with trace.stage("retrieve") as s:
            prep.candidates, prep.scope = self._search(question, requested_scope)
            s["detail"] = f"{self._search_mode}, {len(prep.candidates)} candidates, scope={prep.scope}"

        # 4. Rerank to top-5
        with trace.stage("rerank") as s:
            prep.ranked = self._reranker.rerank(
                query=question, candidates=prep.candidates, top_k=self._rerank_top_k)
            s["detail"] = f"{len(prep.ranked)} passages" + ("" if self._rerank else " (off: retrieval order)")

        # 5. Corrective retrieval (optional)
        if self._agentic and prep.ranked:
            self._grade_and_retry(prep, requested_scope)

        if not prep.ranked:
            prep.response = self._early_response(prep, "escalate", self._caveats["escalate"],
                                                 doc_scope=prep.scope)
        return prep

    def _search(self, query: str, scope: Optional[list[str]]
                ) -> tuple[list[RetrievedChunk], Optional[list[str]]]:
        query_vec = self._embedder.embed_query(query)
        candidates = self._retriever.retrieve(
            query_vector=query_vec, top_k=self._retrieve_top_k, doc_names=scope,
            query_text=query, mode=self._search_mode)
        if scope and not candidates:
            scope = None
            candidates = self._retriever.retrieve(
                query_vector=query_vec, top_k=self._retrieve_top_k,
                query_text=query, mode=self._search_mode)
        return candidates, scope

    def _grade_and_retry(self, prep: _Prepared, requested_scope: Optional[list[str]]) -> None:
        """Grade the passages; if insufficient, rewrite the query and retry once.

        Retry candidates are pooled with the first search and reranked against
        the original question, so a rewrite can add evidence but never steer
        the answer away from what was asked.
        """
        question, trace = prep.question, prep.trace
        try:
            with trace.stage("grade") as s:
                verdict = self._grader.grade(question, prep.ranked)
                s["detail"] = f"sufficient={verdict.sufficient}: {verdict.reason}"
            prep.agent_steps.append({"step": "grade", "query": question, **asdict(verdict)})
            prep.retrieval_sufficient = verdict.sufficient
            if verdict.sufficient:
                return

            with trace.stage("rewrite") as s:
                new_query = self._rewriter.rewrite(question, verdict.reason)
                s["detail"] = new_query
            prep.agent_steps.append({"step": "rewrite", "query": new_query})

            with trace.stage("retrieve (retry)") as s:
                extra, _ = self._search(new_query, requested_scope)
                seen = {c.chunk_id for c in prep.candidates}
                added = [c for c in extra if c.chunk_id not in seen]
                prep.candidates = prep.candidates + added
                s["detail"] = f"{len(added)} new candidates"
            with trace.stage("rerank (retry)") as s:
                prep.ranked = self._reranker.rerank(
                    query=question, candidates=prep.candidates, top_k=self._rerank_top_k)
                s["detail"] = f"{len(prep.ranked)} passages from {len(prep.candidates)} pooled"
            prep.search_query = new_query

            with trace.stage("grade (retry)") as s:
                verdict = self._grader.grade(question, prep.ranked)
                s["detail"] = f"sufficient={verdict.sufficient}: {verdict.reason}"
            prep.agent_steps.append({"step": "grade", "query": new_query, **asdict(verdict)})
            prep.retrieval_sufficient = verdict.sufficient
        except anthropic.APIError as exc:
            # The loop is an optimisation; fall through to generation without it.
            log.warning("Agent loop failed, continuing without it: %s", exc)
            prep.agent_steps.append({"step": "error", "detail": str(exc)})

    def _finish(self, prep: _Prepared, gen: GeneratorResult) -> PipelineResponse:
        # Calibrate confidence
        if self._calibrator is not None:
            cal_conf = float(self._calibrator.predict_proba(gen.raw_confidence)[0])
        else:
            cal_conf = gen.raw_confidence / 100.0

        # Guardrail
        answer = gen.answer
        decision = decide(cal_conf)
        if decision == "escalate":
            answer = self._caveats["escalate"]
        elif decision == "caveat":
            answer = gen.answer.rstrip() + "\n\n" + self._caveats["low_confidence"]

        return PipelineResponse(
            question=prep.question,
            answer=answer,
            is_ood=False,
            refused=decision == "escalate",
            raw_confidence=gen.raw_confidence,
            calibrated_confidence=cal_conf,
            intent=prep.route.intent,
            doc_scope=prep.scope,
            retrieved_chunks=prep.candidates,
            ranked_chunks=prep.ranked,
            usage=gen.usage,
            decision=decision,
            search_query=prep.search_query,
            retrieval_sufficient=prep.retrieval_sufficient,
            agent_steps=prep.agent_steps,
            trace=prep.trace.stages,
        )

    def _early_response(self, prep: _Prepared, decision: str, answer: str, *,
                        is_ood: bool = False, doc_scope=None) -> PipelineResponse:
        return PipelineResponse(
            question=prep.question,
            answer=answer,
            is_ood=is_ood,
            refused=True,
            raw_confidence=0,
            calibrated_confidence=0.0,
            intent=prep.route.intent,
            doc_scope=doc_scope,
            retrieved_chunks=prep.candidates,
            ranked_chunks=[],
            usage={},
            decision=decision,
            search_query=prep.search_query,
            retrieval_sufficient=prep.retrieval_sufficient,
            agent_steps=prep.agent_steps,
            trace=prep.trace.stages,
        )

    # ---------- cache and logging ----------

    def _cache_lookup(self, question: str) -> tuple[Optional[str], Optional[PipelineResponse]]:
        if self._cache is None:
            return None, None
        key = cache_key(question, self.versions())
        hit = self._cache.get(key)
        if hit is None:
            return key, None
        resp = PipelineResponse.from_dict(hit)
        resp.cached = True
        resp.trace = [{"stage": "cache", "seconds": 0.0, "detail": self._cache.backend}]
        return key, resp

    def _complete(self, resp: PipelineResponse, t0: float, key: Optional[str]) -> PipelineResponse:
        resp.latency_s = round(time.perf_counter() - t0, 3)
        if key and self._cache is not None and resp.decision in CACHEABLE_DECISIONS:
            self._cache.set(key, resp.to_dict())
        if self._query_log is not None:
            self._query_log.write(self._log_record(resp))
        return resp

    def _log_record(self, resp: PipelineResponse) -> dict:
        return {
            "question": resp.question,
            "decision": resp.decision,
            "answer": resp.answer,
            "intent": resp.intent,
            "is_ood": resp.is_ood,
            "doc_scope": resp.doc_scope,
            "search_query": resp.search_query,
            "retrieval_sufficient": resp.retrieval_sufficient,
            "raw_confidence": resp.raw_confidence,
            "calibrated_confidence": resp.calibrated_confidence,
            "ranked": [_source(i, rc) for i, rc in enumerate(resp.ranked_chunks, start=1)],
            "agent_steps": resp.agent_steps,
            "usage": resp.usage,
            "latency_s": resp.latency_s,
            "cached": resp.cached,
            "trace": resp.trace,
            "settings": {k: v for k, v in self.settings.items() if k != "query_log"},
            "versions": self.versions(),
        }


def _source(n: int, rc: RankedChunk) -> dict:
    c = rc.chunk
    return {
        "n": n,
        "chunk_id": c.chunk_id,
        "doc_name": c.doc_name,
        "section_path": c.section_path,
        "page_num": c.page_num,
        "rerank_score": round(rc.rerank_score, 4),
        "vector_rank": c.vector_rank,
        "bm25_rank": c.bm25_rank,
    }
