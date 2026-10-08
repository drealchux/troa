"""
HTTP service for TROA.

    uvicorn src.api.app:app --port 8000

Endpoints:
    GET  /health        liveness, corpus size, active settings
    POST /ask           full answer with guardrail decision, confidence, citations, trace
    POST /ask/stream    Server-Sent Events: meta, token..., final

The guardrail decision is part of the contract: clients must show `answer`
(which is the escalation notice when decision == "escalate"), not the
generator's draft. On /ask/stream the token events are that draft, so a
client must replace the streamed text with `final.answer`.

Configured from the environment (src/config.py, .env.example). Endpoint layout
adapted from jamwithai/production-agentic-rag-course (weeks 1 and 5).
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class Citation(BaseModel):
    n: int
    chunk_id: str
    doc_name: str
    section_path: list[str]
    page_num: int
    rerank_score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class AskResponse(BaseModel):
    question: str
    decision: str = Field(..., description="autonomous | caveat | escalate | refuse_ood")
    answer: str
    raw_confidence: int
    calibrated_confidence: float
    intent: str
    doc_scope: Optional[list[str]]
    search_query: str
    retrieval_sufficient: Optional[bool]
    citations: list[Citation]
    agent_steps: list[dict]
    trace: list[dict]
    usage: dict
    latency_s: float
    cached: bool


def to_ask_response(resp: dict) -> AskResponse:
    citations = []
    for i, rc in enumerate(resp["ranked_chunks"], start=1):
        c = rc["chunk"]
        citations.append(Citation(
            n=i, chunk_id=c["chunk_id"], doc_name=c["doc_name"],
            section_path=c["section_path"], page_num=c["page_num"],
            rerank_score=rc["rerank_score"], vector_rank=c.get("vector_rank"),
            bm25_rank=c.get("bm25_rank"),
        ))
    return AskResponse(
        citations=citations,
        **{k: resp[k] for k in AskResponse.model_fields if k != "citations"},
    )


def create_app(pipeline=None) -> FastAPI:
    """Build the app. Pass a pipeline to skip construction from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            from src.config import build_pipeline
            app.state.pipeline = build_pipeline()
        else:
            app.state.pipeline = pipeline
        yield

    app = FastAPI(title="TROA", version="0.1.0",
                  description="Texas Regulatory Oil & gas Assistant", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        pipe = request.app.state.pipeline
        try:
            corpus = pipe.versions()["corpus"]
            status = "ok"
        except Exception as exc:   # Qdrant down or collection missing
            log.warning("Health check: corpus unavailable: %s", exc)
            corpus, status = None, "degraded"
        return {"status": status, "corpus": corpus, **pipe.settings}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        try:
            resp = request.app.state.pipeline.run(body.question)
        except Exception as exc:
            log.exception("Pipeline failed")
            raise HTTPException(status_code=502, detail=f"pipeline error: {exc}") from exc
        return to_ask_response(resp.to_dict())

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
        pipe = request.app.state.pipeline

        def events() -> Iterator[str]:
            try:
                for ev in pipe.stream(body.question):
                    name = ev.pop("event")
                    if name == "final":
                        ev = to_ask_response(ev["response"]).model_dump()
                    yield f"event: {name}\ndata: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:
                log.exception("Pipeline failed while streaming")
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
