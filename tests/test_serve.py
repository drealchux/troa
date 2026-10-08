"""
Tests for the serving layer: hybrid search, answer cache, query log, the
grade/rewrite loop, streaming, and the HTTP API. All model and API calls are
replaced by fakes, so these run offline without Qdrant or an API key.

Run with: pytest tests/test_serve.py -v
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.serve.agent import GradeResult
from src.serve.cache import AnswerCache, cache_key, normalize_question
from src.serve.generate import GeneratorResult, visible_text
from src.serve.hybrid import BM25, rrf_fuse, tokenize
from src.serve.pipeline import Pipeline, PipelineResponse
from src.serve.rerank import RankedChunk
from src.serve.retrieve import RetrievedChunk
from src.serve.router import RouterResult


# ---------------- Hybrid search ----------------


def test_tokenize_keeps_form_numbers_and_their_parts():
    assert tokenize("File Form W-10 by June") == ["file", "form", "w-10", "by", "june", "w", "10"]


def test_bm25_prefers_exact_form_number():
    bm25 = BM25([
        "General well status reporting requirements.",
        "Form W-10 oil well status report is filed annually.",
        "Form P-5 organization report.",
    ])
    assert bm25.top("W-10 filing", k=3)[0] == 1


def test_bm25_mask_excludes_documents():
    bm25 = BM25(["w-10 report", "w-10 schedule"])
    assert bm25.top("w-10", k=2, mask=np.array([False, True])) == [1]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]])
    assert max(fused, key=fused.get) == "b"
    assert set(fused) == {"a", "b", "c", "d"}


# ---------------- Generation helpers ----------------


@pytest.mark.parametrize("text, shown", [
    ("Answer [1].\n<confidence>85</confidence>", "Answer [1].\n"),
    ("Answer [1].\n<confid", "Answer [1].\n"),
    ("Answer [1].\n<c", "Answer [1].\n"),
    ("Answer with no tag", "Answer with no tag"),
])
def test_visible_text_hides_confidence_tag(text, shown):
    assert visible_text(text) == shown


# ---------------- Cache ----------------


def test_cache_key_normalizes_question_and_tracks_versions():
    v1 = {"prompts": "aaa", "corpus": "x"}
    assert normalize_question("  What is W-10?? ") == "what is w-10"
    assert cache_key("What is W-10?", v1) == cache_key("what is   w-10", v1)
    assert cache_key("What is W-10?", v1) != cache_key("What is W-10?", {**v1, "prompts": "bbb"})


def test_memory_cache_expires():
    cache = AnswerCache(ttl_seconds=-1)
    cache.set("k", {"a": 1})
    assert cache.get("k") is None


def test_unreachable_redis_falls_back_to_memory():
    cache = AnswerCache(redis_url="redis://127.0.0.1:1/0")
    assert cache.backend == "memory"
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}


# ---------------- Pipeline with fakes ----------------


def _chunk(cid: str, doc: str = "ola001k_oil_well_status_w10") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, doc_name=doc, section_path=["root", "Sec"],
                          page_num=1, text=f"text {cid}", token_count=3, score=0.5)


class FakeRouter:
    def __init__(self, result: RouterResult | None = None):
        self.result = result or RouterResult(intent="lookup", is_ood=False, ood_confidence=0.0,
                                             doc_scope=["W-10"], raw_response="{}")

    def route(self, question):
        return self.result


class FakeRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def doc_names(self):
        return ["ola001k_oil_well_status_w10", "p5_organization"]

    def fingerprint(self):
        return "corpus-v1"

    def retrieve(self, query_vector, top_k=20, doc_names=None, query_text=None, mode="vector"):
        self.queries.append(query_text)
        prefix = "r" if "rewritten" in (query_text or "") else "c"
        return [_chunk(f"{prefix}{i}") for i in range(3)]


class FakeReranker:
    def __init__(self):
        self.queries: list[str] = []

    def rerank(self, query, candidates, top_k=5):
        self.queries.append(query)
        # Prefer retry chunks when present, so the test can see them win.
        ordered = sorted(candidates, key=lambda c: not c.chunk_id.startswith("r"))
        return [RankedChunk(chunk=c, rerank_score=1.0) for c in ordered[:top_k]]


class FakeGenerator:
    model = "fake-model"

    def __init__(self, confidence: int = 90):
        self.confidence = confidence
        self.calls = 0

    def generate(self, question, ranked):
        self.calls += 1
        return GeneratorResult(answer="The W-10 is filed annually [1].",
                               raw_confidence=self.confidence, context_passages=[],
                               usage={"input_tokens": 10, "output_tokens": 5})

    def stream(self, question, ranked):
        gen = self

        class _Stream:
            result = None

            def __iter__(self):
                gen.calls += 1
                for part in ["The W-10 is ", "filed annually [1].", "\n<confid", "ence>90</confidence>"]:
                    yield part
                self.result = GeneratorResult(answer="The W-10 is filed annually [1].",
                                              raw_confidence=gen.confidence, context_passages=[],
                                              usage={})
        return _Stream()


class FakeEmbedder:
    def embed_query(self, text):
        return np.zeros(4, dtype=np.float32)


class FakeGrader:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def grade(self, question, ranked):
        return self.verdicts.pop(0)


class FakeRewriter:
    def rewrite(self, question, reason):
        return "rewritten W-10 query"


def make_pipeline(**kwargs) -> Pipeline:
    defaults = dict(router=FakeRouter(), retriever=FakeRetriever(), reranker=FakeReranker(),
                    generator=FakeGenerator(), embedder=FakeEmbedder())
    return Pipeline(**{**defaults, **kwargs})


def test_pipeline_autonomous_answer_with_scope_and_trace():
    resp = make_pipeline().run("When is the W-10 due?")
    assert resp.decision == "autonomous"
    assert resp.doc_scope == ["ola001k_oil_well_status_w10"]
    assert [s["stage"] for s in resp.trace] == ["route", "retrieve", "rerank", "generate"]
    assert resp.retrieval_sufficient is None    # agent loop off


def test_passthrough_reranker_keeps_top_k_by_retrieval_score():
    from src.serve.rerank import PassthroughReranker
    chunks = [RetrievedChunk(chunk_id=f"c{i}", doc_name="d", section_path=["root"], page_num=1,
                             text="t", token_count=1, score=s)
              for i, s in enumerate([0.2, 0.9, 0.5])]
    ranked = PassthroughReranker().rerank("q", chunks, top_k=2)
    assert [rc.chunk.chunk_id for rc in ranked] == ["c1", "c2"]
    assert [rc.rerank_score for rc in ranked] == [0.9, 0.5]


def test_pipeline_without_reranker():
    from src.serve.rerank import PassthroughReranker
    pipe = make_pipeline(reranker=None, rerank=False)
    assert isinstance(pipe._reranker, PassthroughReranker)    # cross-encoder never built
    resp = pipe.run("When is the W-10 due?")
    assert resp.decision == "autonomous"
    assert [rc.chunk.chunk_id for rc in resp.ranked_chunks] == ["c0", "c1", "c2"]
    assert "off" in next(s["detail"] for s in resp.trace if s["stage"] == "rerank")
    assert pipe.settings["rerank"] is False
    # A different reranker setting must not share cache entries.
    assert pipe.versions()["models"] != make_pipeline().versions()["models"]


def test_pipeline_escalates_low_confidence():
    resp = make_pipeline(generator=FakeGenerator(confidence=40)).run("q?")
    assert resp.decision == "escalate" and resp.refused


def test_pipeline_refuses_ood_without_retrieval():
    router = FakeRouter(RouterResult("ood", True, 0.95, None, "{}"))
    retriever = FakeRetriever()
    resp = make_pipeline(router=router, retriever=retriever).run("Best pizza in Austin?")
    assert resp.decision == "refuse_ood" and resp.is_ood
    assert retriever.queries == []


def test_agent_loop_skips_retry_when_sufficient():
    grader = FakeGrader([GradeResult(sufficient=True, relevant=[1], reason="ok")])
    retriever = FakeRetriever()
    resp = make_pipeline(agentic=True, grader=grader, rewriter=FakeRewriter(),
                         retriever=retriever).run("When is the W-10 due?")
    assert resp.retrieval_sufficient is True
    assert len(retriever.queries) == 1
    assert resp.search_query == "When is the W-10 due?"


def test_agent_loop_retries_once_pools_and_reranks_against_original_question():
    grader = FakeGrader([GradeResult(False, [], "wrong form"), GradeResult(True, [1], "ok")])
    retriever, reranker = FakeRetriever(), FakeReranker()
    q = "When is the W-10 due?"
    resp = make_pipeline(agentic=True, grader=grader, rewriter=FakeRewriter(),
                         retriever=retriever, reranker=reranker).run(q)

    assert retriever.queries == [q, "rewritten W-10 query"]
    assert reranker.queries == [q, q]           # never reranked against the rewrite
    assert {c.chunk_id for c in resp.retrieved_chunks} == {"c0", "c1", "c2", "r0", "r1", "r2"}
    assert resp.ranked_chunks[0].chunk.chunk_id.startswith("r")
    assert resp.search_query == "rewritten W-10 query"
    assert resp.retrieval_sufficient is True
    assert [s["step"] for s in resp.agent_steps] == ["grade", "rewrite", "grade"]


def test_cache_serves_released_answers_only():
    gen = FakeGenerator()
    pipe = make_pipeline(generator=gen, cache=AnswerCache())
    first = pipe.run("When is the W-10 due?")
    second = pipe.run("when is the w-10 due")
    assert not first.cached and second.cached
    assert gen.calls == 1
    assert second.ranked_chunks[0].chunk.chunk_id == first.ranked_chunks[0].chunk.chunk_id

    low = FakeGenerator(confidence=30)
    pipe = make_pipeline(generator=low, cache=AnswerCache())
    pipe.run("q?")
    pipe.run("q?")
    assert low.calls == 2       # escalations are recomputed


def test_query_log_writes_one_line_per_question(tmp_path):
    path = tmp_path / "queries.jsonl"
    pipe = make_pipeline(query_log=path)
    pipe.run("When is the W-10 due?")
    pipe.run("And the P-5?")
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    rec = lines[0]
    assert rec["decision"] == "autonomous"
    assert rec["ranked"][0]["doc_name"] == "ola001k_oil_well_status_w10"
    assert rec["versions"]["corpus"] == "corpus-v1"
    assert "ts" in rec


def test_stream_hides_confidence_tag_and_ends_with_guardrail_result():
    events = list(make_pipeline().stream("When is the W-10 due?"))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "final"
    streamed = "".join(e["text"] for e in events if e["event"] == "token")
    assert "<conf" not in streamed
    assert streamed.startswith("The W-10 is filed annually")
    assert events[-1]["response"]["decision"] == "autonomous"


def test_response_round_trips_through_dict():
    resp = make_pipeline().run("When is the W-10 due?")
    again = PipelineResponse.from_dict(json.loads(json.dumps(resp.to_dict())))
    assert again == resp


# ---------------- HTTP API ----------------


def test_api_ask_stream_and_health():
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    with TestClient(create_app(make_pipeline())) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["search_mode"] == "vector"

        body = client.post("/ask", json={"question": "When is the W-10 due?"}).json()
        assert body["decision"] == "autonomous"
        assert body["citations"][0]["n"] == 1

        assert client.post("/ask", json={"question": ""}).status_code == 422

        text = client.post("/ask/stream", json={"question": "When is the W-10 due?"}).text
        assert "event: meta" in text and "event: token" in text and "event: final" in text
