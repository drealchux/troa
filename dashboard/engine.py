"""
Lightweight TROA engine for the Streamlit dashboard.

Runs the same stages as src/serve/pipeline.py (route -> retrieve -> generate ->
guardrail) and uses the same versioned prompts, but swaps the heavy pieces for
local ones so it runs without Docker or multi-GB models:

  - Qdrant + bge-large    -> in-memory numpy index + bge-small (cached to disk)
  - bge-reranker-large    -> skipped; hybrid BM25 + vector search fused with RRF instead
  - Platt calibrator      -> none fitted yet, so raw confidence / 100 is used

On top of that it adds an optional agentic retry (grade passages -> rewrite
query -> retrieve again), following the LangGraph pattern in
jamwithai/production-agentic-rag-course.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mvp_rag import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, chunk_text, extract_pages  # noqa: E402
# Thresholds and decision labels are shared with src/serve/pipeline.py and
# re-exported for app.py.
from src.serve.guardrail import (  # noqa: E402,F401
    DECISIONS, OOD_CUTOFF, THRESHOLD_AUTONOMOUS, THRESHOLD_CAVEAT, decide, is_ood_refusal,
)
from src.serve.hybrid import BM25, RRF_K, rrf_fuse, tokenize  # noqa: E402,F401
from src.serve.scope import resolve_scope  # noqa: E402

MANUAL_DIR = ROOT / "data" / "raw" / "manual"
CACHE_DIR = ROOT / "data" / "processed" / "dashboard"
PROMPTS_DIR = ROOT / "src" / "serve" / "prompts"
EVAL_SET_PATH = ROOT / "eval_data" / "eval_set_sample.yaml"
EVAL_RESULTS_PATH = ROOT / "eval_data" / "results_latest.jsonl"

CANDIDATE_POOL = 50   # per-retriever candidates fed into fusion

SEARCH_MODES = {"hybrid": "Hybrid (BM25 + vector, RRF)", "vector": "Vector only",
                "keyword": "Keyword only (BM25)"}

_CONF_RE = re.compile(r"<confidence>\s*(\d+)\s*</confidence>", re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------- Config ----------

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> dict:
    return load_yaml(PROMPTS_DIR / name)


def load_api_key() -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value and "replace-with" not in value and not value.startswith("sk-ant-..."):
                    return value
    return None


def manual_descriptions() -> dict[str, str]:
    """filename -> description, read from the curated list in data/download_data.py."""
    src = (ROOT / "data" / "download_data.py").read_text(encoding="utf-8")
    return dict(re.findall(r'filename="([^"]+)",\s*description="([^"]+)"', src))


def list_manuals() -> list[Path]:
    return sorted(MANUAL_DIR.glob("*.pdf")) if MANUAL_DIR.exists() else []


# ---------- Keyword index ----------
# BM25, tokenize and RRF live in src/serve/hybrid.py, shared with the serving pipeline.


# ---------- Index ----------

def _doc_cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBED_MODEL}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


@dataclass
class Index:
    chunks: list[dict]
    embeddings: np.ndarray
    doc_stats: dict[str, dict]   # doc -> {pages, chunks, size_kb}
    bm25: BM25


def build_index(
    paths: list[Path],
    embedder,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Index:
    """Parse, chunk, and embed each manual; per-doc results are cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    all_embs: list[np.ndarray] = []
    stats: dict[str, dict] = {}

    for i, path in enumerate(paths):
        if progress:
            progress(i / max(len(paths), 1), f"Indexing {path.name}")
        key = _doc_cache_key(path)
        meta_path = CACHE_DIR / f"{path.stem}.{key}.json"
        emb_path = CACHE_DIR / f"{path.stem}.{key}.npy"

        if meta_path.exists() and emb_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            embs = np.load(emb_path)
        else:
            chunks = []
            pages = extract_pages(path)
            for page_num, text in pages:
                for chunk_idx, chunk in enumerate(chunk_text(text)):
                    chunks.append({"document": path.name, "page": page_num,
                                   "chunk_idx": chunk_idx, "text": chunk})
            embs = (embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                                    batch_size=32)
                    if chunks else np.zeros((0, embedder.get_sentence_embedding_dimension())))
            meta = {"pages": len(pages), "chunks": chunks}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            np.save(emb_path, embs)

        stats[path.name] = {"pages": meta["pages"], "chunks": len(meta["chunks"]),
                            "size_kb": round(path.stat().st_size / 1024)}
        all_chunks.extend(meta["chunks"])
        all_embs.append(np.asarray(embs, dtype=np.float32))

    if progress:
        progress(1.0, "Building keyword index")
    dim = embedder.get_sentence_embedding_dimension()
    embeddings = np.vstack(all_embs) if all_embs else np.zeros((0, dim), dtype=np.float32)
    return Index(chunks=all_chunks, embeddings=embeddings, doc_stats=stats,
                 bm25=BM25([c["text"] for c in all_chunks]))


def search(index: Index, query: str, query_vec: np.ndarray, k: int, mode: str = "hybrid",
           doc_scope: Optional[list[str]] = None) -> list[dict]:
    """Vector, BM25, or hybrid (reciprocal rank fusion) search over the index."""
    mask = None
    if doc_scope:
        scope = set(doc_scope)
        mask = np.array([c["document"] in scope for c in index.chunks])
        if not mask.any():
            mask = None

    def ranked(scores: np.ndarray) -> list[int]:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:CANDIDATE_POOL]
        return [int(i) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    vec_scores = index.embeddings @ query_vec
    kw_scores = index.bm25.scores(query)
    vec_rank = ranked(vec_scores) if mode in ("hybrid", "vector") else []
    kw_rank = ranked(kw_scores) if mode in ("hybrid", "keyword") else []

    fused = rrf_fuse([vec_rank, kw_rank], k=RRF_K)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    vpos = {i: r for r, i in enumerate(vec_rank, start=1)}
    kpos = {i: r for r, i in enumerate(kw_rank, start=1)}
    return [{**index.chunks[i], "score": fused[i], "vector_sim": float(vec_scores[i]),
             "bm25": float(kw_scores[i]), "vector_rank": vpos.get(i), "bm25_rank": kpos.get(i)}
            for i in top]


# ---------- LLM stages ----------

@dataclass
class RouteResult:
    intent: str = "lookup"
    is_ood: bool = False
    ood_confidence: float = 0.0
    doc_scope: Optional[list[str]] = None
    raw: str = ""


def _call(client, cfg: dict, user_msg: str) -> str:
    resp = client.messages.create(
        model=cfg["model"], max_tokens=cfg["max_tokens"],
        system=[{"type": "text", "text": cfg["system"], "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text


def _parse_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


def route(client, question: str) -> RouteResult:
    cfg = load_prompt("router_v1.yaml")
    raw = _call(client, cfg, cfg["user_template"].format(question=question))
    parsed = _parse_json(raw)
    return RouteResult(
        intent=parsed.get("intent", "lookup"),
        is_ood=bool(parsed.get("is_ood", False)),
        ood_confidence=float(parsed.get("ood_confidence", 0.0) or 0.0),
        doc_scope=parsed.get("doc_scope") or None,
        raw=raw,
    )


def format_context(passages: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({p['document']} | page {p['page']})\n{p['text']}"
                       for i, p in enumerate(passages, start=1))


def grade(client, question: str, passages: list[dict]) -> dict:
    cfg = load_prompt("grader_v1.yaml")
    parsed = _parse_json(_call(client, cfg, cfg["user_template"].format(
        question=question, context=format_context(passages))))
    return {"sufficient": bool(parsed.get("sufficient", True)),
            "relevant": parsed.get("relevant", []), "reason": parsed.get("reason", "")}


def rewrite(client, question: str, reason: str) -> str:
    cfg = load_prompt("rewrite_v1.yaml")
    return _call(client, cfg, cfg["user_template"].format(question=question, reason=reason)).strip()


# ---------- Pipeline ----------

@dataclass
class Prepared:
    question: str
    route: RouteResult
    passages: list[dict] = field(default_factory=list)
    scope_applied: list[str] = field(default_factory=list)
    search_query: str = ""
    refused_ood: bool = False
    trace: list[dict] = field(default_factory=list)   # [{stage, seconds, detail}]


@dataclass
class RunResult:
    question: str
    decision: str
    answer: str
    draft_answer: str
    raw_confidence: int
    confidence: float
    route: RouteResult
    scope_applied: list[str]
    search_query: str
    passages: list[dict]
    trace: list[dict]
    usage: dict
    model: str
    cached: bool = False

    @property
    def latency(self) -> float:
        return sum(s["seconds"] for s in self.trace)


def prepare(question: str, index: Index, embedder, client, *, top_k: int = 5,
            mode: str = "hybrid", use_router: bool = True, agentic: bool = True) -> Prepared:
    """Everything before generation: route, retrieve, and the optional grade/rewrite loop."""
    trace: list[dict] = []

    def timed(stage: str, fn, detail_fn=lambda r: ""):
        t = time.perf_counter()
        out = fn()
        trace.append({"stage": stage, "seconds": time.perf_counter() - t, "detail": detail_fn(out)})
        return out

    rt = (timed("route", lambda: route(client, question),
                lambda r: f"intent={r.intent}, ood={r.is_ood} ({r.ood_confidence:.2f}), "
                          f"scope={r.doc_scope}")
          if use_router else RouteResult())
    prep = Prepared(question, rt, search_query=question, trace=trace)
    if is_ood_refusal(rt.is_ood, rt.ood_confidence):
        prep.refused_ood = True
        return prep

    scope = resolve_scope(rt.doc_scope, index.doc_stats)
    prep.scope_applied = scope

    def do_search(q: str) -> list[dict]:
        qvec = embedder.encode([q], normalize_embeddings=True)[0]
        return search(index, q, qvec, top_k, mode, scope)

    prep.passages = timed("retrieve", lambda: do_search(question),
                          lambda p: f"{SEARCH_MODES[mode]}, {len(p)} passages")

    if agentic and prep.passages:
        verdict = timed("grade", lambda: grade(client, question, prep.passages),
                        lambda g: f"sufficient={g['sufficient']}, relevant={g['relevant']}: "
                                  f"{g['reason']}")
        if not verdict["sufficient"]:
            new_q = timed("rewrite", lambda: rewrite(client, question, verdict["reason"]),
                          lambda q: q)
            prep.search_query = new_q
            prep.passages = timed("retrieve (retry)", lambda: do_search(new_q),
                                  lambda p: f"{len(p)} passages for rewritten query")
    return prep


def stream_answer(client, prep: Prepared, model: str, usage_out: dict) -> Iterator[str]:
    """Yield answer text as it streams; fills usage_out when the stream ends."""
    cfg = load_prompt("generate_v1.yaml")
    user_msg = cfg["user_template"].format(context=format_context(prep.passages),
                                           question=prep.question)
    with client.messages.stream(
        model=model, max_tokens=cfg["max_tokens"],
        system=[{"type": "text", "text": cfg["system"], "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    usage_out.update(input_tokens=final.usage.input_tokens,
                     output_tokens=final.usage.output_tokens)


def finalize(prep: Prepared, raw_text: str, usage: dict, model: str,
             gen_seconds: float) -> RunResult:
    """Apply the confidence guardrail to a finished generation."""
    caveats = load_prompt("caveats.yaml")
    trace = prep.trace + ([{"stage": "generate", "seconds": gen_seconds, "detail": model}]
                          if not prep.refused_ood else [])
    if prep.refused_ood:
        return RunResult(prep.question, "refuse_ood", caveats["ood"].strip(), "", 0, 0.0,
                         prep.route, [], prep.search_query, [], trace, {}, model)

    m = _CONF_RE.search(raw_text)
    raw_conf = max(0, min(100, int(m.group(1)))) if m else 50
    draft = _CONF_RE.sub("", raw_text).strip()
    conf = raw_conf / 100.0   # no calibrator fitted yet

    decision = decide(conf)
    answer = {
        "autonomous": draft,
        "caveat": draft + "\n\n" + caveats["low_confidence"].strip(),
        "escalate": caveats["escalate"].strip(),
    }[decision]

    return RunResult(prep.question, decision, answer, draft, raw_conf, conf, prep.route,
                     prep.scope_applied, prep.search_query, prep.passages, trace, usage, model)


def run(question: str, index: Index, embedder, client, *, model: str, **opts) -> RunResult:
    """Non-streaming end-to-end run (used by the eval tab)."""
    prep = prepare(question, index, embedder, client, **opts)
    if prep.refused_ood:
        return finalize(prep, "", {}, model, 0.0)
    usage: dict = {}
    t = time.perf_counter()
    text = "".join(stream_answer(client, prep, model, usage))
    return finalize(prep, text, usage, model, time.perf_counter() - t)
