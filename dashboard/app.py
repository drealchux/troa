"""
TROA dashboard: a browser front end for the serving pipeline.

    streamlit run dashboard/app.py

Answers come from src.serve.pipeline.Pipeline over the search index committed in
qdrant_local/ (or the Qdrant configured in .env), so they match the CLI
(ask.py), the HTTP API, and the eval harness.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import altair as alt  # noqa: E402
import anthropic  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from src.config import Settings, load_calibrator  # noqa: E402
from src.serve.cache import AnswerCache  # noqa: E402
from src.serve.guardrail import (  # noqa: E402
    DECISIONS, OOD_CUTOFF, THRESHOLD_AUTONOMOUS, THRESHOLD_CAVEAT)
from src.serve.pipeline import Pipeline, PipelineResponse  # noqa: E402

EVAL_SET_PATH = REPO / "eval_data" / "eval_set_sample.yaml"
RESULTS_DIR = REPO / "eval_data"

st.set_page_config(page_title="TROA", page_icon="🛢️", layout="wide")

# Palette (dataviz reference): categorical slots in fixed order for pipeline
# stages; reserved status colors for guardrail decisions, always paired with
# an icon + label so color never carries meaning alone.
STAGES = ["route", "retrieve", "rerank", "grade", "rewrite",
          "retrieve (retry)", "rerank (retry)", "grade (retry)", "generate"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#7a5af8", "#1baf7a", "#eda100",
               "#e87ba4", "#6b7280", "#c2185b", "#008300"]
SERIES_1 = CATEGORICAL[0]
DECISION_STYLE = {
    "autonomous": ("#0ca30c", "✅"),
    "caveat": ("#fab219", "⚠️"),
    "escalate": ("#ec835a", "⏫"),
    "refuse_ood": ("#d03b3b", "⛔"),
}
CATEGORY_TARGETS = {   # from EVALUATION.md
    "single_doc_factual": 50, "multi_doc_synthesis": 50, "procedural": 40,
    "definitional": 40, "ood": 20,
}


# ---------- Cached resources ----------

@st.cache_resource(show_spinner="Loading TROA (the first run downloads the 1.34 GB embedding model)…")
def components() -> dict:
    """Heavy components, built once and shared by every settings combination.

    Qdrant's local mode allows one client per folder, so all pipelines share
    one Retriever rather than opening their own.
    """
    from src.ingest.embed import Embedder
    from src.serve.agent import Grader, Rewriter
    from src.serve.generate import Generator
    from src.serve.rerank import Reranker
    from src.serve.retrieve import Retriever
    from src.serve.router import Router

    s = Settings.from_env()
    key = s.anthropic_api_key
    embedder = Embedder()
    _ = embedder.model   # load now, behind the spinner (a bare expression would be rendered)
    return {
        "settings": s,
        "retriever": Retriever(url=s.qdrant_url, path=s.qdrant_path),
        "embedder": embedder,
        "reranker": Reranker(),          # model loads only if reranking is switched on
        "router": Router(api_key=key),
        "generator": Generator(api_key=key),
        "grader": Grader(api_key=key),
        "rewriter": Rewriter(api_key=key),
        "calibrator": load_calibrator(s.calibration_path),
    }


@st.cache_resource
def answer_cache() -> AnswerCache:
    return AnswerCache()


@st.cache_resource
def get_pipeline(search_mode: str, agentic: bool, rerank: bool, use_cache: bool) -> Pipeline:
    c = components()
    return Pipeline(
        search_mode=search_mode, agentic=agentic, rerank=rerank,
        calibrator=c["calibrator"], cache=answer_cache() if use_cache else None,
        retriever=c["retriever"], embedder=c["embedder"], reranker=c["reranker"],
        router=c["router"], generator=c["generator"],
        grader=c["grader"], rewriter=c["rewriter"],
    )


@st.cache_resource
def corpus_stats() -> dict:
    return components()["retriever"].doc_stats()


@st.cache_data
def manual_descriptions() -> dict[str, str]:
    sys.path.insert(0, str(REPO / "data"))
    try:
        import download_data
    except ImportError:
        return {}
    return {Path(d.filename).stem: d.description for d in download_data.MANUALS}


@st.cache_data
def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def decision_badge(decision: str) -> str:
    color, icon = DECISION_STYLE[decision]
    return (f"<span style='background:{color}22;border:1px solid {color};border-radius:6px;"
            f"padding:4px 10px;font-weight:600'>{icon} {DECISIONS[decision]}</span>")


# ---------- Sidebar ----------

try:
    comps = components()
except Exception as exc:   # most often: the index folder is open in another process
    st.error(f"Could not open the search index: {exc}")
    st.info("Only one program can open `qdrant_local/` at a time. Close `ask.py`, the API, "
            "or another dashboard, then reload. To build the index yourself, see README, "
            "Quick start.")
    st.stop()
settings: Settings = comps["settings"]
api_key = settings.anthropic_api_key

with st.sidebar:
    st.title("🛢️ TROA")
    st.caption("Texas Oil & Gas Regulatory Operations Assistant")

    st.subheader("Status")
    st.write(("✅" if api_key else "❌") + " Anthropic API key")
    st.write(f"✅ {len(corpus_stats())} documents in the search index")
    st.write(("✅ Calibrated confidence" if comps["calibrator"] is not None
              else "ℹ️ Raw confidence (no calibrator loaded)"))

    st.subheader("Search")
    hybrid = st.toggle("Hybrid search (keyword + semantic)", value=settings.search_mode == "hybrid",
                       help="Fuses BM25 keyword search with semantic search. Keyword search "
                            "catches exact form numbers like W-10.")
    agentic = st.toggle("Grade passages, retry with a rewritten query", value=settings.agentic,
                        help="A Haiku grader checks whether the passages can answer the question. "
                             "If not, the query is rewritten and retrieval runs once more.")
    rerank = st.toggle("Rerank with cross-encoder", value=settings.rerank,
                       help="bge-reranker-large: a 2.24 GB download on first use. Off keeps the "
                            "top 5 by retrieval score.")
    use_cache = st.toggle("Answer cache (exact match)", value=True)

    st.subheader("Guardrail policy")
    st.markdown(
        f"- ≥ {THRESHOLD_AUTONOMOUS:.2f} → ✅ autonomous\n"
        f"- {THRESHOLD_CAVEAT:.2f}–{THRESHOLD_AUTONOMOUS:.2f} → ⚠️ caveat\n"
        f"- < {THRESHOLD_CAVEAT:.2f} → ⏫ escalate\n"
        f"- router out-of-scope ≥ {OOD_CUTOFF:.2f} → ⛔ refuse"
    )
    st.caption("Not for compliance decisions. Check every answer against its cited source.")

pipe = get_pipeline("hybrid" if hybrid else "vector", agentic, rerank, use_cache)
opts = {"search_mode": "hybrid" if hybrid else "vector", "agentic": agentic, "rerank": rerank}

st.session_state.setdefault("history", [])

tab_ask, tab_corpus, tab_eval, tab_log = st.tabs(
    ["💬 Ask", "📚 Corpus", "🧪 Evaluation", "📈 Session log"])


# ---------- Ask ----------

def render_result(res: PipelineResponse, answer_slot=None) -> None:
    (answer_slot or st).markdown(res.answer)
    if res.decision == "escalate" and res.draft_answer:
        with st.expander("Withheld draft answer (below the confidence threshold)"):
            st.markdown(res.draft_answer)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{res.calibrated_confidence:.2f}",
              help=f"Raw self-rated confidence: {res.raw_confidence}/100")
    c2.metric("Intent", res.intent or "—")
    c3.metric("Latency", "cached" if res.cached else f"{res.latency_s:.1f}s")
    c4.metric("Tokens (in / out)",
              f"{res.usage.get('input_tokens', 0)} / {res.usage.get('output_tokens', 0)}")

    if res.search_query and res.search_query != res.question:
        st.info(f"🔁 First retrieval graded insufficient. Rewritten query: *{res.search_query}*")
    if res.doc_scope:
        st.caption("Router narrowed the search to: " + ", ".join(res.doc_scope))

    if res.ranked_chunks:
        st.subheader("Sources")
        for i, rc in enumerate(res.ranked_chunks, start=1):
            c = rc.chunk
            ranks = [f"vector #{c.vector_rank}" if c.vector_rank else "",
                     f"BM25 #{c.bm25_rank}" if c.bm25_rank else ""]
            with st.expander(f"[{i}] {c.doc_name} · page {c.page_num} · "
                             f"{' · '.join(r for r in ranks if r) or 'n/a'}"):
                st.caption(" > ".join(c.section_path[1:]) + f" · score {rc.rerank_score:.4f}")
                st.text(c.text)

    if res.trace:
        with st.expander("Pipeline trace"):
            st.dataframe(pd.DataFrame(res.trace).assign(seconds=lambda d: d["seconds"].round(2)),
                         hide_index=True, width="stretch")


with tab_ask:
    st.header("Ask a regulatory question")
    examples = [q["question"] for q in load_eval_set()]
    picked = st.selectbox("Try an example question, or type your own below",
                          ["—"] + examples, index=0)
    question = st.text_area("Question", value="" if picked == "—" else picked, height=80,
                            placeholder="e.g. What does the Drilling Permit Master dataset contain?")
    go = st.button("Ask TROA", type="primary", disabled=not (api_key and question.strip()))
    if not api_key:
        st.info("Add ANTHROPIC_API_KEY to .env (copy .env.example) and restart to enable answers.")

    if go:
        st.divider()
        badge_slot, answer_slot = st.empty(), st.empty()
        res = None
        try:
            text = ""
            with st.spinner("Routing and searching…"):
                events = pipe.stream(question.strip())
                first = next(events)
            for ev in itertools.chain([first], events):
                if ev["event"] == "token":
                    text += ev["text"]
                    answer_slot.markdown(text + " ▌")
                elif ev["event"] == "final":
                    res = PipelineResponse.from_dict(ev["response"])
            badge_slot.markdown(decision_badge(res.decision)
                                + (" &nbsp; ⚡ served from cache" if res.cached else ""),
                                unsafe_allow_html=True)
            render_result(res, answer_slot)    # final answer replaces the streamed draft
        except anthropic.AuthenticationError:
            st.error("The Anthropic API rejected the key in .env (401). Check ANTHROPIC_API_KEY.")
        except anthropic.APIError as exc:
            st.error(f"Anthropic API error: {exc}")
        if res:
            st.session_state["history"].append(res)
            st.session_state["last"] = res
    elif st.session_state.get("last"):
        res = st.session_state["last"]
        st.divider()
        st.markdown(decision_badge(res.decision), unsafe_allow_html=True)
        render_result(res)


# ---------- Corpus ----------

with tab_corpus:
    st.header("Corpus")
    descriptions = manual_descriptions()
    stats = pd.DataFrame([
        {"Document": d, "Description": descriptions.get(d, ""),
         "Chunks": s["chunks"], "Pages": s["pages"], "Tokens": s["tokens"]}
        for d, s in corpus_stats().items()
    ])

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(stats))
    c2.metric("Chunks", f"{int(stats['Chunks'].sum()):,}")
    c3.metric("Tokens (approx.)", f"{int(stats['Tokens'].sum()):,}")

    chart = (
        alt.Chart(stats)
        .mark_bar(color=SERIES_1, cornerRadiusEnd=4, height=12)
        .encode(
            x=alt.X("Chunks:Q", title="Chunks"),
            y=alt.Y("Document:N", sort="-x", title=None),
            tooltip=["Document", "Description", "Pages", "Chunks", "Tokens"],
        )
        .properties(title="Chunks per document", height=max(240, 18 * len(stats)))
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(stats.sort_values("Chunks", ascending=False), hide_index=True, width="stretch")
    st.caption("Manuals: section-aware chunks of up to 512 tokens. Statewide Rules: one chunk "
               "per rule subsection. Embeddings: bge-large-en-v1.5. "
               "Pages = highest page number among a document's chunks.")


# ---------- Evaluation ----------

def score_case(case: dict, res: PipelineResponse) -> dict:
    gt_docs = {Path(c["document"]).stem for c in case.get("ground_truth_chunks") or []}
    docs = [rc.chunk.doc_name for rc in res.ranked_chunks]
    ranks = [i for i, d in enumerate(docs, start=1) if d in gt_docs]
    return {
        "id": case["id"], "category": case["category"], "question": case["question"],
        "decision": DECISIONS[res.decision], "confidence": res.calibrated_confidence,
        "refused": res.decision in ("escalate", "refuse_ood"),
        "is_ood": case["category"] == "ood",
        "hit@5": bool(ranks) if gt_docs else None,
        "rr": (1 / ranks[0] if ranks else 0.0) if gt_docs else None,
        "rewritten": bool(res.search_query) and res.search_query != res.question,
        "top_doc": docs[0] if docs else "",
        "latency_s": round(res.latency_s, 2),
        "answer": res.answer,
    }


with tab_eval:
    st.header("Evaluation")
    cases = load_eval_set()

    st.subheader("Eval set coverage")
    counts = Counter(c["category"] for c in cases)
    cov = pd.DataFrame([{"Category": k, "Questions": counts.get(k, 0), "Target": v,
                         "Progress": counts.get(k, 0) / v} for k, v in CATEGORY_TARGETS.items()])
    st.dataframe(cov, hide_index=True, width="stretch", column_config={
        "Progress": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption(f"{len(cases)} of {sum(CATEGORY_TARGETS.values())} planned questions written "
               f"({EVAL_SET_PATH.relative_to(REPO).as_posix()}).")

    saved_files = sorted(RESULTS_DIR.glob("results_*.jsonl"))
    if saved_files:
        st.subheader("Saved harness runs")
        chosen = st.selectbox("Results file", saved_files, format_func=lambda p: p.name)
        saved = [json.loads(line) for line in chosen.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        errors = [r for r in saved if r.get("error")]
        if saved and len(errors) == len(saved):
            st.warning(f"All {len(saved)} cases errored, so there are no usable metrics. "
                       f"First error: `{errors[0]['error'][:120]}`")
        else:
            cols = [c for c in ("case_id", "category", "decision", "raw_confidence",
                                "judge_correctness", "judge_faithfulness", "recall_at_5",
                                "latency_s", "error") if any(c in r for r in saved)]
            st.dataframe(pd.DataFrame(saved)[cols], hide_index=True, width="stretch")

    st.subheader("Run the sample eval with the current settings")
    st.caption(f"Runs all {len(cases)} questions with the sidebar settings (no judge; use "
               "`python tasks.py eval` for answer-quality scores). Hit@5 counts a case as "
               "retrieved if any ground-truth document is among the 5 passages used.")
    if st.button("Run eval", disabled=not api_key):
        eval_pipe = get_pipeline(opts["search_mode"], agentic, rerank, False)   # never cached
        rows = []
        bar = st.progress(0.0)
        for i, case in enumerate(cases):
            bar.progress(i / len(cases), text=f"{case['id']}: {case['question'][:60]}")
            try:
                rows.append(score_case(case, eval_pipe.run(case["question"])))
            except anthropic.APIError as exc:
                rows.append({"id": case["id"], "category": case["category"],
                             "question": case["question"], "decision": f"error: {exc}"})
        bar.empty()
        st.session_state["eval_rows"] = rows
        st.session_state["eval_settings"] = opts

    rows = st.session_state.get("eval_rows")
    if rows:
        df = pd.DataFrame(rows)
        ok = df[~df["decision"].astype(str).str.startswith("error")]
        in_scope = ok[~ok["is_ood"].astype(bool)]
        ood = ok[ok["is_ood"].astype(bool)]
        retr = in_scope.dropna(subset=["hit@5"])
        st.caption("Settings: " + json.dumps(st.session_state["eval_settings"]))

        def tile(col, label, value, target, higher_better=True):
            if value is None:
                col.metric(label, "—")
                return
            passed = value >= target if higher_better else value <= target
            col.metric(label, f"{value:.2f}",
                       delta=f"{'✅ meets' if passed else '❌ misses'} target "
                             f"{'≥' if higher_better else '≤'} {target}",
                       delta_color="off")

        c1, c2, c3, c4 = st.columns(4)
        tile(c1, "Hit@5 (document)", retr["hit@5"].astype(float).mean() if len(retr) else None, 0.85)
        tile(c2, "MRR (document)", retr["rr"].astype(float).mean() if len(retr) else None, 0.55)
        tile(c3, "Out-of-scope refusal rate",
             ood["refused"].astype(float).mean() if len(ood) else None, 0.95)
        tile(c4, "In-scope refusal rate",
             in_scope["refused"].astype(float).mean() if len(in_scope) else None,
             0.10, higher_better=False)

        st.dataframe(df.drop(columns=["answer"], errors="ignore"), hide_index=True,
                     width="stretch",
                     column_config={"confidence": st.column_config.NumberColumn(format="%.2f"),
                                    "rr": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download results (JSONL)",
                           "\n".join(json.dumps(r, default=str) for r in rows),
                           file_name="results_dashboard.jsonl")


# ---------- Session log ----------

with tab_log:
    st.header("Session log")
    history: list[PipelineResponse] = st.session_state["history"]
    if not history:
        st.info("Ask a question to start the log.")
    else:
        log = pd.DataFrame([{
            "#": i, "Question": h.question, "Decision": DECISIONS[h.decision],
            "Confidence": h.calibrated_confidence, "Intent": h.intent,
            "Rewritten": bool(h.search_query) and h.search_query != h.question,
            "Cached": h.cached, "Latency (s)": 0.0 if h.cached else round(h.latency_s, 2),
        } for i, h in enumerate(history, start=1)])

        st.subheader("Confidence vs guardrail thresholds")
        domain = [DECISIONS[k] for k in DECISION_STYLE]
        colors = [c for c, _ in DECISION_STYLE.values()]
        points = (
            alt.Chart(log)
            .mark_circle(size=140, stroke="white", strokeWidth=2, opacity=1)
            .encode(
                x=alt.X("#:O", title="Query"),
                y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 1]), title="Confidence"),
                color=alt.Color("Decision:N", scale=alt.Scale(domain=domain, range=colors),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["#", "Question", "Decision", alt.Tooltip("Confidence:Q", format=".2f"),
                         "Intent", "Latency (s)"],
            )
        )
        rules = alt.Chart(pd.DataFrame({
            "y": [THRESHOLD_CAVEAT, THRESHOLD_AUTONOMOUS],
            "label": ["caveat threshold", "autonomous threshold"],
        }))
        lines = rules.mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        labels = rules.mark_text(align="left", dx=4, dy=-6, color="gray").encode(
            y="y:Q", x=alt.value(0), text="label:N")
        st.altair_chart((lines + labels + points).properties(height=300), width="stretch")

        stage_rows = [{"#": i, "Stage": s["stage"], "Seconds": round(s["seconds"], 2)}
                      for i, h in enumerate(history, start=1) if not h.cached for s in h.trace]
        if stage_rows:
            st.subheader("Latency by pipeline stage")
            stages = (
                alt.Chart(pd.DataFrame(stage_rows))
                .mark_bar(height=14, stroke="white", strokeWidth=2)
                .encode(
                    y=alt.Y("#:O", title="Query"),
                    x=alt.X("sum(Seconds):Q", title="Seconds"),
                    color=alt.Color("Stage:N", scale=alt.Scale(domain=STAGES, range=CATEGORICAL),
                                    legend=alt.Legend(orient="top", title=None)),
                    order=alt.Order("stage_order:Q"),
                    tooltip=["#", "Stage", "Seconds"],
                )
                .transform_calculate(stage_order=f"indexof({json.dumps(STAGES)}, datum.Stage)")
                .properties(height=max(120, 28 * len(history)))
            )
            st.altair_chart(stages, width="stretch")

        st.dataframe(log, hide_index=True, width="stretch",
                     column_config={"Confidence": st.column_config.NumberColumn(format="%.2f")})
