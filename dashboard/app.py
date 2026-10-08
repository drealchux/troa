"""
TROA dashboard.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time
from collections import Counter

import altair as alt
import anthropic
import pandas as pd
import streamlit as st

import engine
from src.serve.cache import CACHEABLE_DECISIONS, normalize_question, prompts_fingerprint

CACHE_PROMPTS = ["generate_v1.yaml", "router_v1.yaml", "grader_v1.yaml",
                 "rewrite_v1.yaml", "caveats.yaml"]

st.set_page_config(page_title="TROA Dashboard", page_icon="🛢️", layout="wide")

# Palette (dataviz reference): categorical slots in fixed order for pipeline
# stages; reserved status colors for guardrail decisions, always paired with
# an icon + label so color never carries meaning alone.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STAGES = ["route", "retrieve", "grade", "rewrite", "retrieve (retry)", "generate"]
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
MODEL_OPTIONS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"]


# ---------- Cached resources ----------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(engine.EMBED_MODEL)


@st.cache_resource
def index_store() -> dict:
    """Shared store of built indexes, keyed by corpus. A plain dict rather than a
    cached builder, because the builder draws a progress bar that Streamlit can't
    replay on cache hits."""
    return {}


def get_index(doc_names: tuple[str, ...], progress) -> engine.Index:
    store = index_store()
    if doc_names not in store:
        paths = [engine.MANUAL_DIR / n for n in doc_names]
        store[doc_names] = engine.build_index(paths, get_embedder(), progress=progress)
    return store[doc_names]


@st.cache_resource
def get_client(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def answer_cache() -> dict:
    """Exact-match answer cache shared across sessions, keyed on question + settings."""
    return {}


@st.cache_data
def load_eval_set() -> list[dict]:
    return engine.load_yaml(engine.EVAL_SET_PATH)["questions"]


def load_saved_results() -> list[dict]:
    if not engine.EVAL_RESULTS_PATH.exists():
        return []
    return [json.loads(l) for l in engine.EVAL_RESULTS_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def decision_badge(decision: str) -> str:
    color, icon = DECISION_STYLE[decision]
    return (f"<span style='background:{color}22;border:1px solid {color};border-radius:6px;"
            f"padding:4px 10px;font-weight:600'>{icon} {engine.DECISIONS[decision]}</span>")


def visible_stream_text(text: str) -> str:
    """Hide the trailing <confidence> tag while the answer is still streaming."""
    cut = text.lower().find("<conf")
    return text if cut == -1 else text[:cut]


# ---------- Sidebar ----------

api_key = engine.load_api_key()
manuals = engine.list_manuals()
descriptions = engine.manual_descriptions()
default_model = engine.load_prompt("generate_v1.yaml")["model"]

with st.sidebar:
    st.title("🛢️ TROA")
    st.caption("Texas Oil & Gas Regulatory Operations Assistant")

    st.subheader("Status")
    st.write(("✅" if api_key else "❌") + " Anthropic API key")
    st.write(("✅" if manuals else "❌") + f" {len(manuals)} RRC manuals downloaded")

    st.subheader("Retrieval")
    selected_docs = st.multiselect(
        "Corpus", [m.name for m in manuals], default=[m.name for m in manuals],
        help="Manuals to search. New manuals are embedded on first use and cached "
             "in data/processed/dashboard/.",
    )
    mode = st.radio("Search mode", list(engine.SEARCH_MODES),
                    format_func=engine.SEARCH_MODES.get,
                    help="Hybrid fuses keyword (BM25) and semantic rankings with reciprocal "
                         "rank fusion. Keyword search catches exact form numbers like W-10.")
    top_k = st.slider("Passages retrieved (top-k)", 3, 15, 5)

    st.subheader("Agent")
    use_router = st.toggle("Router (intent + out-of-scope screen)", value=True)
    agentic = st.toggle("Grade passages, rewrite query on miss", value=True,
                        help="A Haiku grader checks whether the passages can answer the question. "
                             "If not, the query is rewritten in RRC terminology and retrieval "
                             "runs once more.")
    use_cache = st.toggle("Answer cache (exact match)", value=True)
    model = st.selectbox("Answer model",
                         [default_model] + [m for m in MODEL_OPTIONS if m != default_model],
                         help="Default comes from src/serve/prompts/generate_v1.yaml")

    st.subheader("Guardrail policy")
    st.markdown(
        f"- ≥ {engine.THRESHOLD_AUTONOMOUS:.2f} → ✅ autonomous\n"
        f"- {engine.THRESHOLD_CAVEAT:.2f}–{engine.THRESHOLD_AUTONOMOUS:.2f} → ⚠️ caveat\n"
        f"- < {engine.THRESHOLD_CAVEAT:.2f} → ⏫ escalate\n"
        f"- router OOD ≥ {engine.OOD_CUTOFF:.2f} → ⛔ refuse"
    )
    st.caption("Confidence is raw (÷100): no Platt calibrator has been fitted yet.")

if not manuals:
    st.error("No manuals found. Run `python data/download_data.py` first.")
    st.stop()
if not selected_docs:
    st.warning("Select at least one manual in the sidebar.")
    st.stop()

bar_slot = st.empty()
progress_bar = bar_slot.progress(0.0, text="Preparing index…")
index = get_index(tuple(selected_docs), lambda f, msg: progress_bar.progress(f, text=msg))
bar_slot.empty()
embedder = get_embedder()
client = get_client(api_key) if api_key else None
opts = dict(top_k=top_k, mode=mode, use_router=use_router, agentic=agentic)

st.session_state.setdefault("history", [])

tab_ask, tab_corpus, tab_eval, tab_log = st.tabs(
    ["💬 Ask", "📚 Corpus", "🧪 Evaluation", "📈 Session log"])


# ---------- Ask ----------

def render_result(res: engine.RunResult, answer_slot=None) -> None:
    (answer_slot or st).markdown(res.answer)
    if res.decision == "escalate" and res.draft_answer:
        with st.expander("Withheld draft answer (below confidence threshold)"):
            st.markdown(res.draft_answer)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{res.confidence:.2f}")
    c2.metric("Intent", res.route.intent if res.route.raw else "—")
    c3.metric("Latency", "cached" if res.cached else f"{res.latency:.1f}s")
    c4.metric("Tokens (in / out)",
              f"{res.usage.get('input_tokens', 0)} / {res.usage.get('output_tokens', 0)}")

    if res.search_query != res.question:
        st.info(f"🔁 First retrieval graded insufficient. Rewritten query: *{res.search_query}*")
    if res.scope_applied:
        st.caption("Router narrowed search to: " + ", ".join(res.scope_applied))

    if res.passages:
        st.subheader("Sources")
        for i, p in enumerate(res.passages, start=1):
            ranks = []
            if p.get("vector_rank"):
                ranks.append(f"vector #{p['vector_rank']}")
            if p.get("bm25_rank"):
                ranks.append(f"BM25 #{p['bm25_rank']}")
            with st.expander(f"[{i}] {p['document']} · page {p['page']} · "
                             f"{' · '.join(ranks) or 'n/a'}"):
                st.caption(f"cosine {p['vector_sim']:.3f} · BM25 {p['bm25']:.2f} · "
                           f"RRF {p['score']:.4f}")
                st.text(p["text"])

    with st.expander("Pipeline trace"):
        st.dataframe(pd.DataFrame(res.trace).assign(seconds=lambda d: d["seconds"].round(2)),
                     hide_index=True, width="stretch")


with tab_ask:
    st.header("Ask a regulatory question")
    examples = [q["question"] for q in load_eval_set()]
    picked = st.selectbox("Try an eval-set question, or type your own below",
                          ["—"] + examples, index=0)
    question = st.text_area("Question", value="" if picked == "—" else picked, height=80,
                            placeholder="e.g. What does the Drilling Permit Master dataset contain?")
    go = st.button("Ask TROA", type="primary", disabled=not (api_key and question.strip()))
    if not api_key:
        st.info("Add ANTHROPIC_API_KEY to .env to enable answers.")

    if go:
        q = question.strip()
        # Prompt versions are part of the key, so editing a prompt never serves a stale answer.
        key = json.dumps([normalize_question(q), opts, model, sorted(selected_docs),
                          prompts_fingerprint(CACHE_PROMPTS)])
        cache = answer_cache()
        st.divider()
        if use_cache and key in cache:
            res = engine.RunResult(**{**cache[key].__dict__, "cached": True})
            st.markdown(decision_badge(res.decision) + " &nbsp; ⚡ served from cache",
                        unsafe_allow_html=True)
            render_result(res)
        else:
            try:
                with st.status("Routing and retrieving…", expanded=False) as status:
                    prep = engine.prepare(q, index, embedder, client, **opts)
                    for s in prep.trace:
                        st.write(f"**{s['stage']}** ({s['seconds']:.2f}s): {s['detail']}")
                    status.update(label="Retrieved. Generating answer…", state="complete")

                badge_slot, answer_slot = st.empty(), st.empty()
                usage: dict = {}
                text, t = "", time.perf_counter()
                if not prep.refused_ood:
                    for delta in engine.stream_answer(client, prep, model, usage):
                        text += delta
                        answer_slot.markdown(visible_stream_text(text) + " ▌")
                res = engine.finalize(prep, text, usage, model, time.perf_counter() - t)
                badge_slot.markdown(decision_badge(res.decision), unsafe_allow_html=True)
                render_result(res, answer_slot)
                if res.decision in CACHEABLE_DECISIONS:   # never cache escalations or refusals
                    cache[key] = res
            except anthropic.APIError as e:
                st.error(f"Anthropic API error: {e}")
                res = None
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
    stats = pd.DataFrame([
        {"Manual": d, "Description": descriptions.get(d, ""), **s}
        for d, s in index.doc_stats.items()
    ]).rename(columns={"pages": "Pages", "chunks": "Chunks", "size_kb": "Size (KB)"})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manuals indexed", len(stats))
    c2.metric("Pages with text", int(stats["Pages"].sum()))
    c3.metric("Chunks", f"{int(stats['Chunks'].sum()):,}")
    c4.metric("Keyword vocabulary", f"{len(index.bm25.postings):,}")

    chart = (
        alt.Chart(stats)
        .mark_bar(color=SERIES_1, cornerRadiusEnd=4, height=12)
        .encode(
            x=alt.X("Chunks:Q", title="Chunks"),
            y=alt.Y("Manual:N", sort="-x", title=None),
            tooltip=["Manual", "Description", "Pages", "Chunks", "Size (KB)"],
        )
        .properties(title="Chunks per manual", height=max(240, 18 * len(stats)))
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(stats.sort_values("Chunks", ascending=False), hide_index=True,
                 width="stretch")
    st.caption(f"Chunking: {engine.CHUNK_SIZE}-char windows, {engine.CHUNK_OVERLAP} overlap · "
               f"Embeddings: {engine.EMBED_MODEL} · Keyword: BM25 (k1=1.5, b=0.75)")


# ---------- Evaluation ----------

def score_case(case: dict, res: engine.RunResult) -> dict:
    gt_docs = {c["document"] for c in case.get("ground_truth_chunks") or []}
    ranks = [i for i, p in enumerate(res.passages, start=1) if p["document"] in gt_docs]
    return {
        "id": case["id"], "category": case["category"], "question": case["question"],
        "decision": engine.DECISIONS[res.decision], "confidence": res.confidence,
        "refused": res.decision in ("escalate", "refuse_ood"),
        "is_ood": case["category"] == "ood",
        "hit@k": bool(ranks) if gt_docs else None,
        "rr": (1 / ranks[0] if ranks else 0.0) if gt_docs else None,
        "rewritten": res.search_query != res.question,
        "top_doc": res.passages[0]["document"] if res.passages else "",
        "latency_s": round(res.latency, 2),
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
               f"(eval_data/eval_set_sample.yaml).")

    saved = load_saved_results()
    if saved:
        errors = [r for r in saved if r.get("error")]
        st.subheader("Last harness run (eval_data/results_latest.jsonl)")
        if len(errors) == len(saved):
            st.warning(f"All {len(saved)} cases errored, so there are no usable metrics. "
                       f"First error: `{errors[0]['error'][:120]}`")
        else:
            st.dataframe(pd.DataFrame(saved), hide_index=True, width="stretch")

    st.subheader("Run the sample eval with the current settings")
    st.caption(f"Runs all {len(cases)} questions with the sidebar settings (2–4 API calls each). "
               "Hit@k counts a case as retrieved if any ground-truth manual is in the top-k.")
    if st.button("Run eval", disabled=not api_key):
        rows = []
        bar = st.progress(0.0)
        for i, case in enumerate(cases):
            bar.progress(i / len(cases), text=f"{case['id']}: {case['question'][:60]}")
            try:
                r = engine.run(case["question"], index, embedder, client, model=model, **opts)
                rows.append(score_case(case, r))
            except anthropic.APIError as e:
                rows.append({"id": case["id"], "category": case["category"],
                             "question": case["question"], "decision": f"error: {e}"})
        bar.empty()
        st.session_state["eval_rows"] = rows
        st.session_state["eval_settings"] = {**opts, "model": model}

    rows = st.session_state.get("eval_rows")
    if rows:
        df = pd.DataFrame(rows)
        ok = df[~df["decision"].astype(str).str.startswith("error")]
        in_scope = ok[~ok["is_ood"].astype(bool)]
        ood = ok[ok["is_ood"].astype(bool)]
        retr = in_scope.dropna(subset=["hit@k"])
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
        tile(c1, "Hit@k (manual)", retr["hit@k"].astype(float).mean() if len(retr) else None, 0.85)
        tile(c2, "MRR (manual)", retr["rr"].astype(float).mean() if len(retr) else None, 0.55)
        tile(c3, "OOD refusal rate", ood["refused"].astype(float).mean() if len(ood) else None,
             0.95)
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
    history: list[engine.RunResult] = st.session_state["history"]
    if not history:
        st.info("Ask a question to start the log.")
    else:
        log = pd.DataFrame([{
            "#": i, "Question": h.question, "Decision": engine.DECISIONS[h.decision],
            "Confidence": h.confidence, "Intent": h.route.intent,
            "Rewritten": h.search_query != h.question, "Cached": h.cached,
            "Latency (s)": 0.0 if h.cached else round(h.latency, 2),
        } for i, h in enumerate(history, start=1)])

        st.subheader("Confidence vs guardrail thresholds")
        domain = [engine.DECISIONS[k] for k in DECISION_STYLE]
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
            "y": [engine.THRESHOLD_CAVEAT, engine.THRESHOLD_AUTONOMOUS],
            "label": ["caveat threshold", "autonomous threshold"],
        }))
        lines = rules.mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        labels = rules.mark_text(align="left", dx=4, dy=-6, color="gray").encode(
            y="y:Q", x=alt.value(0), text="label:N")
        st.altair_chart((lines + labels + points).properties(height=300),
                        width="stretch")

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
