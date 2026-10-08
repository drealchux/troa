# TROA: Texas Oil and Gas Regulatory Operations Assistant

A retrieval-augmented generation (RAG) assistant for questions about Texas Railroad Commission (RRC) oil and gas manuals. A question goes in; TROA returns an answer with **citations** to the manual passages it used, a **confidence score**, and a **guardrail decision**: answer, answer with caveat, escalate, or refuse.

It is a portfolio project. It is not a deployed product and must not be used for compliance decisions.

> **Status (2026-10-08).** The ingestion, serving, evaluation, API, and dashboard code exists and its 65 unit tests pass. **One verifiable end-to-end run exists** (12 questions, no reranker, no answer-quality judge; [Results](#results-and-conclusions)). It misses the retrieval and in-scope refusal targets. Answer quality has not been judged. No confidence calibrator has been fitted. See [Results and conclusions](#results-and-conclusions) and [Verify it yourself](#verify-it-yourself).

---

## Contents

- [The problem](#the-problem)
- [What TROA does](#what-troa-does)
- [Why this technology](#why-this-technology)
- [Project status](#project-status)
- [Results and conclusions](#results-and-conclusions)
- [Next steps](#next-steps)
- [Verify it yourself](#verify-it-yourself)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [License](#license)

---

## The problem

RRC publishes its oil and gas information as dozens of separate PDF manuals, one per form or dataset (drilling permits, well status reports such as W-10 and G-10, operator organisation reports such as P-5, production data, and so on). The questions this project targets look like:

- "What does the Drilling Permit Master dataset contain?"
- "What's the deadline for filing the W-10?"
- "Does this proposed spacing need a Rule 37 exception?"

Two properties make this a hard fit for a plain chatbot:

1. **Answers must be traceable.** A regulatory answer is only useful if the reader can check it against the source page, so every answer cites the passages it used.
2. **A wrong answer costs more than no answer.** The system therefore needs to know when *not* to answer. TROA's design makes the confidence score, and the thresholds that turn it into an action, the central part of the system rather than an add-on. That part is built but not yet validated (no calibrator has been fitted; see [Results](#results-and-conclusions)).

---

## What TROA does

```mermaid
flowchart LR
    U(["Question"]) --> R["Router<br/>Haiku: intent, out-of-scope screen,<br/>document scope"]
    R -- "out of scope" --> X["⛔ Refuse"]
    R -- "in scope" --> S["Retrieve top 20<br/>dense (Qdrant) or<br/>hybrid BM25 + dense"]
    S --> K["Rerank to top 5<br/>cross-encoder (optional)<br/>(optional: grade, rewrite, retry once)"]
    K --> G["Generate<br/>Sonnet: cited answer +<br/>self-rated confidence 0–100"]
    G --> C["Calibrate<br/>Platt scaling, if a model is supplied"]
    C --> D{"Guardrail"}
    D -- "≥ 0.85" --> A["✅ Answer"]
    D -- "0.70–0.85" --> V["⚠️ Answer + caveat"]
    D -- "< 0.70" --> E["⏫ Escalate"]
```

Three lanes, detailed in [ARCHITECTURE.md](ARCHITECTURE.md):

1. **Ingestion (offline):** download the RRC manuals, parse them with font-size-based heading detection, chunk by section, embed with `bge-large-en-v1.5`, and store in Qdrant.
2. **Serving (online):** router → retriever → cross-encoder reranker → Claude generator with citations and confidence → optional calibrator → guardrail. Exposed as a Python class (`Pipeline`) and an HTTP API.
3. **Evaluation:** a harness runs an eval set through the pipeline, scores retrieval and refusals, has Claude Opus judge answer quality, and writes JSONL that the Platt-scaling calibrator can be fitted on.

A **Streamlit dashboard** runs a lighter version of the serving lane in one process (smaller embedding model, no Qdrant, no reranker) for demos and quick ablations.

---

## Why this technology

Each row says what is used, why, and where to check it. "Design rationale" means the reason is the author's stated intent, not something measured in this repository.

| Concern | Choice | Why | Where |
|---|---|---|---|
| PDF parsing | `pypdf` with its text visitor | The visitor reports the font size of each text run. Headings in the manuals are usually larger than body text, so font size is the main signal for finding section boundaries. Falls back to plain text extraction when the visitor fails. | `src/ingest/parse.py` |
| Chunking | Section-aware, max 512 tokens, 64-token overlap (tokens estimated as characters / 4) | Design rationale: a chunk that spans two record layouts confuses both retrieval and citation. Not measured here; an earlier pilot result cited in the design is not reproducible from the repo. | `src/ingest/chunk.py` |
| Embeddings | `BAAI/bge-large-en-v1.5` (1024-d, MIT licence); `bge-small-en-v1.5` in the dashboard | `bge-large` for the reference pipeline. `bge-small` weights are about 0.13 GB against 1.34 GB for `bge-large`, which keeps the dashboard workable on a CPU laptop. | `src/ingest/embed.py`, `mvp_rag.py` |
| Vector store | Qdrant (`qdrant-client` 1.19.1; server image pinned to v1.19.1) | Payload filtering on `doc_name` lets the router restrict a search to the manual a question names. Local-file mode (`path=`) needs no server; the same client talks to a server in Docker. | `src/ingest/store.py`, `src/serve/retrieve.py`, `compose.yml` |
| Keyword search | In-process BM25 fused with dense results by reciprocal rank fusion (RRF) | RRC questions hinge on exact identifiers (`W-10`, `P-5`, `OGA049`) that dense embeddings blur. RRF merges two rankings without putting their scores on one scale. The corpus is small (2,180 chunks in the committed store), so BM25 runs in memory instead of in a second search engine. | `src/serve/hybrid.py` |
| Reranker | `BAAI/bge-reranker-large` cross-encoder | Design rationale: cheaper and faster than asking an LLM to rerank. Not benchmarked here, so it is optional: `rerank=False` (`--no-rerank`, `TROA_RERANK=false`) skips it and keeps the top 5 by retrieval score. On by default. | `src/serve/rerank.py` |
| LLMs | Claude Haiku 4.5 (router, grader, query rewriter), Sonnet 4.6 (answers), Opus 4.7 (judge) | Small model for cheap classification steps, larger model for the answer, and a different model as judge so the generator does not grade itself. System prompts are sent with `cache_control` so repeated calls reuse the prompt cache. | `src/serve/prompts/*.yaml`, `src/eval/harness.py` |
| Confidence | Self-reported `<confidence>0–100</confidence>` tag in the same call, then Platt scaling (scikit-learn `LogisticRegression`) | The cheapest available signal (no extra call). Raw LLM self-ratings are not trustworthy on their own, which is why a calibrator fitted on judged outcomes is part of the design. | `src/serve/generate.py`, `src/eval/calibration.py` |
| HTTP API | FastAPI + Uvicorn | Typed request/response models (Pydantic), Server-Sent Events for streaming, and generated docs at `/docs`. | `src/api/app.py` |
| Answer cache | Redis if `REDIS_URL` is set, otherwise in-process memory | Optional. Any Redis error falls back to memory. Only released answers (autonomous or caveat) are cached. | `src/serve/cache.py` |
| Dashboard | Streamlit + Altair | One local process with no Docker or Qdrant. | `dashboard/` |
| Tests | pytest | 65 tests. Model and API calls are replaced by fakes, so they run offline. | `tests/` |

**Listed but not used.** `requirements.txt` also installs `openai`, `unstructured[pdf]`, `pytesseract`, `Pillow`, `pyarrow`, `ragas`, `arize-phoenix`, and `structlog`. None of them is imported by any file under `src/`, `dashboard/`, `tests/`, `data/`, or `mvp_rag.py`. `matplotlib` is used only by the notebook. The `Dockerfile` installs `tesseract-ocr` and `poppler-utils`, which the code also does not use. Phoenix tracing and OCR are planned but not built; the rest can be pruned.

---

## Project status

| Area | Component | Status |
|---|---|---|
| Data | RRC manual download | ✅ 35 of 36 listed manuals download; `oda037k` returns HTTP 404 (checked 2026-10-08) |
| | Structured data, imaged permits | ⬜ downloadable with `--include-data` / `--include-all`, not parsed |
| | Statewide Rules (16 TAC Chapter 3) | 🟡 a dedicated chunker exists (`src/ingest/rules.py`) but nothing calls it, it has no tests, and the rules PDF is not in the downloader |
| Ingestion | Parser, section-aware chunker, `bge-large` embedder, Qdrant store | ✅ |
| Serving | Router, dense or hybrid retriever, cross-encoder reranker, generator, guardrail | ✅ unit-tested with fakes |
| | Grade/rewrite loop, answer cache, per-stage trace, JSONL query log | ✅ opt-in, not yet measured on the eval set |
| | Calibrator hook | 🟡 implemented, nothing fitted |
| API | FastAPI service (`/health`, `/ask`, `/ask/stream`), Docker Compose with Qdrant and Redis | ✅ not load-tested |
| Dashboard | Hybrid search, agent loop, streaming, cache, tracing, eval tab | ✅ |
| Evaluation | Harness, Opus judge, metrics library (Recall@k, MRR, κ, ECE, OOD F1), calibration CLI | ✅ |
| | Eval set | 🟡 12 of 200 planned questions |
| | Recorded end-to-end results | 🟡 one run without reranker or judge (see below) |
| Production | CI eval gate, Phoenix tracing | ⬜ |

✅ implemented · 🟡 partial · ⬜ planned. Every known difference between the design and the code is listed in [ARCHITECTURE.md §12](ARCHITECTURE.md#12-known-gaps-between-design-and-code).

---

## Results and conclusions

### What has been measured, and can be checked

| Fact | Value | Check with |
|---|---|---|
| Unit tests | 65 passed, 0 failed | `python -m pytest -q` |
| Manuals in the downloader | 36 | `MANUALS` in `data/download_data.py` |
| Manuals downloaded | 35 PDFs, 13 MB | `ls data/raw/manual` |
| Committed Qdrant store | 2,180 points, 1024-d cosine, 36 documents | [Verify it yourself](#verify-it-yourself) |
| Dashboard chunk cache | 4,820 chunks from 35 manuals | `data/processed/dashboard/*.json` |
| Eval set | 12 questions: 3 single-doc factual, 2 multi-doc synthesis, 2 procedural, 2 definitional, 3 out-of-scope | `eval_data/eval_set_sample.yaml` |
| Committed harness output | 12 of 12 cases errored with `401 invalid x-api-key`; no metrics | `eval_data/results_latest.jsonl` |
| First clean run (2026-10-08) | See the table below | `eval_data/results_verify_norerank.jsonl` |

Two details matter when reading these numbers:

- The committed Qdrant store contains 188 chunks from `oda037k_oil_gas_docket`, a manual that no longer downloads. Re-ingesting from today's download produces a different store.
- The two pipelines chunk differently (section-aware in Qdrant, fixed 800-character windows in the dashboard), so their chunk counts are not comparable.

### First end-to-end run (2026-10-08)

`python -m src.eval.harness --qdrant-path qdrant_local --no-judge --no-rerank --output eval_data/results_verify_norerank.jsonl`: vector search, no reranker, no agent loop, no calibrator, no judge, `claude-sonnet-4-6`, 12 questions.

| Metric | Result | Target |
|---|---|---|
| Errors | 0 / 12 | – |
| Recall@5 (manual level) | 0.78 (7 of 9 in-scope) | ≥ 0.85 ❌ |
| Recall@20 (manual level) | 0.89 (8 of 9) | ≥ 0.95 ❌ |
| Out-of-scope refused | 3 / 3 | ≥ 0.95 ✅ |
| In-scope escalated | 5 / 9 (0.56) | ≤ 0.10 ❌ |
| OOD F1 (escalations count as refusals) | 0.55 | ≥ 0.85 ❌ |
| Average latency | 7.0 s | – |

Of the 5 escalated in-scope questions, 3 had a correct manual in the top 5. On those the model rated its own confidence low (2, 22, 30 out of 100), so the escalations were not caused by retrieval misses alone. For example, the W-10 filing-deadline question (`sdf-001`) retrieved the W-10 manual and still got confidence 2. That fits the hypothesis that data-layout manuals don't state rules, but without the judge or a read of the retrieved passages it is not proven. With 9 in-scope questions, one question moves recall by 0.11, so these numbers show direction only.

### Reported but not verifiable

Earlier versions of this README reported a dashboard eval run from 2026-10-08 (hybrid search, router and agent loop, `claude-sonnet-4-6`, 12 questions): Hit@5 0.67, MRR 0.58, OOD refusal 3/3, in-scope refusal 5/9. **No output from that run is saved in the repository**, so it cannot be checked. Treat it as an anecdote until it is reproduced with saved output.

### Conclusions

What the evidence above supports:

1. **The system runs end to end, but answer quality is unmeasured.** One saved run shows retrieval and refusal behaviour (above), but no run has been judged for answer correctness, and none has used the reranker.
2. **The project's central claim, calibrated confidence, is untested.** No calibrator has been fitted, so the guardrail currently acts on the model's raw self-rated confidence. The 0.85 / 0.70 thresholds are design choices, not values derived from data.
3. **The corpus probably limits answerable questions.** By their titles and descriptions in `data/download_data.py`, most manuals document data files (record layouts, field definitions). Rule-type questions such as filing deadlines need the Statewide Rules, which are not ingested. A chunker for them has been written but not connected.
4. **The eval set is too small to support conclusions.** With 12 questions, and ground truth at the manual level rather than the passage level, even a clean run would show direction only.

---

## Next steps

In order of priority. Each step produces something checkable.

1. **Record a judged baseline run.** The API key now works (the 2026-10-08 run had no errors). Run `python tasks.py eval` (with the Opus judge) and commit `eval_data/results_latest.jsonl`, replacing the errored file. Finish the `bge-reranker-large` download first, or add `--no-rerank` ([WORKFLOWS §5](docs/WORKFLOWS.md#5-run-the-evaluation)).
2. **Run the ablation.** `python tasks.py eval-ablation` compares vector, hybrid, and hybrid + agent loop on the same questions. Commit all three result files. Each run also records `recall_at_5` (before reranking) and `ranked_recall_at_5` (after), which shows whether the reranker earns its 2.24 GB; `--no-rerank` runs the pipeline without it.
3. **Add the Statewide Rules.** Add the rules PDF to `data/download_data.py` (the chunker expects a filename starting with `statewide_rules`), call `chunk_rules()` from `src/ingest/pipeline.py` for that file, add tests for `src/ingest/rules.py`, and re-ingest.
4. **Rebuild the Qdrant store** from the current download so it matches what the scripts produce (removes the docket-manual chunks).
5. **Grow the eval set** toward 200 questions with train/dev/holdout splits, writing ground truth from the corpus and recording it at passage level.
6. **Fit and validate the calibrator** on the train split, inspect it on dev (target ECE ≤ 0.08), and then set the thresholds from the cost model in [EVALUATION.md](EVALUATION.md).
7. **Collect human labels** for a subset to validate the Opus judge (target κ ≥ 0.6).
8. **Automate the eval gate in CI**, and prune unused dependencies from `requirements.txt` and the `Dockerfile`.

---

## Verify it yourself

Every figure in this README can be reproduced from the repository:

```bash
python -m pytest -q                                  # 65 passed
ls data/raw/manual | wc -l                           # 35 (after python data/download_data.py)
python -c "import sys; sys.path.insert(0,'data'); import download_data as d; print(len(d.MANUALS))"   # 36

# Committed eval output: count errored cases
python -c "import json; r=[json.loads(l) for l in open('eval_data/results_latest.jsonl')]; print(sum(bool(x['error']) for x in r), 'errors of', len(r))"

# Committed Qdrant store: points, dimension, documents
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(path='qdrant_local'); i = c.get_collection('troa_chunks')
names, off = set(), None
while True:
    pts, off = c.scroll('troa_chunks', with_payload=['doc_name'], limit=1000, offset=off)
    names |= {p.payload['doc_name'] for p in pts}
    if off is None: break
print(i.points_count, i.config.params.vectors.size, len(names))"   # 2180 1024 36
```

Model names are in the `model:` line of each file in `src/serve/prompts/` and in `JUDGE_MODEL` in `src/eval/harness.py`. Guardrail thresholds are in `src/serve/guardrail.py`.

---

## Quick start

Full procedures, with verification steps, are in [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

### 1. Install and configure

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt           # or a lighter profile, see WORKFLOWS §1
cp .env.example .env                      # set ANTHROPIC_API_KEY (workspace-scoped key with credits)
python data/download_data.py              # 35 RRC PDF manuals, about 13 MB
```

### 2. Pick an entry point

| Goal | Command | Needs |
|---|---|---|
| Smoke test | `python mvp_rag.py "What does the Drilling Permit Master dataset contain?"` | Corpus, key |
| Interactive dashboard | `streamlit run dashboard/app.py` | Corpus, key; the first run embeds every manual |
| Reference pipeline | `python -m src.ingest.pipeline --corpus data/raw/manual/ --qdrant-path qdrant_local`, then `Pipeline(qdrant_path="qdrant_local")` | Corpus, key, full install, about 3.6 GB of model downloads |
| Evaluation harness | `python -m src.eval.harness --eval-set eval_data/eval_set_sample.yaml --output eval_data/results_latest.jsonl --qdrant-path qdrant_local` | As above. Add `--search-mode hybrid`, `--agentic`, `--no-rerank`, `--calibration <json>` for ablations |
| HTTP API | `uvicorn src.api.app:app --port 8000` (or `docker compose up`) | Ingested Qdrant; settings in `.env`. Docs at `/docs` |
| Calibrator | `python -m src.eval.calibration train --results <judged results> --output calibration/v1.json` | Judged results; see [WORKFLOWS §6](docs/WORKFLOWS.md#6-fit-and-apply-the-calibrator) |
| Tests | `python -m pytest -q` | Full install |

**Shortcuts:** `python tasks.py` lists short names for these commands, for example `python tasks.py test`, `python tasks.py eval`, `python tasks.py eval-ablation`, `python tasks.py api`. Extra arguments are passed through (`python tasks.py eval --no-judge`). It needs only Python, so it works on Windows without `make`.

---

## Repository layout

```
troa/
├── README.md                  This file
├── ARCHITECTURE.md            System design, diagrams, status, design/code gaps
├── EVALUATION.md              Metric definitions, targets, calibration and threshold methodology
├── docs/
│   ├── WORKFLOWS.md           Step-by-step procedures and troubleshooting
│   └── DASHBOARD.md           Dashboard guide and internals
├── .env.example               Template for .env (API key, serving options)
├── requirements.txt
├── tasks.py                   Task runner: python tasks.py <task>
├── Dockerfile, compose.yml    API image; Qdrant + Redis + API stack
├── mvp_rag.py                 Single-file RAG smoke test (3 manuals, in memory)
├── data/
│   └── download_data.py       Curated RRC dataset downloader → data/raw/ (git-ignored)
├── src/
│   ├── ingest/                parse · chunk · embed · store · pipeline · rules (Statewide Rules chunker, not wired in)
│   ├── serve/                 router · retrieve · hybrid · rerank · agent · generate · guardrail · scope · cache · telemetry · pipeline
│   │   └── prompts/           router_v1 · generate_v1 · grader_v1 · rewrite_v1 · caveats (YAML)
│   ├── eval/                  harness · metrics · calibration
│   ├── api/                   app.py (FastAPI: /health, /ask, /ask/stream)
│   └── config.py              Settings from .env; builds the Pipeline for the API
├── dashboard/
│   ├── app.py                 Streamlit UI
│   └── engine.py              Hybrid search + agentic pipeline (no Streamlit dependency)
├── eval_data/
│   ├── eval_set_sample.yaml   12 questions across 5 categories with ground truth
│   └── results_latest.jsonl   Last harness output (all 12 cases errored; no metrics)
├── qdrant_local/              Local-file Qdrant store (troa_chunks: 2,180 points, 1024-d, 36 documents)
├── notebooks/
│   └── 01_calibration_analysis.ipynb   Reliability diagrams and threshold policy (needs judged results)
└── tests/                     test_ingest.py (29) · test_metrics.py (13) · test_serve.py (23)
```

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | `.env` or environment | – (required) |
| Router model / prompt | `src/serve/prompts/router_v1.yaml` | `claude-haiku-4-5-20251001` |
| Answer model / prompt | `src/serve/prompts/generate_v1.yaml` | `claude-sonnet-4-6` |
| Grader and rewriter | `grader_v1.yaml`, `rewrite_v1.yaml` | `claude-haiku-4-5-20251001` |
| API serving options | `.env`: `TROA_SEARCH_MODE`, `TROA_AGENTIC`, `TROA_RERANK`, `TROA_CALIBRATION`, `TROA_CACHE`, `REDIS_URL`, `TROA_QUERY_LOG` | vector, off, on, none, on (memory), none, off |
| Judge model | `JUDGE_MODEL` in `src/eval/harness.py` | `claude-opus-4-7` |
| Guardrail thresholds | `THRESHOLD_*` and `OOD_CUTOFF` in `src/serve/guardrail.py` (shared by pipeline and dashboard) | 0.85 / 0.70; OOD 0.85 |
| Caveat and refusal text | `src/serve/prompts/caveats.yaml` | – |
| Embedding model | `Pipeline(embed_model=…)` · `EMBED_MODEL` in `mvp_rag.py` | `bge-large-en-v1.5` · `bge-small-en-v1.5` |
| Chunking | `ChunkingConfig` in `src/ingest/chunk.py` · `CHUNK_SIZE` / `CHUNK_OVERLAP` in `mvp_rag.py` | 512 / 64 tokens · 800 / 150 chars |
| Retrieval depth | `Pipeline(retrieve_top_k=20, rerank_top_k=5)` · dashboard sidebar | 20 → 5 · 5 |

`Pipeline` itself leaves the cache and every other optional feature off; the API turns the cache on unless `TROA_CACHE=false`. Prompts are versioned by filename. To change one, add a new version rather than editing in place ([WORKFLOWS §7](docs/WORKFLOWS.md#7-change-a-prompt-or-model)).

---

## Documentation

| Document | Read it for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview, component status, ingestion, serving, dashboard, guardrail, eval flow, data model, design decisions, known gaps |
| [EVALUATION.md](EVALUATION.md) | Eval set design, metric definitions and targets, judge validation, calibration method, cost-based threshold policy, CI gates |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | Setup, download, ingestion, querying, evaluation, calibration, prompt changes, adding manuals and questions, troubleshooting |
| [docs/DASHBOARD.md](docs/DASHBOARD.md) | Dashboard tabs, settings, request lifecycle, caches, extension points |

---

## Limitations

- This is a portfolio project on public regulatory data. It has not been reviewed by working compliance officers. **Do not use it for compliance decisions.**
- The eval set was written by the author, not by domain experts, and has 12 questions.
- Retrieval ground truth is at the manual level, not the passage level, so Hit@k and MRR are lenient.
- Confidence is uncalibrated until a calibrator is fitted. The planned calibration is aggregate, not per question category.
- The LLM judge has not been validated: the κ computation exists, but no human labels have been collected.
- The Statewide Rules are not in the searchable corpus.
- Deliberately out of scope for v1: fine-tuning the embedder or reranker, OCR or vision for the imaged W-1 permits (which contain drawings), and monitoring beyond JSONL logs.

## License

The intended licence for the code is Apache 2.0, but no `LICENSE` file is committed yet. The RRC manuals are downloaded from rrc.texas.gov. The PDFs are not committed (`data/raw/` is git-ignored), but the committed `qdrant_local/` store contains text extracted from them. Check RRC's terms before reusing them.
