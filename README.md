# TROA: Texas Oil and Gas Regulatory Operations Assistant

A grounded RAG assistant for operations and compliance staff working under Texas Railroad Commission (RRC) rules. Built as a portfolio project demonstrating senior-scientist hygiene applied to an industrial domain: real regulatory data, custom evaluation methodology, calibrated confidence translated into operational thresholds, and light agentic patterns.

## Why this project

Operators and compliance officers spend hours navigating regulatory text to answer questions like:

- "What forms do I need for a horizontal well re-entry in the Spraberry?"
- "What's the deadline for filing the W-10 if I missed the original window?"
- "Does this proposed spacing need a Rule 37 exception?"

These are real, repeated workflows where wrong answers carry regulatory cost. That cost is what makes calibration and operational thresholds matter, not a nice-to-have but the actual point.

## What the system does

Takes a natural-language question about Texas O&G regulations and returns a grounded answer with:

- Citations back to source documents (PDF manual, page, paragraph)
- A calibrated confidence score
- A guardrail decision: autonomous answer, answer with caveat, escalate, or refuse

When the system isn't confident enough to answer reliably, it refuses rather than hallucinates. The confidence-to-action mapping is set by an explicit threshold policy, not a default.

## Data

All from the Texas Railroad Commission public datasets page (rrc.texas.gov). Free, no auth, real industrial regulatory content.

| Source | Type | Use |
|--------|------|-----|
| ~30 PDF user manuals (form schemas, data layouts) | Unstructured text | Primary RAG corpus |
| Oil & Gas Field Rules | Structured ASCII | Rule lookups, structured queries |
| W-1 drilling permit imaged PDFs | Unstructured | Real-world examples |
| Inspections and violations data | Structured TXT | Operational context |
| UIC (injection control) data | Structured | UIC-specific queries |
| Statewide Field Data | Structured | Field-level context |

Run `python data/download_data.py` to pull the corpus.

## Architecture

Three lanes:

1. **Ingestion (offline, weekly):** Parse PDFs, semantic chunking, embed with `bge-large-en-v1.5`, store in Qdrant with metadata.
2. **Serving (online, per query):** Router classifies intent and screens for out-of-distribution, retriever pulls top-20, cross-encoder reranks, Claude Sonnet generates with citations, Platt-scaled calibrator outputs a confidence score, guardrail gates the response.
3. **Evaluation (CI on every PR):** Run pipeline against held-out eval set, score retrieval and answer quality metrics, train and validate calibration on held-out data, block deploys on regression.

See `ARCHITECTURE.md` for the full system design, and `EVALUATION.md` for the metric and calibration methodology.

## What makes this senior-level

These are the things a reviewer at an industrial-AI scientist role looks for, in priority order:

1. **Custom evaluation methodology, not just RAGAS off the shelf.** 200-question eval set across 5 categories (single-doc factual, multi-doc synthesis, procedural, definitional, out-of-scope). Metrics tied to deployment go/no-go decisions. LLM-as-judge validated against human labels on a holdout subset.

2. **Calibrated confidence translated into operational thresholds.** Confidence elicited from the LLM, calibrated with Platt scaling on a held-out set, validated with reliability diagrams and Expected Calibration Error (ECE). Threshold policy ties calibrated probability to concrete actions (autonomous, caveat, escalate, refuse), with thresholds chosen against an explicit cost model.

3. **Out-of-distribution behavior.** Refusal pattern for queries outside the regulatory domain. F1 on refusal as a first-class metric.

4. **Production hygiene.** Version-controlled prompts (YAML in git), CI runs the full eval on every PR, drift monitoring on query distribution and answer quality, observability with structured logs.

5. **Agentic patterns where they earn their keep.** Query router for intent classification and OOD screening. Reranker as a critic stage. Optional multi-hop planner for queries that need multiple retrievals. No agents for the sake of agents.

## Stack

- Python 3.11
- Embeddings: `bge-large-en-v1.5` (BAAI, MIT license)
- Reranker: `bge-reranker-large` (cross-encoder)
- Vector DB: Qdrant (local for dev, Qdrant Cloud for serving)
- LLM (production): Claude Sonnet via Anthropic API
- LLM (eval judge): Claude Opus
- Calibration: scikit-learn (Platt scaling, isotonic regression)
- Eval: custom harness + RAGAS for selected metrics
- CI: GitHub Actions
- Observability: structured logs + Phoenix (Arize) for traces

## Repository layout

```
troa-rag/
├── README.md                   This file
├── ARCHITECTURE.md             System design
├── EVALUATION.md               Eval methodology and metrics
├── ROADMAP.md                  Week-by-week plan
├── requirements.txt
├── .env.example
├── data/
│   └── download_data.py        Pulls the RRC corpus
├── src/
│   ├── ingest/                 Parsing, chunking, embedding
│   ├── serve/                  Online query pipeline
│   └── eval/                   Eval harness and calibration
├── eval_data/
│   └── eval_set_sample.yaml    Sample questions with ground truth
├── notebooks/                  Analysis notebooks
└── tests/
```

## Quick start

```bash
# Set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY

# Pull the corpus (takes ~10 minutes on a decent connection)
python data/download_data.py

# Ingest into the vector store (assumes Qdrant running on localhost:6333)
python -m src.ingest.pipeline --corpus data/processed/

# Run the eval harness
python -m src.eval.harness --eval-set eval_data/eval_set_v1.yaml

# Train calibration on the eval results
python -m src.eval.calibration --results eval_results/latest.jsonl
```

## What's intentionally missing from v1

- Fine-tuning the embedding or reranker on RRC-specific terms (would help but adds compute and a feedback loop; defer to v2)
- Production monitoring stack beyond logs and Phoenix (v2)
- Multi-modal: the imaged W-1 PDFs have plats (drawings) that aren't yet OCR'd or vision-modeled (v2)
- Web UI: API only in v1; a thin Next.js front end can come in v2

Each of these is a deliberate scope cut, not an oversight. The point of v1 is the science and engineering rigor, not the surface area.

## Limitations and honest claims

- This is a portfolio project on public regulatory data, not a deployed product. It hasn't been red-teamed by actual compliance officers.
- The eval set was constructed by me, not by domain experts. A real production version would need expert review of the ground truth.
- Calibration is based on aggregate eval performance, not per-query-category subgroups. A v2 would calibrate per intent category.
- LLM-as-judge has known limitations. I validate against human labels on a 50-question holdout, but the validation sample is small.

These limits are documented because they matter for any conversation about deploying something similar in a regulated industrial setting.

## License

Apache 2.0 for code. The Texas RRC datasets are public domain.
