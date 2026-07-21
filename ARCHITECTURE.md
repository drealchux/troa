# ARCHITECTURE.md

System design for TROA. Three lanes: ingestion, serving, evaluation.

## Ingestion (offline)

### Sources

- **PDF manuals** (~30 documents): primary corpus. Each manual describes one RRC form or dataset, including field-level data layouts and procedural rules.
- **Field rules ASCII**: structured spacing and production rules per field. Parsed into a side-by-side structured + text representation.
- **W-1 imaged permits** (sample of ~500): real drilling permit examples. Useful for procedural questions.
- **Inspection and violations TXT**: weekly updates of inspection results. Provides operational context.
- **UIC database extract**: injection well data, schema available from the UIC PDF manual.

### Parsing strategy

Different document types get different parsers:

- **PDF manuals**: `pypdf` for text extraction, then `unstructured` for layout-aware parsing where tables matter. Manual review of 5 representative manuals to validate parser output.
- **Structured ASCII/CSV files**: parsed per the dataset's PDF manual (each manual contains the field-level data layout). Converted to Parquet for downstream joins.
- **Imaged W-1 PDFs**: OCR with `tesseract` (the imaged permits are scanned forms). Layout-aware parsing of form fields where possible; otherwise full-text fallback.

### Chunking

Semantic chunking, not fixed-window. For the PDF manuals, we chunk by document structure:

- Section headers detected via font-size and styling heuristics
- Each leaf section becomes one chunk, capped at 512 tokens with 64-token overlap into the next chunk
- Chunks below 100 tokens are merged with the next chunk
- Metadata attached to every chunk: document name, section path, page number, dataset reference

Why not fixed-window: regulatory text has strong section semantics. Chunking across a section boundary destroys retrieval signal. We tested fixed-window (512/64) against semantic chunking on a 50-question pilot and semantic improved Recall@5 by 11 points.

### Embedding

`bge-large-en-v1.5` (BAAI, MIT license). 1024-dimensional, strong on retrieval benchmarks, runs on a single GPU or modestly on CPU for ingestion.

Alternative considered: `e5-large-v2`. Comparable quality. Chose BGE for its better behavior on domain-specific queries in pilot tests.

### Vector store

Qdrant. Local Docker for dev, Qdrant Cloud for serving.

- Vector index on the chunk embedding
- Payload indexes on metadata: document name, dataset reference, section path
- Filtered search supported (e.g. retrieve only from W-1 manual when the router says it's a W-1 question)

Why Qdrant over alternatives (Pinecone, Weaviate, pgvector): metadata filtering is fast and ergonomic, local dev is trivial, Apache 2.0 license, can self-host.

## Serving (online)

### Router

A small LLM call (Claude Haiku for speed and cost) classifies the query into:

- Intent: lookup / procedural / definitional / synthesis / OOD
- Document scope: optional filter for which manuals to retrieve from
- Confidence: how sure the router is

OOD detection is part of the router. If the router says OOD with confidence >= 0.7, we short-circuit to a refusal response without ever retrieving.

### Retriever

Dense retrieval against Qdrant. Top-k = 20.

Filters applied:
- If router specified a document scope, filter by document_name in metadata
- If router specified a dataset reference (e.g. "this is about W-10"), filter to that dataset's manual

We tested adding BM25 as a hybrid signal. Marginal improvement on multi-doc synthesis (+2 points Recall@20), no improvement on single-doc factual. Reserved for v2.

### Reranker

`bge-reranker-large`, a cross-encoder. Takes the 20 retrieved chunks plus the original query, scores them, returns top 5.

The reranker is also the critic stage. If the top reranker score is below a threshold (calibrated separately), we treat retrieval as low-confidence and propagate that into the calibrator. This catches cases where the retriever returned 20 mediocre chunks rather than 5 great ones.

### Generator

Claude Sonnet. Prompt includes:

- The user query
- The top-5 reranked chunks, each with a citation marker [1], [2], ..., [5]
- Instructions: answer concisely, cite every claim, return a confidence score 0-100 at the end of the answer in a structured tag

Why Sonnet: strong instruction following, good citation behavior, much cheaper than Opus, and we use Opus for the eval judge separately so we want independence between production and judge models.

System prompt is versioned in `src/serve/prompts/generate_v3.yaml`. Every change to the prompt is a PR that runs the eval gate.

### Calibrator

Platt scaling fit on (raw_confidence, correct) pairs from the train split. Implemented as a scikit-learn LogisticRegression on a single feature.

We also tested isotonic regression. Performance was similar (ECE within 0.01) but Platt is more interpretable and less prone to overfitting on small calibration sets. Picked Platt.

The calibrator's coefficients are checked into git (a ~50-byte JSON file). Retrained whenever the eval set or prompt changes.

### Guardrail

The threshold-to-action mapping from `EVALUATION.md`. A simple lookup, but it's the layer that turns a calibrated probability into a deployable system behavior.

Implementation note: the guardrail also adds a standardized caveat banner to answers below 0.85 confidence ("This answer is provided with moderate confidence; verify with the cited source"). The caveat text is templated and lives in `src/serve/prompts/caveats.yaml`.

## Evaluation (CI)

Triggered on every PR touching `src/` or `eval_data/`. Implementation lives in `src/eval/`.

Stages:

1. **Run pipeline** on the full 200-question eval set with the PR's branch code.
2. **Score retrieval** (Recall@k, MRR) using ground-truth chunks.
3. **Score answers** with the Opus judge using rubrics.
4. **Validate judge** by computing kappa against the 50-question human-labeled subset.
5. **Train calibrator** on train split, evaluate calibration on dev split, compute ECE.
6. **Compare to main-branch baseline** on every metric.
7. **Decide:** pass, warn, or block.

Total wall time on a fresh checkout: ~6 minutes (200 queries, parallelized to 5 concurrent).

## Observability

Two layers:

- **Structured logs** for every query: query, retrieval hits, reranker scores, raw confidence, calibrated confidence, action, latency. JSONL to disk, can ship to Loki or similar in production.
- **Phoenix (Arize)** for traces. Each query is a trace with retrieval, reranking, and generation as spans. Useful for inspecting individual failures during eval analysis.

For v1 we don't have a metrics dashboard. Phoenix UI is enough.

## Deployment

API only in v1. FastAPI service exposing `/answer` and `/health`.

Docker Compose for local: Qdrant + the API. A `docker-compose.yml` is in the repo. Production would deploy the API on Cloud Run / Fly / your platform of choice, with Qdrant Cloud as the vector store. Out of scope for v1.

## Design decisions worth flagging

A few choices a reviewer might question; here are the reasons:

- **Single LLM (Claude Sonnet) for generation rather than a model gateway.** Simpler. v2 could add model switching for cost optimization.
- **Cross-encoder reranker rather than a second LLM call.** Faster and cheaper. The downside is less flexibility; if we needed reasoning during reranking we'd revisit.
- **Calibration on aggregate, not per-category.** v1 simplicity. The 200-question eval set is too small for reliable per-category calibration. v2 with a larger eval set would calibrate per intent category.
- **No fine-tuning.** Domain-specific embeddings would help, but the cost-vs-improvement ratio is poor for a portfolio v1. Reserved for v2 with a proper feedback loop.
