# ROADMAP.md

12-week plan to build TROA from empty repo to v1.0 release with a blog writeup. Calibrated to roughly 10 hours per week of focused work.

The order matters. Each milestone produces something demoable. If you fall behind, ship a smaller version of the next milestone rather than skipping ahead.

## Week 1: Data and chunking

- Set up the repo, dev environment, Qdrant in Docker.
- Run `data/download_data.py`. Inspect what came down.
- Hand-read 5 representative PDF manuals. Note the document structure (sections, tables, references between manuals).
- Build the PDF parser. Validate output against the 5 hand-read manuals.
- First chunking pass: fixed-window 512/64. Get something in the vector store.

**Deliverable:** vector store with ~3000 chunks, sample queries return roughly relevant results.

## Week 2: Baseline RAG and eval seed

- Wire up the generator: simple retrieve-then-generate with Claude Sonnet. No reranker, no router yet.
- Write 30 eval questions by hand across the 5 categories. This is the seed eval set.
- Build the first version of the eval harness: run pipeline, score with a basic LLM judge.
- Run the baseline. Look at what fails.

**Deliverable:** end-to-end working RAG, baseline metrics on 30 questions documented.

## Week 3: Eval set construction

- Expand the eval set from 30 to 200 questions, with ground-truth supporting chunks identified for each.
- Build the eval set as YAML in `eval_data/` with schema.
- Split into train (100) / dev (50) / holdout (50). Document the splits.
- Hand-label 50 of them for judge validation.

**Deliverable:** 200-question eval set committed, splits documented.

## Week 4: Semantic chunking and retrieval tuning

- Build the semantic chunker. Test against fixed-window on the eval set.
- Try alternative embedding models (e5-large-v2) for comparison. Document the choice.
- Add metadata filtering to Qdrant.
- Recall@5, Recall@20, MRR baselines locked in.

**Deliverable:** retrieval metrics meet targets (Recall@5 >= 0.85, Recall@20 >= 0.95, MRR >= 0.55).

## Week 5: Reranker and answer quality

- Add the cross-encoder reranker.
- Tune the generator prompt to get clean citations and a structured confidence output.
- Run the LLM-as-judge pipeline. Compute faithfulness, relevance, citation accuracy.
- Validate judge against human labels. Iterate the judge prompt until kappa >= 0.6.

**Deliverable:** answer quality metrics meet targets, judge validation kappa >= 0.6.

## Week 6: Router and OOD

- Build the router. Test intent classification accuracy on the eval set.
- Add OOD detection. Measure refusal rates on in-scope vs OOD.
- Wire the router's document scope hints into the retriever as metadata filters.

**Deliverable:** OOD F1 >= 0.85, in-scope refusal rate <= 0.10.

## Week 7: Calibration

This is the centerpiece week. Don't rush it.

- Capture raw LLM confidence scores from the generator on the train split.
- Label each answer as correct using faithfulness >= 2 AND relevance >= 1.
- Fit Platt scaling. Validate on dev split.
- Plot reliability diagrams (before and after calibration).
- Compute ECE, Brier, log loss.
- Write the calibration notebook (`notebooks/01_calibration_analysis.ipynb`). This becomes the centerpiece of the blog post later.

**Deliverable:** calibrated confidence, ECE <= 0.08, reliability diagram looks honest.

## Week 8: Threshold policy and guardrail

- Define the cost model. What does a wrong answer cost? A delayed answer? A refusal?
- Derive threshold values from the cost model.
- Implement the guardrail as a thin layer that maps calibrated confidence to action.
- Re-run the full eval with the guardrail in place. Document what changed.

**Deliverable:** end-to-end system with threshold-driven actions, eval metrics on the guarded outputs.

## Week 9: CI gates and prompt versioning

- GitHub Actions workflow that runs the eval suite on every PR.
- Threshold-based gates as specified in EVALUATION.md.
- Pull all prompts into `src/serve/prompts/*.yaml`, version-controlled.
- Add a baseline metrics file (`eval_results/main_baseline.json`) that PRs are compared to.

**Deliverable:** CI gate is live, a deliberate small regression in a test PR is blocked.

## Week 10: Observability and the failure analysis loop

- Structured JSONL logging for every query.
- Phoenix integration for traces.
- Build the failure analysis notebook: bucket every failure into the categories in EVALUATION.md.
- Document the top 3 failure modes and propose specific fixes for v2.

**Deliverable:** observable system, failure analysis report.

## Week 11: Polish and packaging

- README cleanup, install instructions tested from scratch on a fresh machine.
- API wrapper (FastAPI) over the pipeline.
- Docker Compose for local dev.
- Smoke tests in `tests/`.
- License files, contribution notes.

**Deliverable:** repo is ready for public release.

## Week 12: Blog post and release

- Write a blog post: "Building a calibrated RAG system for Texas oil and gas regulations." Aim for 2000-3000 words.
- Sections: the problem, the eval methodology, the calibration deep dive (with reliability diagrams), the threshold policy, what failed and why, what's next.
- Release the repo publicly. Post the blog. Share in relevant communities (HN, r/MachineLearning, LinkedIn).

**Deliverable:** public repo, public blog post, evidence of senior-level rigor for hiring conversations.

## What to do if you fall behind

Cut from the bottom, not the top. Skip the FastAPI wrapper before you skip the calibration work. The calibration work and the eval methodology are the things a senior reviewer cares about. Everything else is supporting infrastructure.

If you have only 8 weeks instead of 12:

- Keep weeks 1-8 as written.
- Combine weeks 9-10 into one week of "CI + observability."
- Skip the polish week.
- Compress the blog post into 1500 words.

The minimum viable version is: working pipeline, 100-question eval set, calibration with reliability diagram, threshold policy, blog post. Everything else is gravy.

## What to do if you have extra time

Pick one:

1. Per-category calibration (requires expanding the eval set to ~400 questions).
2. Adversarial robustness testing (prompt injection through chunks).
3. A thin Next.js UI for live demo.
4. Hybrid search (BM25 + dense) with proper ablation study.

Option 1 is the highest-leverage for the hiring conversation.
