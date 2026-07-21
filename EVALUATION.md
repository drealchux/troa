# EVALUATION.md

## Eval principles

1. **Every metric ties to a deployment decision.** If we can't articulate what we'd do differently at metric value X versus Y, we don't track it. Vanity metrics waste signal.

2. **Probabilistic outputs need probabilistic evaluation.** Faithfulness and relevance are not binary. We use rubrics with discrete levels, document them, and stress-test the rubric for inter-judge agreement.

3. **LLM-as-judge gets validated against human labels.** Always. A 50-question subset is human-labeled. We compute Cohen's kappa between the LLM judge and human labels. If kappa drops below 0.6, the judge prompt or model is unreliable and we re-spec.

4. **Calibration is a deployment requirement, not a research nicety.** The system has to know when not to answer. That knowledge is calibrated and validated, not assumed.

5. **Evaluation runs in CI on every PR.** A regression on any tracked metric blocks the deploy until justified.

## Eval set construction

200 questions, hand-built by me with reference to the RRC manuals. Each question has:

- The question text
- Ground truth answer (free text, 1-3 sentences)
- Ground truth supporting chunks (which manual, which section)
- Difficulty (1-3)
- Category (one of 5 below)

Splits: 100 train (for calibration fitting), 50 dev (for iteration), 50 holdout (used once at v1.0 release).

### Categories and distribution

| Category | Count | Example |
|----------|-------|---------|
| Single-doc factual | 50 | "What's the deadline for filing form W-10?" |
| Multi-doc synthesis | 50 | "If I'm drilling a horizontal well in the Spraberry, what forms and rules apply?" |
| Procedural | 40 | "Walk me through the steps to file an exception to Rule 37" |
| Definitional | 40 | "What does 'designated operator' mean under P-5?" |
| Out-of-scope (refusal) | 20 | "What's the weather in Stavanger?" |

The OOD category is critical and often skipped in tutorial-grade evals. A system that confidently answers OOD questions is unsafe in a regulated setting.

## Metrics

Five families. Each has a target threshold below which v1.0 cannot ship.

### 1. Retrieval quality

We measure retrieval against the ground-truth supporting chunks identified during eval set construction.

- **Recall@5:** did we retrieve at least one ground-truth chunk in the top 5? Target: >= 0.85
- **Recall@20:** same at top 20 (this is the pool the reranker sees). Target: >= 0.95
- **MRR (mean reciprocal rank):** average of 1/rank of the first relevant chunk. Target: >= 0.55

If Recall@5 < 0.85 we have a retrieval problem before we even get to generation. No amount of LLM tuning fixes that. We diagnose with embedding quality (try a different model), chunking strategy (try smaller/larger chunks, different overlap), or query rewriting (add a query rewriter to the router).

### 2. Answer quality (LLM-as-judge, validated)

For each generated answer, an Opus-class judge model scores on three dimensions using rubrics:

- **Faithfulness (0/1/2):** does every claim in the answer follow from the retrieved chunks? 0 = hallucinated claims present, 1 = mostly grounded with minor unsupported additions, 2 = fully grounded.
- **Relevance (0/1/2):** does the answer address the question asked? 0 = off-topic, 1 = partial, 2 = directly addresses.
- **Citation accuracy (0/1/2):** do the cited chunks actually support the claims they're cited for? 0 = wrong cites, 1 = some right, 2 = all right.

Targets: mean faithfulness >= 1.7, mean relevance >= 1.7, mean citation accuracy >= 1.6.

### 3. Judge validation

50-question subset is double-scored by me (human) using the same rubrics. We compute Cohen's kappa per dimension. Target: kappa >= 0.6 (substantial agreement). If kappa drops below 0.6 in any future change, the judge prompt is re-tuned or the model swapped before any other metric is trusted.

### 4. Calibration

This is the deepest section. The system elicits a confidence score from the LLM (a 0-100 rating in the same generation call). Raw LLM confidence scores are systematically miscalibrated (overconfident on factual tasks, underconfident on ambiguous ones). We fix this in two steps:

**Step 1: collect ground truth confidence labels.** On the 100 train-split questions, we run the pipeline, capture the raw LLM confidence, and label each answer as correct (1) or incorrect (0) using the faithfulness + relevance scores: correct iff faithfulness == 2 AND relevance >= 1.

**Step 2: fit a calibrator.** Platt scaling: fit a logistic regression that maps raw LLM confidence -> calibrated probability of correct. Validate on the 50 dev questions.

**Validation:**
- Plot reliability diagram (10 bins, x = predicted, y = empirical). A perfectly calibrated model lies on the diagonal.
- Compute Expected Calibration Error (ECE):

```
ECE = sum over bins b of (|bin_b| / N) * |acc(bin_b) - conf(bin_b)|
```

Target: ECE <= 0.08 after calibration (raw ECE typically 0.15-0.25 before calibration).

We also report Brier score and log loss for completeness.

### 5. Refusal / OOD performance

For OOD questions (the 20 out-of-scope), the system should refuse. For in-scope, it should not refuse.

- **OOD refusal rate:** fraction of OOD questions correctly refused. Target: >= 0.95.
- **In-scope refusal rate:** fraction of in-scope questions wrongly refused. Target: <= 0.10.
- **OOD F1:** treating "refuse" as the positive class for OOD detection. Target: >= 0.85.

We tune the refusal threshold (calibrated confidence below which we refuse) on the dev set. The default starting point is 0.45.

## Threshold policy: from calibrated confidence to action

Once calibration is in place, the system maps confidence to one of four actions:

| Calibrated confidence | Action | Rationale |
|------------------------|--------|-----------|
| >= 0.85 | Autonomous answer | High enough that the cost of a wrong answer is dominated by the value of fast answers |
| 0.65 - 0.85 | Answer with uncertainty caveat | Useful but flag for the user to double-check |
| 0.45 - 0.65 | Answer + "verify with compliance officer" | Soft escalation, treat as draft |
| < 0.45 | Refuse + explanation | Better to refuse than mislead in a regulated setting |

These thresholds are decisions, not defaults. They're set against a notional cost model:
- Wrong answer accepted by user: high cost (regulatory fine, ops delay)
- Right answer delayed by escalation: low-medium cost (compliance officer time)
- Refusal of in-scope question: medium cost (user goes elsewhere)

The thresholds above approximately minimize expected cost under a cost ratio of ~10x for wrong-answer vs delayed-answer. Different operating contexts (more risk-tolerant teams, lower-cost domains) would shift the thresholds; the framework stays the same.

## CI gates

GitHub Actions runs the full eval suite on every PR that touches `src/`. The PR blocks if:

- Recall@5 drops by more than 3 percentage points from the main branch baseline
- Faithfulness or relevance drops by more than 0.15 from baseline
- ECE increases by more than 0.03 from baseline
- OOD F1 drops by more than 0.05 from baseline

These thresholds are intentionally a little loose. We want to allow legitimate trade-offs (you might accept a slight recall hit for a big faithfulness improvement) but force a discussion when they happen.

## Failure analysis

After every full eval run, we bucket every failure (score below threshold on any dimension) into one of:

- Retrieval miss (right chunk not retrieved)
- Reranker error (right chunk retrieved but pushed out by reranker)
- Generation hallucination (chunk retrieved, model invented claim)
- Citation error (chunk used but cited wrong)
- Calibration error (model was confidently wrong, or unconfidently right)
- Query parsing error (router misclassified the intent)
- Ambiguity (eval question is genuinely ambiguous, eval label may be wrong)

The distribution of failure modes tells us where to invest next. After v1.0 we expect retrieval misses to dominate (and that's where the next iteration goes).

## What this eval does not measure (yet)

Things we don't measure in v1 but flag explicitly for the next iteration:

- **Latency under load.** A real production system needs p50/p95/p99 latency targets. Out of scope for v1 portfolio version.
- **Cost per query.** Token budget per query matters at scale. We log it but don't budget it.
- **Adversarial robustness.** Prompt injection through retrieved chunks, jailbreak attempts. Out of scope.
- **Distribution shift over time.** New manuals, new rules, query language drift. Out of scope for v1, but the design supports it (versioned vector store, eval set additions).

Documenting what we don't measure is more useful than claiming we measure everything.
