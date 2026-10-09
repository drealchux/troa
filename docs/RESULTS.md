# Evaluation results

Every recorded end-to-end run of TROA, with the commands to reproduce each figure. Metric definitions and targets are in [EVALUATION.md](../EVALUATION.md); the summary is in the [README](../README.md#results).

## What has been measured, and can be checked

| Fact | Value | Check with |
|---|---|---|
| Unit tests | 82 passed, 0 failed | `python -m pytest -q` |
| PDFs in the downloader | 37 (36 manuals + Statewide Rules) | `MANUALS` in `data/download_data.py` |
| PDFs downloaded | 36, 16 MB | `ls data/raw/manual` |
| Bundled search index | 2,941 chunks (1,992 manual + 949 rules), 1024-d cosine, 36 documents: exactly the downloaded PDFs | [Verify it yourself](#verify-it-yourself) |
| Eval set | 12 questions: 3 single-doc factual, 2 multi-doc synthesis, 2 procedural, 2 definitional, 3 out-of-scope | `eval_data/eval_set_sample.yaml` |
| Committed harness output | 12 of 12 cases errored with `401 invalid x-api-key`; no metrics | `eval_data/results_latest.jsonl` |
| First clean run (2026-10-08) | See the table below | `eval_data/results_verify_norerank.jsonl` |
| Same run with the Statewide Rules indexed | See the table below | `eval_data/results_verify_norerank_rules.jsonl` |
| Judged run | See the table below | `eval_data/results_judged_norerank.jsonl` |

The three runs below used the index before 188 chunks from `oda037k_oil_gas_docket` (a manual that no longer downloads) were removed, so re-running them now gives slightly different retrieval.

## First end-to-end run (2026-10-08)

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

Of the 5 escalated in-scope questions, 3 had a correct manual in the top 5. On those the model rated its own confidence low (2, 22, 30 out of 100), so the escalations were not caused by retrieval misses alone. For example, the W-10 filing-deadline question (`sdf-001`) retrieved the W-10 manual and still got confidence 2. With 9 in-scope questions, one question moves recall by 0.11, so these numbers show direction only.

## Second run: with the Statewide Rules (2026-10-08)

Same command and settings, after adding the rules to the index (`--output eval_data/results_verify_norerank_rules.jsonl`).

| Metric | Without rules | With rules |
|---|---|---|
| Errors | 0 / 12 | 0 / 12 |
| Recall@5 (manual level) | 0.78 | 0.67 |
| Recall@20 (manual level) | 0.89 | 0.89 |
| Out-of-scope refused | 3 / 3 | 3 / 3 |
| In-scope decisions: answered / caveat / escalated | 3 / 1 / 5 | 2 / 3 / 4 |
| Average latency | 7.0 s | 7.4 s |

How to read it:

- **Rules passages displaced manuals for 3 questions.** For the high-cost-gas questions (`pro-001`, `mds-001`), all 5 top passages came from Rule 3.101 (*Certification for Severance Tax Exemption…*). For the transportation-authority question (`sdf-003`), 4 of 5 came from Rule 3.58 (*Certificate of Compliance and Transportation Authority…*). By title these are the governing rules, but the eval ground truth lists only manuals, so `pro-001` now scores as a retrieval miss. That is the whole drop in Recall@5: an eval-set gap, not necessarily worse retrieval.
- **Confidence rose where rules were retrieved.** `pro-001` went from escalate (30) to caveat (82); `mds-001` from 42 to 62, still escalated.
- **The W-10 deadline question is unchanged** (escalate, confidence 2). No rules passage was retrieved for it, and only one rules passage contains "W-10" (in Rule 3.86, horizontal drainhole wells). Its answer may be in a source TROA does not have, or its ground truth may need checking.
- `pro-002` moved from answered (88) to caveat (82) with no rules passages retrieved, which suggests the model's self-rated confidence varies between runs by itself. That is one more reason to calibrate before trusting the thresholds.

## Third run: judged, and a smoke-test calibrator (2026-10-08)

Same settings as the second run, with the Opus judge on (`eval_data/results_judged_norerank.jsonl`). The judge scores every generated draft, including escalated ones, against the eval set's reference answer. A draft counts as **correct** when correctness = 2 (it gives the reference information) and faithfulness = 2 (nothing unsupported).

| Case | Decision | Raw confidence | Correct? | Judge's reason, shortened |
|---|---|---|---|---|
| `sdf-003` | answered | 90 | ✅ | identifies Form P-4 |
| `def-001` | caveat | 82 | ✅ | defines a P-5 organisation correctly |
| `def-002` | answered | 88 | ✅ | describes Rule 14(b)(2) |
| `pro-002` | answered | 88 | ❌ | omits the key reference detail |
| `pro-001` | caveat | 82 | ❌ | well grounded, but misses part of the reference |
| `mds-001` | escalated | 62 | ❌ | misses the key reference point |
| `mds-002` | escalated | 22 | ❌ | does not give the key distinction |
| `sdf-001` | escalated | 2 | ❌ | declines to give the W-10 deadline |
| `sdf-002` | escalated | 2 | ❌ | says it cannot find the update frequency |

3 of 9 drafts are correct. Every correct draft had confidence ≥ 82, and every draft under 70 was wrong, so raw confidence carries signal. But two confident drafts (82 and 88) were wrong, and one of them (`pro-002`) was released as an autonomous answer.

`python -m src.eval.calibration train --results eval_data/results_judged_norerank.jsonl --output calibration/v0_smoke.json` fits a Platt calibrator: ECE 0.26 raw → 0.17 calibrated, **measured on the same 9 examples it was fitted on**. It maps every raw confidence into 0.25–0.40 (raw 90 → 0.38), so with the 0.70 / 0.85 thresholds it would escalate everything. Two reasons: 9 examples, and L2 regularisation (`C=1.0`) on a 0–1 feature, which holds the slope near zero at this sample size. It shows the loop works; it is not a calibrator to serve. The fitted file is git-ignored (`calibration/*.json`) and reproducible from the committed results.

## Reported but not verifiable

Earlier versions of this README reported a dashboard eval run from 2026-10-08 (hybrid search, router and agent loop, `claude-sonnet-4-6`, 12 questions): Hit@5 0.67, MRR 0.58, OOD refusal 3/3, in-scope refusal 5/9. **No output from that run is saved in the repository**, and it used an earlier dashboard engine that has since been replaced, so it cannot be checked.

## Verify it yourself

Every figure in this README can be reproduced from the repository:

```bash
python -m pytest -q                                  # 82 passed (pip install -r requirements-dev.txt)
ls data/raw/manual | wc -l                           # 36 (after python data/download_data.py)
python -c "import sys; sys.path.insert(0,'data'); import download_data as d; print(len(d.MANUALS))"   # 37

# Committed eval output: count errored cases
python -c "import json; r=[json.loads(l) for l in open('eval_data/results_latest.jsonl')]; print(sum(bool(x['error']) for x in r), 'errors of', len(r))"

# Bundled search index: points, dimension, documents
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(path='qdrant_local'); i = c.get_collection('troa_chunks')
names, off = set(), None
while True:
    pts, off = c.scroll('troa_chunks', with_payload=['doc_name'], limit=1000, offset=off)
    names |= {p.payload['doc_name'] for p in pts}
    if off is None: break
print(i.points_count, i.config.params.vectors.size, len(names))"   # 2941 1024 36
```

Model names are in the `model:` line of each file in `src/serve/prompts/` and in `JUDGE_MODEL` in `src/eval/harness.py`. Guardrail thresholds are in `src/serve/guardrail.py`.
