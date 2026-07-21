"""
Custom retrieval and answer quality metrics for TROA.

Implements what we measure in EVALUATION.md:
  - Retrieval: Recall@k, Precision@k, MRR
  - Answer quality: faithfulness, relevance, citation accuracy (LLM-as-judge)
  - Judge validation: Cohen's kappa against human labels
  - Refusal: F1 on OOD detection

The judge implementation uses Claude Opus with a versioned prompt. The prompt
itself is in src/serve/prompts/judge_v2.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RetrievalCase:
    """One eval case for retrieval scoring."""

    query_id: str
    retrieved_chunk_ids: list[str]    # in retrieval order
    ground_truth_chunk_ids: set[str]  # set of relevant chunks


def recall_at_k(case: RetrievalCase, k: int) -> float:
    """Did we retrieve at least one ground-truth chunk in the top k?

    Defined per case as 1.0 if any ground-truth chunk appears in top-k, else 0.0.
    The dataset-level recall@k is the mean across cases.
    """
    if not case.ground_truth_chunk_ids:
        return float("nan")
    top_k = set(case.retrieved_chunk_ids[:k])
    return float(bool(top_k & case.ground_truth_chunk_ids))


def precision_at_k(case: RetrievalCase, k: int) -> float:
    """Fraction of top-k chunks that are ground-truth-relevant."""
    if k <= 0:
        return 0.0
    top_k = case.retrieved_chunk_ids[:k]
    hits = sum(1 for cid in top_k if cid in case.ground_truth_chunk_ids)
    return hits / k


def reciprocal_rank(case: RetrievalCase) -> float:
    """1 / rank of the first relevant chunk; 0 if none retrieved."""
    for rank, cid in enumerate(case.retrieved_chunk_ids, start=1):
        if cid in case.ground_truth_chunk_ids:
            return 1.0 / rank
    return 0.0


def aggregate_retrieval(cases: Iterable[RetrievalCase], ks: tuple[int, ...] = (5, 20)) -> dict:
    """Aggregate retrieval metrics across a set of cases."""
    cases = list(cases)
    if not cases:
        return {"n": 0}

    out: dict[str, float | int] = {"n": len(cases)}
    for k in ks:
        recalls = [recall_at_k(c, k) for c in cases]
        precisions = [precision_at_k(c, k) for c in cases]
        out[f"recall@{k}"] = float(np.nanmean(recalls))
        out[f"precision@{k}"] = float(np.nanmean(precisions))

    out["mrr"] = float(np.mean([reciprocal_rank(c) for c in cases]))
    return out


# --------- Answer quality (LLM-as-judge) ---------

JUDGE_RUBRIC = """\
You are an evaluation judge. Score the assistant's answer to the user's question
against the retrieved chunks shown below. Be strict.

Give three integer scores, each on the scale 0, 1, or 2.

FAITHFULNESS:
  0 = the answer contains claims that are not supported by the retrieved chunks
      (hallucination present, even if other parts are grounded)
  1 = the answer is mostly grounded but adds minor unsupported claims
      (e.g. an unsupported example or unsupported caveat)
  2 = every claim in the answer is supported by at least one retrieved chunk

RELEVANCE:
  0 = the answer is off-topic or does not address the question
  1 = the answer partially addresses the question
  2 = the answer directly addresses the question

CITATION_ACCURACY:
  0 = citations are wrong (the cited chunks do not support the claims)
  1 = some citations are correct, some are wrong
  2 = every citation correctly maps to a chunk that supports the cited claim

Respond with ONLY a single JSON object of the form:
{"faithfulness": int, "relevance": int, "citation_accuracy": int, "rationale": str}

The rationale should be one or two sentences explaining the scores.

User question:
{question}

Retrieved chunks (cited as [1], [2], ...):
{chunks}

Assistant's answer:
{answer}
"""


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: int
    relevance: int
    citation_accuracy: int
    rationale: str

    def correct(self) -> int:
        """Binary correctness: used for training the calibrator."""
        return int(self.faithfulness == 2 and self.relevance >= 1)


# --------- Judge validation (Cohen's kappa) ---------

def cohens_kappa(rater_a: list[int], rater_b: list[int], categories: tuple[int, ...] = (0, 1, 2)) -> float:
    """
    Cohen's kappa between two raters on the same items.

    kappa = (p_o - p_e) / (1 - p_e)

    where p_o is observed agreement and p_e is expected agreement by chance.
    Returns NaN if perfect chance agreement.
    """
    assert len(rater_a) == len(rater_b), "Raters must have the same number of items"
    n = len(rater_a)
    if n == 0:
        return float("nan")

    # Confusion matrix
    cm = np.zeros((len(categories), len(categories)), dtype=int)
    cat_index = {c: i for i, c in enumerate(categories)}
    for a, b in zip(rater_a, rater_b):
        cm[cat_index[a], cat_index[b]] += 1

    # Observed agreement
    p_o = np.trace(cm) / n

    # Expected agreement by chance
    row_marginals = cm.sum(axis=1) / n
    col_marginals = cm.sum(axis=0) / n
    p_e = float((row_marginals * col_marginals).sum())

    if p_e == 1.0:
        return float("nan")
    return float((p_o - p_e) / (1 - p_e))


def validate_judge(
    human_scores: list[JudgeScore],
    llm_scores: list[JudgeScore],
    min_kappa: float = 0.6,
) -> dict:
    """
    Compute per-dimension kappa between human and LLM judge.

    Returns a dict with per-dimension kappa and a pass/fail flag.
    """
    assert len(human_scores) == len(llm_scores)
    results = {}
    all_pass = True
    for dim in ("faithfulness", "relevance", "citation_accuracy"):
        human = [getattr(s, dim) for s in human_scores]
        llm = [getattr(s, dim) for s in llm_scores]
        k = cohens_kappa(human, llm)
        results[f"kappa_{dim}"] = k
        if not (k >= min_kappa):
            all_pass = False
    results["n"] = len(human_scores)
    results["min_kappa_threshold"] = min_kappa
    results["pass"] = all_pass
    return results


# --------- Refusal / OOD ---------

@dataclass(frozen=True)
class RefusalCase:
    query_id: str
    is_ood: bool         # ground truth: is this query out of scope?
    refused: bool        # did the system refuse?


def refusal_metrics(cases: Iterable[RefusalCase]) -> dict:
    """
    OOD refusal rate (recall on OOD class), in-scope refusal rate (false positive),
    and F1 treating "refuse" as positive for OOD.
    """
    cases = list(cases)
    if not cases:
        return {"n": 0}

    ood = [c for c in cases if c.is_ood]
    in_scope = [c for c in cases if not c.is_ood]

    ood_refused = sum(c.refused for c in ood)
    in_refused = sum(c.refused for c in in_scope)

    ood_refusal_rate = ood_refused / len(ood) if ood else float("nan")
    in_refusal_rate = in_refused / len(in_scope) if in_scope else float("nan")

    # F1 treating "refuse" as positive class for OOD detection
    tp = ood_refused
    fp = in_refused
    fn = len(ood) - ood_refused
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision and recall and not np.isnan(precision) and not np.isnan(recall)) else float("nan"))

    return {
        "n": len(cases),
        "n_ood": len(ood),
        "n_in_scope": len(in_scope),
        "ood_refusal_rate": ood_refusal_rate,
        "in_scope_refusal_rate": in_refusal_rate,
        "ood_precision": precision,
        "ood_recall": recall,
        "ood_f1": f1,
    }
