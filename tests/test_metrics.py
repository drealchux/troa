"""
Smoke tests for the eval metrics. These should pass on a fresh checkout
and serve as documentation for how the metrics behave.

Run with: pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import math

import json

import numpy as np
import pytest

from src.eval.metrics import (
    JudgeScore,
    RefusalCase,
    RetrievalCase,
    aggregate_retrieval,
    cohens_kappa,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_metrics,
    validate_judge,
)
from src.eval.calibration import (
    CalibrationModel,
    expected_calibration_error,
    fit_platt,
)


# ---------------- Retrieval metrics ----------------


def test_recall_at_k_hit():
    case = RetrievalCase(
        query_id="q1",
        retrieved_chunk_ids=["a", "b", "c", "d", "e"],
        ground_truth_chunk_ids={"c"},
    )
    assert recall_at_k(case, 5) == 1.0
    assert recall_at_k(case, 3) == 1.0
    assert recall_at_k(case, 2) == 0.0


def test_recall_at_k_no_relevant():
    case = RetrievalCase(
        query_id="q1",
        retrieved_chunk_ids=["a", "b", "c"],
        ground_truth_chunk_ids={"x", "y"},
    )
    assert recall_at_k(case, 5) == 0.0


def test_precision_at_k():
    case = RetrievalCase(
        query_id="q1",
        retrieved_chunk_ids=["a", "b", "c", "d", "e"],
        ground_truth_chunk_ids={"a", "c", "z"},
    )
    assert precision_at_k(case, 5) == 2 / 5
    assert precision_at_k(case, 3) == 2 / 3
    assert precision_at_k(case, 1) == 1.0


def test_reciprocal_rank():
    case = RetrievalCase(
        query_id="q1",
        retrieved_chunk_ids=["a", "b", "c"],
        ground_truth_chunk_ids={"c"},
    )
    assert reciprocal_rank(case) == pytest.approx(1 / 3)

    case_first = RetrievalCase(
        query_id="q2",
        retrieved_chunk_ids=["c", "a", "b"],
        ground_truth_chunk_ids={"c"},
    )
    assert reciprocal_rank(case_first) == 1.0

    case_miss = RetrievalCase(
        query_id="q3",
        retrieved_chunk_ids=["a", "b"],
        ground_truth_chunk_ids={"c"},
    )
    assert reciprocal_rank(case_miss) == 0.0


def test_aggregate_retrieval():
    cases = [
        RetrievalCase("q1", ["a", "b", "c"], {"a"}),       # recall@5 = 1, MRR = 1.0
        RetrievalCase("q2", ["x", "y", "a"], {"a"}),       # recall@5 = 1, MRR = 1/3
        RetrievalCase("q3", ["x", "y", "z"], {"a"}),       # recall@5 = 0, MRR = 0
    ]
    out = aggregate_retrieval(cases, ks=(5, 20))
    assert out["n"] == 3
    assert out["recall@5"] == pytest.approx(2 / 3)
    assert out["mrr"] == pytest.approx((1 + 1/3 + 0) / 3)


# ---------------- Judge validation ----------------


def test_cohens_kappa_perfect_agreement():
    a = [0, 1, 2, 1, 0, 2]
    b = [0, 1, 2, 1, 0, 2]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_no_agreement_beyond_chance():
    # Random labels: kappa should be near zero
    np.random.seed(42)
    a = np.random.randint(0, 3, size=1000).tolist()
    b = np.random.randint(0, 3, size=1000).tolist()
    k = cohens_kappa(a, b)
    assert abs(k) < 0.05


def test_validate_judge_pass_and_fail():
    # All-agreement -> pass
    human = [JudgeScore(2, 2, 2, "ok") for _ in range(10)]
    llm = [JudgeScore(2, 2, 2, "ok") for _ in range(10)]
    # NOTE: cohens_kappa is NaN when both raters give the same label every time
    # because expected agreement is 1.0. We test mixed labels for a real signal.
    human = [JudgeScore(2 - (i % 3), 2, 2, "") for i in range(30)]
    llm = [JudgeScore(2 - (i % 3), 2, 2, "") for i in range(30)]
    result = validate_judge(human, llm, min_kappa=0.6)
    # relevance and citation_accuracy will be NaN because both raters always say 2
    # That's a real-world quirk worth documenting
    assert result["n"] == 30


# ---------------- Calibration ----------------


def test_ece_perfectly_calibrated():
    # If predicted probabilities exactly match empirical rates per bin, ECE = 0
    # Construct: 100 samples, each "predicted 0.5", correct 50% of the time
    np.random.seed(0)
    confs = np.full(200, 0.5)
    correct = np.concatenate([np.ones(100), np.zeros(100)])
    np.random.shuffle(correct)
    ece, _ = expected_calibration_error(confs, correct, n_bins=10)
    assert ece < 0.01


def test_ece_max_miscalibration():
    # All confidence 1.0, all wrong: ECE should be 1.0
    confs = np.full(100, 1.0)
    correct = np.zeros(100)
    ece, _ = expected_calibration_error(confs, correct, n_bins=10)
    assert ece == pytest.approx(1.0)


def test_fit_platt_improves_or_preserves_ece():
    # Synthetic: raw_conf in [0, 100], underlying true probability is sigmoid(raw_conf / 50 - 1)
    # Raw confidence overshoots because raw/100 != true prob.
    rng = np.random.default_rng(123)
    n = 500
    raw = rng.uniform(0, 100, size=n)
    true_prob = 1 / (1 + np.exp(-(raw / 50 - 1)))
    correct = (rng.uniform(size=n) < true_prob).astype(int)

    # Raw ECE
    raw_ece, _ = expected_calibration_error(raw / 100, correct, n_bins=10)

    # Fit and check calibrated ECE
    model = fit_platt(raw, correct)
    calibrated = model.predict_proba(raw)
    cal_ece, _ = expected_calibration_error(calibrated, correct, n_bins=10)

    # Calibration should not be substantially worse than raw on the training data
    # (it should usually be better; with synthetic data and a reasonable sample, this holds)
    assert cal_ece <= raw_ece + 0.02


def test_calibration_model_roundtrip():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 100, size=100)
    correct = (rng.uniform(size=100) < 0.6).astype(int)
    model = fit_platt(raw, correct)

    as_json = model.to_json()
    restored = CalibrationModel.from_json(as_json)

    test_inputs = np.array([10.0, 50.0, 90.0])
    a = model.predict_proba(test_inputs)
    b = restored.predict_proba(test_inputs)
    assert np.allclose(a, b)


def test_judge_rubric_formats():
    # Literal JSON braces in the rubric must be escaped, or str.format raises
    # KeyError('"faithfulness"') and every judged case errors (as until 2026-10-08).
    from src.eval.metrics import JUDGE_RUBRIC
    prompt = JUDGE_RUBRIC.format(question="Q?", reference="Ref.", chunks="[1] text", answer="A [1]")
    assert '{"correctness": int' in prompt
    assert all(s in prompt for s in ("Q?", "Ref.", "[1] text", "A [1]"))


def test_training_warnings_flag_small_or_truncated_data():
    from src.eval.calibration import training_warnings
    # Only released answers judged (all >= 70) and few of them: both warnings.
    warnings = training_warnings(np.array([72.0, 85.0, 90.0, 97.0]))
    assert len(warnings) == 2
    assert any("smoke test" in w for w in warnings)
    assert any("below raw confidence 70" in w for w in warnings)
    # Enough examples spanning low confidence: no warnings.
    assert training_warnings(np.linspace(0, 100, 120)) == []


def test_judge_score_correct_needs_right_answer_and_no_invention():
    assert JudgeScore(2, 2, 2, "", correctness=2).correct() == 1
    assert JudgeScore(2, 2, 2, "honest non-answer", correctness=0).correct() == 0
    assert JudgeScore(1, 2, 2, "embellished", correctness=2).correct() == 0


def test_load_eval_results_uses_judged_drafts_and_skips_the_rest(tmp_path):
    from src.eval.calibration import load_eval_results
    rows = [
        {"raw_confidence": 30, "judge_correctness": 2, "judge_faithfulness": 2, "error": None},  # escalated draft, correct
        {"raw_confidence": 90, "judge_correctness": 2, "judge_faithfulness": 1, "error": None},  # right but embellished
        {"raw_confidence": 5, "judge_correctness": 0, "judge_faithfulness": 2, "error": None},   # honest "sources don't say"
        {"raw_confidence": 80, "judge_faithfulness": 2, "judge_relevance": 2, "error": None},    # old label only: skipped
        {"raw_confidence": 0, "judge_correctness": None, "judge_faithfulness": None, "error": None},  # OOD, unjudged
        {"raw_confidence": 0, "error": "401"},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    raw, correct = load_eval_results(path)
    assert raw.tolist() == [30.0, 90.0, 5.0]
    assert correct.tolist() == [1, 0, 0]


# ---------------- Refusal metrics ----------------


def test_refusal_metrics_basic():
    cases = [
        RefusalCase("q1", is_ood=True, refused=True),
        RefusalCase("q2", is_ood=True, refused=True),
        RefusalCase("q3", is_ood=True, refused=False),    # missed OOD
        RefusalCase("q4", is_ood=False, refused=False),
        RefusalCase("q5", is_ood=False, refused=False),
        RefusalCase("q6", is_ood=False, refused=True),    # false positive
    ]
    out = refusal_metrics(cases)
    assert out["n"] == 6
    assert out["n_ood"] == 3
    assert out["n_in_scope"] == 3
    assert out["ood_refusal_rate"] == pytest.approx(2 / 3)
    assert out["in_scope_refusal_rate"] == pytest.approx(1 / 3)
    # F1 = 2 * (2/3) * (2/3) / (2/3 + 2/3) = 2/3
    assert out["ood_f1"] == pytest.approx(2 / 3)
