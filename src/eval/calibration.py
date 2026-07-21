"""
Confidence calibration for TROA.

Raw LLM confidence scores are systematically miscalibrated. The Generator emits
a 0-100 confidence; this module fits a Platt scaler that maps that raw score
to a calibrated probability of correctness, and reports diagnostics (reliability
diagram, ECE, Brier).

Usage:
    # Train calibration on eval results
    python -m src.eval.calibration --results eval_results/latest.jsonl --output calibration/v1.json

    # Inspect calibration quality
    python -m src.eval.calibration --inspect calibration/v1.json --results eval_results/holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationModel:
    """A fitted Platt scaling calibrator. Two parameters, easy to inspect."""

    coef: float           # logistic regression coefficient on raw_confidence
    intercept: float      # logistic regression intercept
    feature_mean: float   # for documentation; we standardize raw confidence to [0, 1]
    feature_std: float
    n_train: int
    train_ece: float
    train_brier: float

    def predict_proba(self, raw_confidence: np.ndarray | float) -> np.ndarray:
        """Map raw confidence (0-100) to calibrated probability (0-1)."""
        x = np.atleast_1d(np.asarray(raw_confidence, dtype=float)) / 100.0
        z = self.coef * x + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "CalibrationModel":
        return cls(**data)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    Expected Calibration Error.

    Bins confidences into n_bins equal-width bins, computes per-bin |accuracy - mean_confidence|,
    weighted by the bin's share of the total samples. Returns the overall ECE and per-bin diagnostics.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    assert confidences.shape == correct.shape, "Shape mismatch"
    n = len(confidences)
    if n == 0:
        return 0.0, []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)

    ece = 0.0
    per_bin = []
    for b in range(n_bins):
        mask = bin_idx == b
        bin_n = int(mask.sum())
        if bin_n == 0:
            per_bin.append({"bin": b, "lo": float(bin_edges[b]), "hi": float(bin_edges[b + 1]),
                            "n": 0, "accuracy": None, "mean_confidence": None, "gap": None})
            continue

        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (bin_n / n) * gap

        per_bin.append({
            "bin": b,
            "lo": float(bin_edges[b]),
            "hi": float(bin_edges[b + 1]),
            "n": bin_n,
            "accuracy": bin_acc,
            "mean_confidence": bin_conf,
            "gap": gap,
        })

    return ece, per_bin


def fit_platt(raw_confidences: np.ndarray, correct: np.ndarray) -> CalibrationModel:
    """
    Fit Platt scaling: logistic regression on the single raw_confidence feature.

    We could use sklearn's CalibratedClassifierCV, but fitting it directly keeps the
    coefficients inspectable. With only ~100 training points, regularization matters,
    so we use L2 with a mild C.
    """
    x = np.asarray(raw_confidences, dtype=float).reshape(-1, 1) / 100.0
    y = np.asarray(correct, dtype=int)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(x, y)

    cal = CalibrationModel(
        coef=float(clf.coef_[0, 0]),
        intercept=float(clf.intercept_[0]),
        feature_mean=float(x.mean()),
        feature_std=float(x.std()),
        n_train=len(x),
        train_ece=0.0,    # filled in below
        train_brier=0.0,
    )

    calibrated = cal.predict_proba(raw_confidences)
    cal.train_ece, _ = expected_calibration_error(calibrated, y, n_bins=10)
    cal.train_brier = float(brier_score_loss(y, calibrated))
    return cal


def reliability_diagram_data(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Per-bin data points suitable for plotting a reliability diagram."""
    _, per_bin = expected_calibration_error(confidences, correct, n_bins)
    return per_bin


def load_eval_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load eval results JSONL. Each line is a record from a single eval question.
    Expected fields:
      - raw_confidence: 0-100 from the LLM generator
      - faithfulness: 0/1/2 from the judge
      - relevance: 0/1/2 from the judge

    Correct is derived as: faithfulness == 2 AND relevance >= 1.
    """
    raw = []
    correct = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                rc = float(record["raw_confidence"])
                faith = int(record["faithfulness"])
                rel = int(record["relevance"])
            except (KeyError, ValueError, TypeError) as exc:
                # Skip malformed records but tell the user
                print(f"WARN skipping record: {exc}: {record}")
                continue
            raw.append(rc)
            correct.append(int(faith == 2 and rel >= 1))

    return np.array(raw), np.array(correct)


def print_reliability_table(per_bin: list[dict]) -> None:
    print(f"  {'bin':>3}  {'range':>15}  {'n':>5}  {'acc':>7}  {'conf':>7}  {'gap':>7}")
    for b in per_bin:
        if b["n"] == 0:
            print(f"  {b['bin']:>3}  [{b['lo']:.2f}, {b['hi']:.2f})  {b['n']:>5}  {'--':>7}  {'--':>7}  {'--':>7}")
        else:
            print(f"  {b['bin']:>3}  [{b['lo']:.2f}, {b['hi']:.2f})  {b['n']:>5}  "
                  f"{b['accuracy']:>7.3f}  {b['mean_confidence']:>7.3f}  {b['gap']:>7.3f}")


def cmd_train(args) -> None:
    raw, correct = load_eval_results(args.results)
    print(f"Loaded {len(raw)} eval records from {args.results}")
    print(f"  Correct: {int(correct.sum())} / {len(correct)} ({100 * correct.mean():.1f}%)")
    if len(np.unique(correct)) < 2:
        raise ValueError(
            "Cannot fit calibration: training labels are all the same class. "
            "Expand the eval set or check the judge output."
        )

    # Raw (uncalibrated) ECE for comparison
    raw_scaled = raw / 100.0
    raw_ece, raw_per_bin = expected_calibration_error(raw_scaled, correct, n_bins=10)
    raw_brier = float(brier_score_loss(correct, raw_scaled))
    print(f"\nUncalibrated (raw LLM confidence):")
    print(f"  ECE   = {raw_ece:.4f}")
    print(f"  Brier = {raw_brier:.4f}")
    print(f"  Reliability table:")
    print_reliability_table(raw_per_bin)

    # Fit
    model = fit_platt(raw, correct)
    print(f"\nFitted Platt scaling:")
    print(f"  coef        = {model.coef:.4f}")
    print(f"  intercept   = {model.intercept:.4f}")
    print(f"  train ECE   = {model.train_ece:.4f}")
    print(f"  train Brier = {model.train_brier:.4f}")

    calibrated = model.predict_proba(raw)
    cal_per_bin = reliability_diagram_data(calibrated, correct, n_bins=10)
    print(f"\nCalibrated (Platt-scaled):")
    print(f"  Reliability table:")
    print_reliability_table(cal_per_bin)

    improvement = raw_ece - model.train_ece
    print(f"\nECE improvement: {improvement:+.4f}  ({raw_ece:.4f} -> {model.train_ece:.4f})")
    if improvement < 0:
        print("  WARNING: calibration made things worse. Possible causes:")
        print("    - Eval set too small for Platt to generalize")
        print("    - Raw confidence not informative (constant or noisy)")
        print("    - Train/test distribution mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(model.to_json(), f, indent=2)
    print(f"\nWrote calibration model to {args.output}")


def cmd_inspect(args) -> None:
    with open(args.inspect, "r") as f:
        model = CalibrationModel.from_json(json.load(f))
    raw, correct = load_eval_results(args.results)

    calibrated = model.predict_proba(raw)
    ece, per_bin = expected_calibration_error(calibrated, correct, n_bins=10)
    brier = float(brier_score_loss(correct, calibrated))
    ll = float(log_loss(correct, np.clip(calibrated, 1e-6, 1 - 1e-6), labels=[0, 1]))

    print(f"Calibration inspection on {args.results}")
    print(f"  n            = {len(raw)}")
    print(f"  ECE          = {ece:.4f}")
    print(f"  Brier        = {brier:.4f}")
    print(f"  Log loss     = {ll:.4f}")
    print(f"  Coefficient  = {model.coef:.4f}")
    print(f"  Intercept    = {model.intercept:.4f}")
    print(f"\nReliability table (calibrated):")
    print_reliability_table(per_bin)

    if ece > 0.08:
        print(f"\nWARNING: ECE = {ece:.4f} exceeds the v1 target of 0.08.")
        print(f"  Consider: re-fitting with more data, or switching to isotonic regression.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Train (default if no subcommand)
    p_train = sub.add_parser("train", help="Fit a new calibration model")
    p_train.add_argument("--results", type=Path, required=True, help="Path to eval results JSONL")
    p_train.add_argument("--output", type=Path, default=Path("calibration/v1.json"))

    p_inspect = sub.add_parser("inspect", help="Inspect calibration quality on new data")
    p_inspect.add_argument("--inspect", type=Path, required=True, help="Calibration JSON to load")
    p_inspect.add_argument("--results", type=Path, required=True, help="Eval results to test against")

    args = parser.parse_args()
    if args.cmd == "inspect":
        cmd_inspect(args)
    else:
        # Default to train; let the user pass --results --output at the top level too
        if not hasattr(args, "results"):
            parser.error("Specify a subcommand (train | inspect)")
        cmd_train(args)


if __name__ == "__main__":
    main()
