"""
Guardrail policy for TROA: maps confidence to an action.

Single source of truth for the thresholds, shared by src/serve/pipeline.py and
dashboard/engine.py. Kept free of heavy imports so the dashboard can use it
without Qdrant or the reranker.

Policy (three bands plus out-of-scope refusal):
    router OOD with ood_confidence >= OOD_CUTOFF  -> refuse_ood
    confidence >= THRESHOLD_AUTONOMOUS            -> autonomous
    THRESHOLD_CAVEAT <= confidence < AUTONOMOUS   -> caveat (answer + banner)
    confidence < THRESHOLD_CAVEAT                 -> escalate (answer withheld)

EVALUATION.md describes a four-band target policy; reconciling it with this
one is planned once a calibrator has been fitted.
"""

from __future__ import annotations

THRESHOLD_AUTONOMOUS = 0.85
THRESHOLD_CAVEAT = 0.70
OOD_CUTOFF = 0.85

DECISIONS = {
    "autonomous": "Autonomous answer",
    "caveat": "Answer with caveat",
    "escalate": "Escalate",
    "refuse_ood": "Refuse (out of scope)",
}


def is_ood_refusal(is_ood: bool, ood_confidence: float) -> bool:
    return bool(is_ood) and ood_confidence >= OOD_CUTOFF


def decide(confidence: float) -> str:
    """Return 'autonomous', 'caveat', or 'escalate' for a [0, 1] confidence."""
    if confidence >= THRESHOLD_AUTONOMOUS:
        return "autonomous"
    if confidence >= THRESHOLD_CAVEAT:
        return "caveat"
    return "escalate"
