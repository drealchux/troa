"""
Evaluation harness for TROA.

Loads eval_data/eval_set_sample.yaml, runs each question through the full
pipeline, calls the Claude Opus judge, and writes a JSONL result file.

Usage:
    python -m src.eval.harness --eval-set eval_data/eval_set_sample.yaml \
        --output eval_data/results_latest.jsonl

    # Skip judge (fast retrieval-only eval):
    python -m src.eval.harness --eval-set eval_data/eval_set_sample.yaml \
        --output eval_data/results_latest.jsonl --no-judge

    # Dry run (skip pipeline, just validate the YAML):
    python -m src.eval.harness --eval-set eval_data/eval_set_sample.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import os

import anthropic
import yaml

from .metrics import (
    JUDGE_RUBRIC,
    JudgeScore,
    RetrievalCase,
    RefusalCase,
    aggregate_retrieval,
    refusal_metrics,
)

log = logging.getLogger(__name__)

JUDGE_MODEL = "claude-opus-4-7"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


#  Eval case schema

@dataclass
class EvalCase:
    id: str
    category: str
    difficulty: int
    question: str
    ground_truth_answer: str
    ground_truth_chunks: list[dict]   # [{document, section}, ...]
    notes: str

    @property
    def is_ood(self) -> bool:
        return self.category == "ood"

    def ground_truth_doc_names(self) -> set[str]:
        """Return set of doc_name values (path.stem — no .pdf extension)."""
        names = set()
        for c in self.ground_truth_chunks:
            doc = c.get("document", "")
            # Strip .pdf extension to match stored doc_name (path.stem)
            if doc.lower().endswith(".pdf"):
                doc = doc[:-4]
            if doc:
                names.add(doc)
        return names


@dataclass
class EvalResult:
    case_id: str
    category: str
    difficulty: int
    question: str
    answer: str
    is_ood: bool
    refused: bool
    raw_confidence: int
    calibrated_confidence: float
    intent: str
    retrieved_chunk_ids: list[str]
    ground_truth_doc_names: list[str]
    recall_at_5: Optional[float]
    recall_at_20: Optional[float]
    judge_faithfulness: Optional[int]
    judge_relevance: Optional[int]
    judge_citation_accuracy: Optional[int]
    judge_rationale: Optional[str]
    latency_s: float
    error: Optional[str]


# YAML loading 

def load_eval_set(path: Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = []
    for q in data["questions"]:
        cases.append(EvalCase(
            id=q["id"],
            category=q["category"],
            difficulty=q["difficulty"],
            question=q["question"],
            ground_truth_answer=q.get("ground_truth_answer", ""),
            ground_truth_chunks=q.get("ground_truth_chunks") or [],
            notes=q.get("notes", ""),
        ))
    return cases


# Judge 

def _call_judge(
    client: anthropic.Anthropic,
    question: str,
    answer: str,
    chunks: list[str],
) -> Optional[JudgeScore]:
    chunks_block = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(chunks))
    prompt = JUDGE_RUBRIC.format(
        question=question,
        chunks=chunks_block,
        answer=answer,
    )
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        m = _JSON_RE.search(raw)
        if not m:
            log.warning("Judge returned no JSON for case: %s", question[:60])
            return None
        parsed = json.loads(m.group())
        return JudgeScore(
            faithfulness=int(parsed["faithfulness"]),
            relevance=int(parsed["relevance"]),
            citation_accuracy=int(parsed["citation_accuracy"]),
            rationale=str(parsed.get("rationale", "")),
        )
    except Exception as exc:
        log.warning("Judge call failed: %s", exc)
        return None


# Retrieval recall helper 

def _retrieval_recall(
    case: EvalCase,
    retrieved_doc_names: list[str],
    k: int,
) -> Optional[float]:
    """Recall@k at the document level (no chunk-level IDs in ground truth)."""
    gt_docs = case.ground_truth_doc_names()
    if not gt_docs:
        return None
    top_k = set(retrieved_doc_names[:k])
    return float(bool(top_k & gt_docs))


# ── Main runner 

def run_harness(
    eval_set_path: Path,
    output_path: Path,
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: Optional[str] = None,
    use_judge: bool = True,
    dry_run: bool = False,
) -> None:
    cases = load_eval_set(eval_set_path)
    print(f"Loaded {len(cases)} eval cases from {eval_set_path}")

    if dry_run:
        print("Dry run — validating YAML only, no pipeline calls.")
        for c in cases:
            print(f"  [{c.id}] {c.category} d={c.difficulty}: {c.question[:70]}")
        return

    from src.serve.pipeline import Pipeline
    pipe = Pipeline(qdrant_url=qdrant_url, qdrant_path=qdrant_path)

    judge_client = anthropic.Anthropic() if use_judge else None

    results: list[EvalResult] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for case in cases:
            print(f"  [{case.id}] {case.question[:70]}...")
            t0 = time.time()
            error = None

            try:
                resp = pipe.run(case.question)
                latency = time.time() - t0

                retrieved_doc_names = [c.doc_name for c in resp.retrieved_chunks]
                recalled_at_5 = _retrieval_recall(case, retrieved_doc_names, 5)
                recalled_at_20 = _retrieval_recall(case, retrieved_doc_names, 20)

                judge: Optional[JudgeScore] = None
                if use_judge and not resp.refused and resp.ranked_chunks:
                    passage_texts = [rc.chunk.text for rc in resp.ranked_chunks]
                    judge = _call_judge(
                        judge_client,
                        case.question,
                        resp.answer,
                        passage_texts,
                    )

                result = EvalResult(
                    case_id=case.id,
                    category=case.category,
                    difficulty=case.difficulty,
                    question=case.question,
                    answer=resp.answer,
                    is_ood=resp.is_ood,
                    refused=resp.refused,
                    raw_confidence=resp.raw_confidence,
                    calibrated_confidence=resp.calibrated_confidence,
                    intent=resp.intent,
                    retrieved_chunk_ids=[c.chunk_id for c in resp.retrieved_chunks],
                    ground_truth_doc_names=list(case.ground_truth_doc_names()),
                    recall_at_5=recalled_at_5,
                    recall_at_20=recalled_at_20,
                    judge_faithfulness=judge.faithfulness if judge else None,
                    judge_relevance=judge.relevance if judge else None,
                    judge_citation_accuracy=judge.citation_accuracy if judge else None,
                    judge_rationale=judge.rationale if judge else None,
                    latency_s=round(latency, 3),
                    error=None,
                )

            except Exception as exc:
                log.error("Case %s failed: %s", case.id, exc)
                latency = time.time() - t0
                result = EvalResult(
                    case_id=case.id,
                    category=case.category,
                    difficulty=case.difficulty,
                    question=case.question,
                    answer="",
                    is_ood=case.is_ood,
                    refused=False,
                    raw_confidence=0,
                    calibrated_confidence=0.0,
                    intent="unknown",
                    retrieved_chunk_ids=[],
                    ground_truth_doc_names=list(case.ground_truth_doc_names()),
                    recall_at_5=None,
                    recall_at_20=None,
                    judge_faithfulness=None,
                    judge_relevance=None,
                    judge_citation_accuracy=None,
                    judge_rationale=None,
                    latency_s=round(latency, 3),
                    error=str(exc),
                )

            results.append(result)
            out_f.write(json.dumps(asdict(result)) + "\n")
            out_f.flush()

    _print_summary(results)
    print(f"\nResults written to {output_path}")


def _print_summary(results: list[EvalResult]) -> None:
    n = len(results)
    errors = sum(1 for r in results if r.error)
    refused = sum(1 for r in results if r.refused)

    # Retrieval recall (non-OOD, non-error cases with ground truth)
    r5_vals = [r.recall_at_5 for r in results if r.recall_at_5 is not None]
    r20_vals = [r.recall_at_20 for r in results if r.recall_at_20 is not None]
    avg_r5 = sum(r5_vals) / len(r5_vals) if r5_vals else float("nan")
    avg_r20 = sum(r20_vals) / len(r20_vals) if r20_vals else float("nan")

    # Judge scores
    faith = [r.judge_faithfulness for r in results if r.judge_faithfulness is not None]
    rel = [r.judge_relevance for r in results if r.judge_relevance is not None]
    avg_faith = sum(faith) / len(faith) if faith else float("nan")
    avg_rel = sum(rel) / len(rel) if rel else float("nan")

    # Refusal metrics (OOD detection)
    refusal_cases = [
        RefusalCase(
            query_id=r.case_id,
            is_ood=r.category == "ood",
            refused=r.refused,
        )
        for r in results
        if not r.error
    ]
    ref_m = refusal_metrics(refusal_cases)

    # Avg latency
    lats = [r.latency_s for r in results if not r.error]
    avg_lat = sum(lats) / len(lats) if lats else float("nan")

    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY  n={n}  errors={errors}  refused={refused}")
    print(f"{'='*50}")
    print(f"Retrieval  recall@5={avg_r5:.2f}  recall@20={avg_r20:.2f}")
    print(f"Judge      faithfulness={avg_faith:.2f}  relevance={avg_rel:.2f}")
    print(f"OOD        refusal_f1={ref_m.get('ood_f1', float('nan')):.2f}  "
          f"ood_recall={ref_m.get('ood_refusal_rate', float('nan')):.2f}  "
          f"fp_rate={ref_m.get('in_scope_refusal_rate', float('nan')):.2f}")
    print(f"Latency    avg={avg_lat:.1f}s")
    print(f"{'='*50}")


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval-set", type=Path,
                        default=Path("eval_data/eval_set_sample.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("eval_data/results_latest.jsonl"))
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-path", default=None,
                        help="Local Qdrant storage path (overrides --qdrant-url)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge calls (faster, retrieval metrics only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate YAML only, no pipeline or judge calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    run_harness(
        eval_set_path=args.eval_set,
        output_path=args.output,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        use_judge=not args.no_judge,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
