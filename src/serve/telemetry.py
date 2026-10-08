"""
Per-stage tracing and structured query logs for the TROA pipeline.

Trace: times each pipeline stage (route, retrieve, rerank, grade, generate, ...)
and is returned on every PipelineResponse, like the dashboard's trace panel.

QueryLog: appends one JSON line per answered question. The log is TROA's
production data source: refit the calibrator on judged samples of it, and mine
escalations for new eval-set questions.

Pattern adapted from the observability week of
jamwithai/production-agentic-rag-course (Langfuse spans with latency and
cost). A local JSONL file keeps TROA dependency-free; the records carry the
same fields a tracing backend would need.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Union


class Trace:
    def __init__(self) -> None:
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name: str, detail: str = "") -> Iterator[dict]:
        """Time a block. Set entry["detail"] inside the block to annotate it."""
        entry = {"stage": name, "seconds": 0.0, "detail": detail}
        t0 = time.perf_counter()
        try:
            yield entry
        finally:
            entry["seconds"] = round(time.perf_counter() - t0, 4)
            self.stages.append(entry)

    @property
    def total_seconds(self) -> float:
        return round(sum(s["seconds"] for s in self.stages), 4)


class QueryLog:
    """Thread-safe JSONL appender."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record},
            default=str,
        )
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def open_query_log(path: Optional[Union[str, Path]]) -> Optional[QueryLog]:
    return QueryLog(path) if path else None
