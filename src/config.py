"""
Runtime settings for TROA services, read from the environment (and .env).

One place for the knobs the API and other long-running entry points share.
See .env.example for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _opt(name: str) -> Optional[str]:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: Optional[str]
    qdrant_url: str
    qdrant_path: Optional[str]
    search_mode: str            # vector | hybrid
    agentic: bool               # grade → rewrite → retry once
    rerank: bool                # cross-encoder reranker; off keeps top-k by retrieval score
    calibration_path: Optional[str]     # calibration/v1.json from `calibration.py train`
    redis_url: Optional[str]    # answer cache backend; in-memory when unset
    cache_enabled: bool
    cache_ttl_seconds: int
    query_log_path: Optional[str]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=_opt("ANTHROPIC_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_path=_opt("QDRANT_PATH"),
            search_mode=os.getenv("TROA_SEARCH_MODE", "vector"),
            agentic=_flag("TROA_AGENTIC"),
            rerank=_flag("TROA_RERANK", True),
            calibration_path=_opt("TROA_CALIBRATION"),
            redis_url=_opt("REDIS_URL"),
            cache_enabled=_flag("TROA_CACHE", True),
            cache_ttl_seconds=int(os.getenv("TROA_CACHE_TTL", str(7 * 24 * 3600))),
            query_log_path=_opt("TROA_QUERY_LOG"),
        )


def load_calibrator(path: Optional[str]):
    """Load a fitted CalibrationModel, or None when no path is given."""
    if not path:
        return None
    import json
    from src.eval.calibration import CalibrationModel
    with open(path, encoding="utf-8") as f:
        return CalibrationModel.from_json(json.load(f))


def build_pipeline(settings: Optional[Settings] = None):
    """Construct a Pipeline from settings."""
    from src.serve.cache import AnswerCache
    from src.serve.pipeline import Pipeline

    s = settings or Settings.from_env()
    return Pipeline(
        qdrant_url=s.qdrant_url,
        qdrant_path=s.qdrant_path,
        api_key=s.anthropic_api_key,
        calibrator=load_calibrator(s.calibration_path),
        search_mode=s.search_mode,
        agentic=s.agentic,
        rerank=s.rerank,
        cache=AnswerCache(ttl_seconds=s.cache_ttl_seconds, redis_url=s.redis_url)
        if s.cache_enabled else None,
        query_log=s.query_log_path,
    )
