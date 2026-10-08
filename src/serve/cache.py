"""
Exact-match answer cache for the TROA pipeline.

Pattern adapted from the caching week of jamwithai/production-agentic-rag-course
(Redis, TTL, graceful fallback when Redis is down), with two TROA rules:

  1. The key covers everything that changes an answer, not just the question:
     prompt files, guardrail thresholds, calibrator, models, search settings,
     and the corpus fingerprint. Editing a prompt, refitting the calibrator, or
     re-ingesting therefore never serves a stale answer.
  2. Only answers the guardrail released (autonomous or caveat) are cached.
     Escalations and refusals are recomputed, so a later corpus fix is picked up.

Backend: Redis when REDIS_URL is set and the `redis` package is installed,
otherwise a process-local dict. Any Redis error falls back to the dict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
CACHEABLE_DECISIONS = frozenset({"autonomous", "caveat"})


def normalize_question(question: str) -> str:
    q = re.sub(r"\s+", " ", question.strip().lower())
    return q.rstrip("?.! ")


def prompts_fingerprint(names: list[str]) -> str:
    """Hash of the named prompt files' contents."""
    h = hashlib.sha1()
    for name in sorted(names):
        h.update(name.encode())
        h.update((_PROMPTS_DIR / name).read_bytes())
    return h.hexdigest()[:12]


def cache_key(question: str, versions: dict) -> str:
    raw = json.dumps({"q": normalize_question(question), "v": versions}, sort_keys=True)
    return "troa:answer:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


class AnswerCache:
    def __init__(self, ttl_seconds: int = 7 * 24 * 3600, redis_url: Optional[str] = None,
                 max_items: int = 1000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._mem: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._redis = None
        if redis_url:
            try:
                import redis
                client = redis.Redis.from_url(redis_url, socket_timeout=1)
                client.ping()
                self._redis = client
            except Exception as exc:   # missing package or unreachable server
                log.warning("Redis unavailable (%s); using in-memory answer cache", exc)

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def get(self, key: str) -> Optional[dict]:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception as exc:
                log.warning("Redis get failed (%s); falling back to memory", exc)
                self._redis = None
        with self._lock:
            hit = self._mem.get(key)
            if hit is None:
                return None
            expires, raw = hit
            if expires < time.time():
                del self._mem[key]
                return None
            return json.loads(raw)

    def set(self, key: str, value: dict) -> None:
        raw = json.dumps(value, default=str)
        if self._redis is not None:
            try:
                self._redis.set(key, raw, ex=self.ttl)
                return
            except Exception as exc:
                log.warning("Redis set failed (%s); falling back to memory", exc)
                self._redis = None
        with self._lock:
            if len(self._mem) >= self.max_items:
                self._mem.pop(next(iter(self._mem)))   # drop the oldest entry
            self._mem[key] = (time.time() + self.ttl, raw)
