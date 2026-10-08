"""
Corrective retrieval for TROA: grade the reranked passages, and if they can't
answer the question, rewrite the query and search once more.

Pattern adapted from the agentic week of jamwithai/production-agentic-rag-course
(guardrail -> retrieve -> grade -> rewrite -> generate), with TROA-specific
choices made in src/serve/pipeline.py:

  - One retry only. For regulatory answers, a second failed search should end
    in the guardrail's escalation, not in more rewrites that drift away from
    what the user asked.
  - The rewritten query is used for retrieval only. Candidates from both
    searches are pooled and reranked against the *original* question.
  - The grader's final verdict is recorded on the response
    (retrieval_sufficient) as a retrieval-confidence signal for calibration.

The router already plays the course's guardrail-node role (out-of-domain
refusal), so it is not repeated here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import yaml

from .generate import _format_context
from .rerank import RankedChunk

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_prompt(name: str) -> dict:
    with open(_PROMPTS_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class GradeResult:
    sufficient: bool
    relevant: list[int] = field(default_factory=list)   # 1-based passage numbers
    reason: str = ""


class _PromptCaller:
    def __init__(self, prompt_file: str, client=None, api_key: Optional[str] = None):
        cfg = _load_prompt(prompt_file)
        self.model = cfg["model"]
        self.max_tokens = cfg["max_tokens"]
        self._system_text = cfg["system"]
        self._user_template = cfg["user_template"]
        self._client = client or anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def _call(self, **fields) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self._system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": self._user_template.format(**fields)}],
        )
        return response.content[0].text


class Grader(_PromptCaller):
    def __init__(self, prompt_file: str = "grader_v1.yaml", **kwargs):
        super().__init__(prompt_file, **kwargs)

    def grade(self, question: str, ranked: list[RankedChunk]) -> GradeResult:
        context, _ = _format_context(ranked)
        raw = self._call(question=question, context=context)
        m = _JSON_RE.search(raw)
        try:
            parsed = json.loads(m.group()) if m else {}
        except json.JSONDecodeError:
            parsed = {}
        if not parsed:
            # An unparseable grade must not trigger a rewrite on its own.
            log.warning("Grader returned no JSON; treating passages as sufficient")
            return GradeResult(sufficient=True, reason="grader output unparseable")
        return GradeResult(
            sufficient=bool(parsed.get("sufficient", True)),
            relevant=[int(n) for n in parsed.get("relevant", []) if str(n).isdigit()],
            reason=str(parsed.get("reason", "")),
        )


class Rewriter(_PromptCaller):
    def __init__(self, prompt_file: str = "rewrite_v1.yaml", **kwargs):
        super().__init__(prompt_file, **kwargs)

    def rewrite(self, question: str, reason: str) -> str:
        return self._call(question=question, reason=reason).strip().strip('"')
