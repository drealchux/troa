"""
Intent router for TROA: Claude Haiku classifies queries before retrieval.

Identifies intent category, flags out-of-domain queries, and optionally
narrows the retrieval scope to specific documents named in the query.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
import yaml

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

INTENT_TYPES = frozenset({"lookup", "procedural", "definitional", "synthesis", "ood"})


@dataclass
class RouterResult:
    intent: str
    is_ood: bool
    ood_confidence: float
    doc_scope: Optional[list[str]]
    raw_response: str


def _load_prompt(name: str) -> dict:
    path = _PROMPTS_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_router_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in router response: {text!r}")
    return json.loads(m.group())


class Router:
    def __init__(
        self,
        prompt_file: str = "router_v1.yaml",
        api_key: Optional[str] = None,
    ):
        cfg = _load_prompt(prompt_file)
        self.model = cfg["model"]
        self.max_tokens = cfg["max_tokens"]
        self._system_text = cfg["system"]
        self._user_template = cfg["user_template"]
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def route(self, question: str) -> RouterResult:
        user_msg = self._user_template.format(question=question)

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
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text

        try:
            parsed = _parse_router_json(raw)
        except (ValueError, json.JSONDecodeError):
            return RouterResult(
                intent="lookup",
                is_ood=False,
                ood_confidence=0.0,
                doc_scope=None,
                raw_response=raw,
            )

        intent = parsed.get("intent", "lookup")
        if intent not in INTENT_TYPES:
            intent = "lookup"

        return RouterResult(
            intent=intent,
            is_ood=bool(parsed.get("is_ood", False)),
            ood_confidence=float(parsed.get("ood_confidence", 0.0)),
            doc_scope=parsed.get("doc_scope") or None,
            raw_response=raw,
        )
