"""
Generator for TROA: Claude Sonnet with cached system prompt, citation formatting,
and structured confidence extraction.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
import yaml

from .rerank import RankedChunk

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_CONF_RE = re.compile(r"<confidence>\s*(\d+)\s*</confidence>", re.IGNORECASE)


@dataclass
class GeneratorResult:
    answer: str
    raw_confidence: int
    context_passages: list[str]
    usage: dict


def _load_prompt(name: str) -> dict:
    path = _PROMPTS_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_context(ranked: list[RankedChunk]) -> tuple[str, list[str]]:
    passages = []
    lines = []
    for i, rc in enumerate(ranked, start=1):
        src = f"{rc.chunk.doc_name} | {' > '.join(rc.chunk.section_path[1:])}"
        passage = rc.chunk.text
        passages.append(passage)
        lines.append(f"[{i}] ({src})\n{passage}")
    return "\n\n".join(lines), passages


class Generator:
    def __init__(
        self,
        prompt_file: str = "generate_v1.yaml",
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

    def generate(
        self,
        question: str,
        ranked_chunks: list[RankedChunk],
    ) -> GeneratorResult:
        context_block, passages = _format_context(ranked_chunks)
        user_msg = self._user_template.format(
            context=context_block,
            question=question,
        )

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

        raw_text = response.content[0].text

        m = _CONF_RE.search(raw_text)
        raw_confidence = int(m.group(1)) if m else 50
        raw_confidence = max(0, min(100, raw_confidence))

        answer = _CONF_RE.sub("", raw_text).rstrip()

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        }

        return GeneratorResult(
            answer=answer,
            raw_confidence=raw_confidence,
            context_passages=passages,
            usage=usage,
        )
