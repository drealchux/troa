"""
Generator for TROA: Claude Sonnet with cached system prompt, citation formatting,
and structured confidence extraction.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

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

    def _request(self, question: str, ranked_chunks: list[RankedChunk]) -> tuple[dict, list[str]]:
        context_block, passages = _format_context(ranked_chunks)
        user_msg = self._user_template.format(
            context=context_block,
            question=question,
        )
        request = dict(
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
        return request, passages

    def generate(
        self,
        question: str,
        ranked_chunks: list[RankedChunk],
    ) -> GeneratorResult:
        request, passages = self._request(question, ranked_chunks)
        response = self._client.messages.create(**request)
        return _parse_response(response.content[0].text, response.usage, passages)

    def stream(self, question: str, ranked_chunks: list[RankedChunk]) -> "GenerationStream":
        """Stream the answer text; the parsed GeneratorResult is on .result afterwards."""
        request, passages = self._request(question, ranked_chunks)
        return GenerationStream(self._client, request, passages)


class GenerationStream:
    """Iterate to receive text deltas. The trailing <confidence> tag is part of
    the stream, so callers showing text live should hide it (see
    visible_text). After iteration, .result holds the parsed GeneratorResult."""

    def __init__(self, client, request: dict, passages: list[str]):
        self._client = client
        self._request = request
        self._passages = passages
        self.result: Optional[GeneratorResult] = None

    def __iter__(self) -> Iterator[str]:
        parts: list[str] = []
        with self._client.messages.stream(**self._request) as stream:
            for delta in stream.text_stream:
                parts.append(delta)
                yield delta
            final = stream.get_final_message()
        self.result = _parse_response("".join(parts), final.usage, self._passages)


def visible_text(text: str) -> str:
    """Text safe to show while streaming: drops a complete or partial confidence tag."""
    cut = text.lower().find("<conf")
    if cut != -1:
        return text[:cut]
    # A tag split across deltas: hide a trailing "<", "<c", ... until it resolves.
    for n in range(4, 0, -1):
        if text.lower().endswith("<conf"[:n]):
            return text[:-n]
    return text


def _parse_response(raw_text: str, usage_obj, passages: list[str]) -> GeneratorResult:
    m = _CONF_RE.search(raw_text)
    raw_confidence = int(m.group(1)) if m else 50
    raw_confidence = max(0, min(100, raw_confidence))

    answer = _CONF_RE.sub("", raw_text).rstrip()

    usage = {
        "input_tokens": usage_obj.input_tokens,
        "output_tokens": usage_obj.output_tokens,
        "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    }

    return GeneratorResult(
        answer=answer,
        raw_confidence=raw_confidence,
        context_passages=passages,
        usage=usage,
    )
