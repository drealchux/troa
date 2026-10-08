"""
Ask TROA a question from the terminal.

    python ask.py "What does the Drilling Permit Master dataset contain?"
    python ask.py                      # interactive: one question per line, blank line to quit
    python ask.py --hybrid "When must an inactive well be plugged?"

Uses the search index committed in qdrant_local/ and ANTHROPIC_API_KEY from .env,
so no ingestion is needed. The first question loads the embedding model
(bge-large-en-v1.5, about 1.34 GB; downloaded once).

Every answer carries a guardrail decision. An escalated answer is withheld:
TROA was not confident enough, so check the cited sources or rrc.texas.gov.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import replace

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import anthropic  # noqa: E402

from src.config import Settings, build_pipeline  # noqa: E402
from src.serve.guardrail import DECISIONS  # noqa: E402

ICONS = {"autonomous": "✅", "caveat": "⚠️", "escalate": "⏫", "refuse_ood": "⛔"}


def print_response(resp) -> None:
    print()
    print(f"{ICONS.get(resp.decision, '')} {DECISIONS.get(resp.decision, resp.decision)}"
          f"  ·  confidence {resp.calibrated_confidence:.2f}"
          f"{'  ·  cached' if resp.cached else f'  ·  {resp.latency_s:.1f}s'}")
    print()
    for para in resp.answer.split("\n"):
        print(textwrap.fill(para, width=100) if para.strip() else "")
    if resp.ranked_chunks:
        print("\nSources:")
        for i, rc in enumerate(resp.ranked_chunks, start=1):
            c = rc.chunk
            section = " > ".join(c.section_path[1:]) or "(start of document)"
            print(f"  [{i}] {c.doc_name}, page {c.page_num}: {section[:90]}")
    print()


def ask(pipe, question: str) -> int:
    try:
        print_response(pipe.run(question))
        return 0
    except anthropic.AuthenticationError:
        print("\nThe Anthropic API rejected the key in .env (401). Check ANTHROPIC_API_KEY.")
    except anthropic.APIError as exc:
        print(f"\nAnthropic API error: {exc}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask TROA a question about RRC oil and gas manuals and the Statewide Rules.")
    parser.add_argument("question", nargs="*", help="The question (omit for interactive mode)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Combine keyword (BM25) and semantic search")
    parser.add_argument("--agentic", action="store_true",
                        help="Grade the passages and retry once with a rewritten query")
    parser.add_argument("--rerank", action="store_true",
                        help="Rerank with bge-reranker-large (2.24 GB download)")
    args = parser.parse_args()
    # Windows consoles default to a code page that cannot print the decision icons.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings.from_env()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1
    if settings.qdrant_path and not os.path.isdir(settings.qdrant_path):
        print(f"Search index not found at {settings.qdrant_path}. "
              "See README, Quick start, or build it with: python tasks.py ingest")
        return 1
    settings = replace(
        settings,
        search_mode="hybrid" if args.hybrid else settings.search_mode,
        agentic=args.agentic or settings.agentic,
        rerank=args.rerank or settings.rerank,
    )

    print("Loading TROA (the first run downloads the embedding model)...", flush=True)
    pipe = build_pipeline(settings)

    if args.question:
        return ask(pipe, " ".join(args.question))

    print("Ask a question about RRC oil and gas regulation. Blank line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        ask(pipe, question)


if __name__ == "__main__":
    sys.exit(main())
