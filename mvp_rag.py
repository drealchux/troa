"""
mvp_rag.py - Minimum viable RAG pipeline for TROA.

Parses 3 PDF manuals, chunks them, embeds with bge-small, retrieves top-k,
and asks Claude to answer with citations. About 120 lines, no Docker, no Qdrant.

Usage:
    python mvp_rag.py "What's the filing deadline for Form W-10?"
    python mvp_rag.py    # uses a default question
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pypdf
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer


# ---- Config ----
MANUAL_DIR = Path("data/raw/manual")
MANUALS_TO_USE = [
    "oga049_drilling_permit_master.pdf",
    "ola001k_oil_well_status_w10.pdf",
    "oil_gas_field_rules.pdf",
]
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 150        # overlap between chunks
TOP_K = 5
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # ~130 MB, runs fine on CPU
CLAUDE_MODEL = "claude-sonnet-4-5"        # use sonnet for the answer


# ---- Load API key from .env (so you don't have to set env vars manually) ----
def load_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value and not value.startswith("sk-ant-..."):
                    return value
    print("ERROR: ANTHROPIC_API_KEY not found.")
    print("Create a .env file in this directory with:")
    print("  ANTHROPIC_API_KEY=sk-ant-your-real-key-here")
    sys.exit(1)


# ---- Step 1: Parse PDFs ----
def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_num, text) for pages that have real content."""
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) > 50:    # skip whitespace-only cover pages
            pages.append((i, text))
    return pages


# ---- Step 2: Chunk ----
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-window chunker. Crude but works for MVP."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def build_corpus() -> list[dict]:
    """Parse all manuals into a flat list of chunks with metadata."""
    corpus = []
    for name in MANUALS_TO_USE:
        path = MANUAL_DIR / name
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        print(f"  Parsing {name}...")
        for page_num, page_text in extract_pages(path):
            for chunk_idx, chunk in enumerate(chunk_text(page_text)):
                corpus.append({
                    "document": name,
                    "page": page_num,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                })
    return corpus


# ---- Step 3: Embed ----
def embed_corpus(corpus: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in corpus]
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


# ---- Step 4: Retrieve ----
def retrieve(query: str, model: SentenceTransformer, corpus: list[dict],
             embeddings: np.ndarray, k: int = TOP_K) -> list[dict]:
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    # Cosine similarity reduces to dot product when both are normalized
    scores = embeddings @ query_emb
    top_idx = np.argsort(-scores)[:k]
    results = []
    for i in top_idx:
        item = dict(corpus[int(i)])
        item["score"] = float(scores[int(i)])
        results.append(item)
    return results


# ---- Step 5: Generate ----
def generate_answer(query: str, retrieved: list[dict], client: Anthropic) -> str:
    chunks_text = "\n\n".join([
        f"[{i+1}] (source: {c['document']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(retrieved)
    ])
    prompt = f"""You answer questions about Texas Railroad Commission oil and gas regulations.

Use ONLY the retrieved chunks below. If the answer is not clearly in the chunks, say so honestly rather than guessing.
Cite chunks like [1], [2] for every claim. Be concise.

Retrieved chunks:
{chunks_text}

Question: {query}

Answer:"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---- Main ----
def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What information is contained in the Drilling Permit Master dataset?"

    print(f"Query: {query}\n")

    api_key = load_api_key()

    print("Step 1: Parsing PDFs")
    corpus = build_corpus()
    print(f"  -> {len(corpus)} chunks from {len(MANUALS_TO_USE)} manuals\n")

    if not corpus:
        print("ERROR: no chunks were built. Check that PDFs exist in data/raw/manual/")
        sys.exit(1)

    print(f"Step 2: Loading embedding model {EMBED_MODEL}")
    print("  (first run downloads ~130 MB, subsequent runs are fast)")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"\nStep 3: Embedding {len(corpus)} chunks")
    embeddings = embed_corpus(corpus, embedder)

    print(f"\nStep 4: Retrieving top {TOP_K} chunks")
    retrieved = retrieve(query, embedder, corpus, embeddings)
    for i, r in enumerate(retrieved, 1):
        preview = r["text"][:90].replace("\n", " ")
        print(f"  [{i}] {r['document']} p.{r['page']} (score={r['score']:.3f})")
        print(f"      {preview}...")

    print(f"\nStep 5: Generating answer with Claude")
    client = Anthropic(api_key=api_key)
    answer = generate_answer(query, retrieved, client)

    print(f"\n{'='*70}")
    print("ANSWER")
    print('='*70)
    print(answer)
    print('='*70)


if __name__ == "__main__":
    main()
