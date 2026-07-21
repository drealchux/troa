"""
Ingestion pipeline for the TROA RAG corpus.

Orchestrates: parse PDF → chunk → embed → store in Qdrant.

Usage:
    # Full ingest (requires Qdrant running on localhost:6333)
    python -m src.ingest.pipeline --corpus data/raw/manual/

    # Dry run: parse and chunk only, no embedding or storage (fast sanity check)
    python -m src.ingest.pipeline --corpus data/raw/manual/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .chunk import ChunkingConfig, chunk_document
from .embed import Embedder
from .parse import parse_pdf
from .store import QdrantStore

log = logging.getLogger(__name__)


def run(
    corpus_dir: Path,
    qdrant_url: str = "http://localhost:6333",
    qdrant_path: Optional[str] = None,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    embed_batch_size: int = 32,
    dry_run: bool = False,
) -> dict:
    """
    Ingest all PDFs in corpus_dir into Qdrant.

    Returns a summary dict with counts and any per-file errors.
    """
    pdf_paths = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found in {corpus_dir}")

    print(f"Found {len(pdf_paths)} PDFs in {corpus_dir}")
    if dry_run:
        print("Dry run: will parse and chunk but not embed or store.")

    config = ChunkingConfig()
    embedder = Embedder(model_name=embed_model) if not dry_run else None
    store: QdrantStore | None = None

    if not dry_run:
        store = QdrantStore(url=qdrant_url, path=qdrant_path)
        store.ensure_collection(vector_size=embedder.vector_size)

    total_chunks = 0
    errors: dict[str, str] = {}

    for pdf_path in tqdm(pdf_paths, desc="Ingesting"):
        try:
            doc = parse_pdf(pdf_path)
            chunks = chunk_document(doc, config)

            if not chunks:
                log.warning("No chunks produced for %s", pdf_path.name)
                continue

            if dry_run:
                token_counts = [c.token_count for c in chunks]
                tqdm.write(
                    f"  {pdf_path.name}: {len(chunks)} chunks, "
                    f"avg {sum(token_counts)//len(token_counts)} tokens"
                )
                total_chunks += len(chunks)
            else:
                texts = [c.text for c in chunks]
                embeddings = embedder.embed_passages(texts, batch_size=embed_batch_size)
                n = store.upsert(chunks, embeddings)
                total_chunks += n

        except Exception as exc:
            log.error("Failed to process %s: %s", pdf_path.name, exc)
            errors[pdf_path.name] = str(exc)

    summary = {
        "pdfs_processed": len(pdf_paths) - len(errors),
        "pdfs_failed": len(errors),
        "total_chunks": total_chunks,
        "errors": errors,
    }

    print(f"\nDone. {total_chunks} chunks from {summary['pdfs_processed']} PDFs.")
    if errors:
        print(f"  {len(errors)} failures:")
        for name, msg in errors.items():
            print(f"    {name}: {msg}")
    if store:
        print(f"  Collection size: {store.count()} points")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Directory of PDF manuals to ingest")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-path", default=None,
                        help="Local Qdrant storage path (overrides --qdrant-url, no server needed)")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and chunk only; skip embedding and storage")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run(
        corpus_dir=args.corpus,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        embed_model=args.embed_model,
        embed_batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
