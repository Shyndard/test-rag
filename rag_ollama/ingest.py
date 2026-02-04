from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from rag_ollama.chunking import chunk_markdown
from rag_ollama.ollama_client import OllamaClient, OllamaError
from rag_ollama.vector_store import VectorStore


def iter_markdown_files(folder: Path) -> list[Path]:
    files = sorted([p for p in folder.rglob("*.md") if p.is_file()])
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest markdown files into a local SQLite vector store using Ollama embeddings.")
    parser.add_argument("--docs", default="christmas_tree_specs", help="Folder containing .md docs")
    parser.add_argument("--db", default="rag.sqlite", help="SQLite DB path")
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument(
        "--preset",
        choices=["default", "cpu"],
        default=os.environ.get("RAG_PRESET", "default"),
        help="Model preset. 'cpu' prefers smaller/faster models on CPU-only machines.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Ollama embedding model (overrides preset/env OLLAMA_EMBED_MODEL)",
    )
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--clear", action="store_true", help="Clear DB before ingest")

    args = parser.parse_args()

    preset_embed = "nomic-embed-text"
    embed_model = args.embed_model or os.environ.get("OLLAMA_EMBED_MODEL") or preset_embed

    docs_dir = Path(args.docs).expanduser().resolve()
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise SystemExit(f"Docs folder not found: {docs_dir}")

    client = OllamaClient(base_url=args.base_url)
    store = VectorStore(args.db)

    if args.clear:
        store.clear()

    md_files = iter_markdown_files(docs_dir)
    if not md_files:
        raise SystemExit(f"No .md files found under: {docs_dir}")

    try:
        client.tags()
    except OllamaError as e:
        raise SystemExit(str(e))

    total_chunks = 0
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            source=str(path.relative_to(docs_dir)),
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )

        for ch in chunks:
            emb = client.embeddings(model=embed_model, prompt=ch.content)
            vec = np.array(emb, dtype=np.float32)
            store.add(source=ch.source, chunk_index=ch.chunk_index, content=ch.content, embedding=vec)
            total_chunks += 1

    print(f"Ingested {len(md_files)} files → {total_chunks} chunks into {args.db} (total rows: {store.count()}).")
    print(f"Embedding model: {embed_model} | Ollama: {args.base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
