from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from rag_ollama.chunking import chunk_markdown
from rag_ollama.ollama_client import OllamaClient, OllamaError
from rag_ollama.vector_store import VectorStore


def iter_markdown_files(folder: Path) -> list[Path]:
    files = sorted([p for p in folder.rglob("*.md") if p.is_file()])
    return files


def process_file(args) -> tuple[str, list[tuple[str, int, str, np.ndarray]]]:
    path, embed_model, base_url, max_chars, overlap_chars, docs_dir = args
    client = OllamaClient(base_url=base_url)
    text = path.read_text(encoding="utf-8")
    chunks = chunk_markdown(
        text,
        source=str(path.relative_to(docs_dir)),
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    chunk_data = []
    for ch in chunks:
        emb = client.embeddings(model=embed_model, prompt=ch.content)
        vec = np.array(emb, dtype=np.float32)
        chunk_data.append((ch.source, ch.chunk_index, ch.content, vec))
    return str(path), chunk_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest markdown files into a local FAISS vector store using Ollama embeddings.")
    parser.add_argument("--docs", default="doc", help="Folder containing .md docs")
    parser.add_argument("--db", default="rag.sqlite", help="FAISS index path (will create .faiss and .json files)")
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
    parser.add_argument("--clear", action="store_true", help="Clear index before ingest")
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 4), help="Number of parallel workers")

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

    # Prepare args for parallel processing
    process_args = [
        (path, embed_model, args.base_url, args.max_chars, args.overlap_chars, docs_dir)
        for path in md_files
    ]

    total_chunks = 0
    with Pool(args.workers) as pool:
        results = pool.map(process_file, process_args)
        for file_path, chunk_data in results:
            for source, chunk_index, content, vec in chunk_data:
                store.add(source=source, chunk_index=chunk_index, content=content, embedding=vec)
                total_chunks += 1

    store._save()  # Save once at the end
    print(f"Ingested {len(md_files)} files → {total_chunks} chunks into {args.db} (total vectors: {store.count()}).")
    print(f"Embedding model: {embed_model} | Ollama: {args.base_url} | Workers: {args.workers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
