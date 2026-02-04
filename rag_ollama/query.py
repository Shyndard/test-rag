from __future__ import annotations

import argparse
import os

import numpy as np

from rag_ollama.ollama_client import OllamaClient, OllamaError
from rag_ollama.vector_store import VectorStore


SYSTEM_PROMPT = """You are a helpful assistant.
Use ONLY the provided context to answer.
If the context does not contain the answer, say you don't know.
When relevant, cite which source file(s) you used.
"""


def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(contexts)
    return f"""Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions over an indexed markdown folder (RAG) using Ollama.")
    parser.add_argument("question", help="Your question")
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
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Ollama chat/generate model (overrides preset/env OLLAMA_LLM_MODEL)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-sources", action="store_true")

    args = parser.parse_args()

    # Resolve models with precedence: CLI arg > env var > preset default
    preset_embed = "nomic-embed-text"
    preset_llm = "llama3.2:3b"
    if args.preset == "cpu":
        preset_llm = "llama3.2:1b"

    embed_model = args.embed_model or os.environ.get("OLLAMA_EMBED_MODEL") or preset_embed
    llm_model = args.llm_model or os.environ.get("OLLAMA_LLM_MODEL") or preset_llm

    store = VectorStore(args.db)
    if store.count() == 0:
        raise SystemExit(f"No chunks found in {args.db}. Run ingest first.")

    client = OllamaClient(base_url=args.base_url)
    try:
        client.tags()
    except OllamaError as e:
        raise SystemExit(str(e))

    q_emb = client.embeddings(model=embed_model, prompt=args.question)
    q_vec = np.array(q_emb, dtype=np.float32)

    results = store.search(query_embedding=q_vec, top_k=args.top_k)
    if not results:
        raise SystemExit("No retrieval results.")

    contexts: list[str] = []
    used_sources: list[str] = []
    for r in results:
        if args.show_sources:
            contexts.append(f"Source: {r.source}\nScore: {r.score:.4f}\n---\n{r.content}")
        else:
            contexts.append(r.content)
        used_sources.append(r.source)

    prompt = build_prompt(args.question, contexts)
    answer = client.generate(
        model=llm_model,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        options={"temperature": 0.2},
    ).strip()

    print(answer)

    if args.show_sources:
        uniq = sorted(set(used_sources))
        print("\nSources:")
        for s in uniq:
            print(f"- {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
