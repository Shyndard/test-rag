# Local RAG with Ollama (Docker)

This is a minimal RAG pipeline that:
- reads Markdown docs from `christmas_tree_specs/`
- chunks them
- creates embeddings via Ollama (`/api/embeddings`)
- stores them in a local SQLite DB
- retrieves top-k chunks and calls an LLM via Ollama (`/api/generate`)

## 1) Start Ollama in Docker

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Check it:
```bash
curl http://localhost:11434/api/tags
```

## 2) Pull models in Ollama

You need:
- an embedding model (default: `nomic-embed-text`)
- an LLM for generation (default preset: `llama3.2:3b`)

### CPU-only servers (recommended)
Use a smaller LLM for much better latency:
- LLM: `llama3.2:1b`
- Embeddings: `nomic-embed-text`

```bash
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull llama3.2:1b
```

### Default (heavier)
```bash
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull llama3.2:3b
```

## 3) Install Python deps

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Ingest docs → build vector DB

Default preset:
```bash
python -m rag_ollama.ingest --docs christmas_tree_specs --db rag.sqlite --clear
```

CPU preset:
```bash
python -m rag_ollama.ingest --preset cpu --docs christmas_tree_specs --db rag.sqlite --clear
```

Optional flags:
- `--embed-model nomic-embed-text`
- `--base-url http://localhost:11434`
- `--max-chars 1200 --overlap-chars 200`

## 5) Ask questions

Default preset:
```bash
python -m rag_ollama.query "What is the height range of Christmas Tree Type 1?" --db rag.sqlite --show-sources
```

CPU preset:
```bash
python -m rag_ollama.query --preset cpu "What is the height range of Christmas Tree Type 1?" --db rag.sqlite --show-sources
```

Optional flags:
- `--llm-model llama3.2:1b`
- `--top-k 5`

## Environment variables (optional)

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text` or your chosen preset override)
- `OLLAMA_LLM_MODEL` (default: preset-dependent; CPU preset uses `llama3.2:1b`)
- `RAG_PRESET` (default: `default`, set to `cpu` for CPU-friendly defaults)
