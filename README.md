# Local RAG with Ollama (Docker)

This is a minimal RAG pipeline that:
- reads Markdown docs from `christmas_tree_specs/`
- chunks them
- creates embeddings via Ollama (`/api/embeddings`)
- stores them in a local FAISS index (with metadata in JSON)
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
python -m rag_ollama.ingest --docs christmas_tree_specs --db rag --clear
```

CPU preset:
```bash
python -m rag_ollama.ingest --preset cpu --docs christmas_tree_specs --db rag --clear
```

Optional flags:
- `--embed-model nomic-embed-text`
- `--base-url http://localhost:11434`
- `--max-chars 1200 --overlap-chars 200`
- `--workers 4` (number of parallel processes for ingestion)

## 5) Ask questions

### Option A: Command Line

Default preset:
```bash
python -m rag_ollama.query "What is the height range of Christmas Tree Type 1?" --db rag --show-sources
```

CPU preset:
```bash
python -m rag_ollama.query --preset cpu "What is the height range of Christmas Tree Type 1?" --db rag --show-sources
```

Optional flags:
- `--llm-model llama3.2:1b`
- `--top-k 5`

### Option B: REST API

Start the API server:
```bash
python api.py
```

The server will start on `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

**API Endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `POST /ask` - Ask a question

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the height range of Christmas Tree Type 1?",
    "top_k": 5,
    "show_sources": true
  }'
```

**Example using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What is the height range of Christmas Tree Type 1?",
        "top_k": 5,
        "show_sources": True
    }
)
print(response.json())
```

**Configuration via environment variables:**
```bash
RAG_DB_PATH=rag \
OLLAMA_BASE_URL=http://localhost:11434 \
RAG_PRESET=cpu \
PORT=8000 \
python api.py
```

## Environment variables (optional)

**General:**
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text` or your chosen preset override)
- `OLLAMA_LLM_MODEL` (default: preset-dependent; CPU preset uses `llama3.2:1b`)
- `RAG_PRESET` (default: `default`, set to `cpu` for CPU-friendly defaults)

**API Server specific:**
- `RAG_DB_PATH` (default: `rag`) - Base path for FAISS index and metadata files
- `HOST` (default: `0.0.0.0`) - API server host
- `PORT` (default: `8000`) - API server port
