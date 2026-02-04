#!/usr/bin/env python3
"""REST API for the RAG question-answering system."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_ollama.ollama_client import OllamaClient, OllamaError
from rag_ollama.query import SYSTEM_PROMPT, build_prompt
from rag_ollama.vector_store import VectorStore

app = FastAPI(
    title="RAG Question Answering API",
    description="Ask questions over indexed markdown documents using Ollama",
    version="1.0.0",
)

# Configuration
CONFIG = {
    "db_path": os.environ.get("RAG_DB_PATH", "rag"),
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "preset": os.environ.get("RAG_PRESET", "default"),
    "embed_model": os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    "llm_model": os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:3b"),
}


class QuestionRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., description="The question to ask", min_length=1)
    top_k: int = Field(5, description="Number of top documents to retrieve", ge=1, le=20)
    show_sources: bool = Field(False, description="Include source information in the response")
    embed_model: Optional[str] = Field(None, description="Override embedding model")
    llm_model: Optional[str] = Field(None, description="Override LLM model")


class Source(BaseModel):
    """Source document information."""

    source: str
    score: float
    content: str


class QuestionResponse(BaseModel):
    """Response model for a question."""

    answer: str
    sources: Optional[List[Source]] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Question Answering API",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Check API health",
            "/docs": "Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database
        store = VectorStore(CONFIG["db_path"])
        chunk_count = store.count()
        
        if chunk_count == 0:
            return {
                "status": "warning",
                "message": "Database is empty. Run ingest first.",
                "database": CONFIG["db_path"],
                "chunks": 0,
            }
        
        # Check Ollama connection
        client = OllamaClient(base_url=CONFIG["base_url"])
        client.tags()
        
        return {
            "status": "healthy",
            "database": CONFIG["db_path"],
            "chunks": chunk_count,
            "ollama_url": CONFIG["base_url"],
        }
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on indexed documents.
    
    The system will:
    1. Embed the question using the specified embedding model
    2. Search for relevant document chunks in the vector store
    3. Generate an answer using the LLM with the retrieved context
    """
    try:
        # Initialize components
        store = VectorStore(CONFIG["db_path"])
        if store.count() == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No chunks found in {CONFIG['db_path']}. Run ingest first.",
            )
        
        # Determine models
        embed_model = request.embed_model or CONFIG["embed_model"]
        llm_model = request.llm_model or CONFIG["llm_model"]
        
        client = OllamaClient(base_url=CONFIG["base_url"])
        
        # Check Ollama connection
        try:
            client.tags()
        except OllamaError as e:
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
        
        # Embed question
        q_emb = client.embeddings(model=embed_model, prompt=request.question)
        q_vec = np.array(q_emb, dtype=np.float32)
        
        # Search for relevant chunks
        results = store.search(query_embedding=q_vec, top_k=request.top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        
        # Build context
        contexts: List[str] = []
        sources: List[Source] = []
        
        for r in results:
            if request.show_sources:
                contexts.append(f"Source: {r.source}\nScore: {r.score:.4f}\n---\n{r.content}")
                sources.append(Source(source=r.source, score=r.score, content=r.content))
            else:
                contexts.append(r.content)
        
        # Generate answer
        prompt = build_prompt(request.question, contexts)
        answer = client.generate(
            model=llm_model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            options={"temperature": 0.2},
        ).strip()
        
        response = QuestionResponse(answer=answer)
        if request.show_sources:
            response.sources = sources
        
        return response
        
    except HTTPException:
        raise
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting RAG API server on {host}:{port}")
    print(f"Database: {CONFIG['db_path']}")
    print(f"Ollama: {CONFIG['base_url']}")
    print(f"Interactive docs: http://localhost:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
