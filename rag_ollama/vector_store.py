from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import faiss
import numpy as np


@dataclass(frozen=True)
class StoredChunk:
    id: int
    source: str
    chunk_index: int
    content: str
    score: float


class VectorStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.index_path = f"{db_path}.faiss"
        self.meta_path = f"{db_path}.json"
        self._load_or_init()

    def _load_or_init(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = None
            self.metadata = []

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def clear(self) -> None:
        self.index = None
        self.metadata = []
        self._save()

    def add(self, *, source: str, chunk_index: int, content: str, embedding: np.ndarray) -> None:
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if self.index is None:
            dim = embedding.shape[0]
            self.index = faiss.IndexFlatIP(dim)
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding) + 1e-12
        embedding /= norm
        self.index.add(embedding.reshape(1, -1))
        chunk_id = len(self.metadata)
        self.metadata.append({
            "id": chunk_id,
            "source": source,
            "chunk_index": chunk_index,
            "content": content,
        })
        # No _save here; call _save() explicitly at the end of ingestion

    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def sources(self) -> list[str]:
        return sorted(set(meta["source"] for meta in self.metadata))

    def search(
        self,
        *,
        query_embedding: np.ndarray,
        top_k: int = 5,
        source_filter: Optional[Iterable[str]] = None,
    ) -> list[StoredChunk]:
        if self.index is None or self.index.ntotal == 0:
            return []
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        # Normalize query
        norm = np.linalg.norm(query_embedding) + 1e-12
        query_embedding /= norm
        query = query_embedding.reshape(1, -1)

        # Search index
        scores, indices = self.index.search(query, min(top_k * 2, self.index.ntotal))  # Search more for filtering
        scores = scores[0]
        indices = indices[0]

        # Filter by source if needed
        candidates = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if source_filter and meta["source"] not in source_filter:
                continue
            candidates.append((meta, scores[i]))

        # Sort by score descending and take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        out = []
        for meta, score in candidates[:top_k]:
            out.append(
                StoredChunk(
                    id=meta["id"],
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    content=meta["content"],
                    score=float(score),
                )
            )
        return out
