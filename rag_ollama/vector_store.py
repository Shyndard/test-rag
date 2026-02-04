from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

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
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")

    def add(self, *, source: str, chunk_index: int, content: str, embedding: np.ndarray) -> None:
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        blob = embedding.tobytes()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chunks(source, chunk_index, content, embedding, embedding_dim) VALUES (?, ?, ?, ?, ?)",
                (source, chunk_index, content, blob, int(embedding.shape[0])),
            )

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(1) FROM chunks").fetchone()
        return int(n)

    def sources(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT source FROM chunks ORDER BY source").fetchall()
        return [r[0] for r in rows]

    def search(
        self,
        *,
        query_embedding: np.ndarray,
        top_k: int = 5,
        source_filter: Optional[Iterable[str]] = None,
    ) -> list[StoredChunk]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        where = ""
        params: list[object] = []
        if source_filter:
            srcs = list(source_filter)
            if srcs:
                placeholders = ",".join(["?"] * len(srcs))
                where = f"WHERE source IN ({placeholders})"
                params.extend(srcs)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, source, chunk_index, content, embedding, embedding_dim FROM chunks {where}",
                params,
            ).fetchall()

        if not rows:
            return []

        mat: list[np.ndarray] = []
        meta: list[tuple[int, str, int, str]] = []
        for chunk_id, source, chunk_index, content, emb_blob, emb_dim in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32, count=int(emb_dim))
            mat.append(vec)
            meta.append((int(chunk_id), str(source), int(chunk_index), str(content)))

        A = np.vstack(mat)
        q = query_embedding.reshape(1, -1)

        # cosine similarity
        A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        sims = (A @ q.T) / (A_norm * q_norm.T)
        sims = sims.reshape(-1)

        k = min(top_k, sims.shape[0])
        idxs = np.argpartition(-sims, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]

        out: list[StoredChunk] = []
        for i in idxs:
            chunk_id, source, chunk_index, content = meta[int(i)]
            out.append(
                StoredChunk(
                    id=chunk_id,
                    source=source,
                    chunk_index=chunk_index,
                    content=content,
                    score=float(sims[int(i)]),
                )
            )
        return out
