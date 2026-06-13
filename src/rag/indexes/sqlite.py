"""SQLite-backed vector store implementation."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from rag.documents.models import Chunk
from rag.exceptions import VectorStoreError
from rag.indexes.memory import _cosine_similarity
from rag.retrieval.filters import MetadataFilter, matches_metadata
from rag.retrieval.models import SearchResult


class SQLiteVectorStore:
    """Persistent cosine-similarity vector store backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            msg = "chunks and embeddings must have the same length."
            raise VectorStoreError(msg)

        records = [
            (chunk.model_dump_json(), json.dumps(list(embedding)))
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._connection.executemany(
            "INSERT INTO vectors (chunk_json, embedding_json) VALUES (?, ?)",
            records,
        )
        self._connection.commit()

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be greater than zero.")
        if not query_embedding:
            raise VectorStoreError("Query embedding must not be empty.")

        results: list[SearchResult] = []
        query = tuple(query_embedding)

        for chunk_json, embedding_json in self._connection.execute(
            "SELECT chunk_json, embedding_json FROM vectors"
        ):
            chunk = Chunk.model_validate_json(chunk_json)
            if metadata_filter and not matches_metadata(
                chunk.metadata,
                metadata_filter,
            ):
                continue

            embedding = tuple(float(value) for value in json.loads(embedding_json))
            score = _cosine_similarity(query, embedding)
            if score <= 0.0:
                continue
            results.append(SearchResult(chunk=chunk, score=score))

        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def clear(self) -> None:
        self._connection.execute("DELETE FROM vectors")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __len__(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) FROM vectors")
        return int(cursor.fetchone()[0])

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
