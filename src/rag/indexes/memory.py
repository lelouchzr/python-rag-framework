"""In-memory vector store implementation."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from rag.documents.models import Chunk
from rag.exceptions import VectorStoreError
from rag.retrieval.filters import MetadataFilter, matches_metadata
from rag.retrieval.models import SearchResult


@dataclass(frozen=True)
class _VectorRecord:
    chunk: Chunk
    embedding: tuple[float, ...]


class InMemoryVectorStore:
    """A small cosine-similarity vector store for local RAG applications."""

    def __init__(self) -> None:
        self._records: list[_VectorRecord] = []

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            msg = "chunks and embeddings must have the same length."
            raise VectorStoreError(msg)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if not embedding:
                raise VectorStoreError("Embedding vectors must not be empty.")
            self._records.append(_VectorRecord(chunk=chunk, embedding=tuple(embedding)))

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

        query = tuple(query_embedding)
        scored: list[SearchResult] = []

        for record in self._records:
            if metadata_filter and not matches_metadata(
                record.chunk.metadata,
                metadata_filter,
            ):
                continue

            score = _cosine_similarity(query, record.embedding)
            if score <= 0.0:
                continue

            scored.append(SearchResult(chunk=record.chunk, score=score))

        return sorted(scored, key=lambda result: result.score, reverse=True)[:limit]

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        msg = "Embedding dimensions must match."
        raise VectorStoreError(msg)

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    raw_score = dot_product / (left_norm * right_norm)
    return max(0.0, min(1.0, raw_score))
