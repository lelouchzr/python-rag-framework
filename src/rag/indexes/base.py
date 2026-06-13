"""Vector store interfaces."""

from collections.abc import Sequence
from typing import Protocol

from rag.documents.models import Chunk
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.models import SearchResult


class VectorStore(Protocol):
    """Stores chunk embeddings and performs similarity search."""

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add chunk embeddings to the store."""

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return the most similar chunks."""

    def clear(self) -> None:
        """Remove all indexed vectors."""
