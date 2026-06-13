"""Retrieval orchestration."""

from rag.documents.models import Chunk
from rag.embeddings.base import EmbeddingModel
from rag.indexes.base import VectorStore
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.models import SearchResult
from rag.retrieval.rerankers import Reranker


class Retriever:
    """Indexes chunks and retrieves relevant context."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        rerank_candidate_multiplier: int = 4,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._reranker = reranker
        self._rerank_candidate_multiplier = max(1, rerank_candidate_multiplier)

    def index(self, chunks: list[Chunk]) -> None:
        embeddings = self._embedding_model.embed([chunk.text for chunk in chunks])
        self._vector_store.add(chunks, embeddings)

    def retrieve(
        self,
        query: str,
        *,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        query_embedding = self._embedding_model.embed([query])[0]
        candidate_limit = limit
        if self._reranker is not None:
            candidate_limit = max(limit, limit * self._rerank_candidate_multiplier)

        results = self._vector_store.search(
            query_embedding,
            limit=candidate_limit,
            metadata_filter=metadata_filter,
        )
        if self._reranker is None:
            return results
        return self._reranker.rerank(query, results, limit=limit)

    def clear(self) -> None:
        self._vector_store.clear()
