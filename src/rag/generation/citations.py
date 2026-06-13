"""Citation construction."""

from rag.generation.models import Citation
from rag.retrieval.models import SearchResult


class CitationBuilder:
    """Build citations from retrieved search results."""

    def build(self, results: list[SearchResult]) -> list[Citation]:
        citations: list[Citation] = []
        for result in results:
            source = str(
                result.chunk.metadata.get("source")
                or result.chunk.metadata.get("filename")
                or result.chunk.document_id
            )
            citations.append(
                Citation(
                    source=source,
                    document_id=result.chunk.document_id,
                    chunk_index=result.chunk.index,
                    score=result.score,
                    metadata=result.chunk.metadata,
                )
            )
        return citations
