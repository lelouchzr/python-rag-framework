"""In-memory corpus state."""

from rag.documents.models import Chunk, Document


class Corpus:
    """Stores loaded documents and their chunks."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._chunks: list[Chunk] = []

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(self._documents)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def add(self, document: Document, chunks: list[Chunk]) -> None:
        self._documents.append(document)
        self._chunks.extend(chunks)

    def clear(self) -> None:
        self._documents.clear()
        self._chunks.clear()
