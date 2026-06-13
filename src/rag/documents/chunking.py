"""Document chunking primitives."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.documents.models import Chunk, Document
from rag.exceptions import EmptyDocumentError


class ChunkingConfig(BaseModel):
    """Configuration for deterministic text chunking."""

    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=1_000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size."
            raise ValueError(msg)
        return self


class TextChunker:
    """Split documents into readable, overlapping text chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text.strip():
            raise EmptyDocumentError("Cannot chunk an empty document.")

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = self._choose_end_boundary(text, start)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_id=document.id,
                        index=index,
                        start_char=start,
                        end_char=end,
                        metadata=document.metadata,
                    )
                )
                index += 1

            if end >= len(text):
                break

            next_start = max(end - self.config.chunk_overlap, start + 1)
            start = self._skip_leading_whitespace(text, next_start)

        return chunks

    def chunk_many(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks

    def _choose_end_boundary(self, text: str, start: int) -> int:
        hard_end = min(start + self.config.chunk_size, len(text))
        if hard_end >= len(text):
            return len(text)

        minimum_boundary = start + self.config.chunk_size // 2
        boundaries = (
            text.rfind("\n\n", start, hard_end),
            text.rfind("\n", start, hard_end),
            text.rfind(". ", start, hard_end),
            text.rfind(" ", start, hard_end),
        )

        for boundary in boundaries:
            if boundary >= minimum_boundary:
                return boundary + 1

        return hard_end

    @staticmethod
    def _skip_leading_whitespace(text: str, start: int) -> int:
        while start < len(text) and text[start].isspace():
            start += 1
        return start
