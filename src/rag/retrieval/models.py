"""Validated retrieval data models."""

from pydantic import BaseModel, ConfigDict, Field

from rag.documents.models import Chunk


class SearchResult(BaseModel):
    """A retrieved chunk and its similarity score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)
