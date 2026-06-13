"""Validated answer generation data models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.documents.models import Metadata
from rag.retrieval.models import SearchResult


class Citation(BaseModel):
    """Source information attached to a generated answer."""

    model_config = ConfigDict(frozen=True)

    source: str
    document_id: str
    chunk_index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    metadata: Metadata = Field(default_factory=dict)


class Message(BaseModel):
    """A chat message stored in conversation history."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Message content must not be empty."
            raise ValueError(msg)
        return value


class Answer(BaseModel):
    """Generated answer with retrieval context and citations."""

    model_config = ConfigDict(frozen=True)

    text: str
    question: str
    citations: list[Citation]
    results: list[SearchResult]

    @field_validator("text", "question")
    @classmethod
    def text_fields_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Answer text and question must not be empty."
            raise ValueError(msg)
        return value
