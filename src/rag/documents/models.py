"""Validated document data models."""

from types import MappingProxyType
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataValue = str | int | float | bool | None
Metadata = dict[str, MetadataValue]


class Document(BaseModel):
    """A loaded source document before chunking."""

    model_config = ConfigDict(frozen=True)

    text: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Document text must not be empty."
            raise ValueError(msg)
        return value

    def metadata_view(self) -> MappingProxyType[str, str | int | float | bool | None]:
        return MappingProxyType(self.metadata)


class Chunk(BaseModel):
    """A chunk of text derived from a source document."""

    model_config = ConfigDict(frozen=True)

    text: str
    document_id: str
    index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Chunk text must not be empty."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "Chunk":
        if self.end_char <= self.start_char:
            msg = "end_char must be greater than start_char."
            raise ValueError(msg)
        return self
