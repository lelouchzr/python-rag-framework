"""Embedding interfaces."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    """Converts text into numeric vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""
