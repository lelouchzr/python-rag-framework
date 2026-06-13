"""Loader interfaces."""

from pathlib import Path
from typing import Protocol

from rag.documents.models import Document


class DocumentLoader(Protocol):
    """Loads one source file into a `Document`."""

    supported_suffixes: frozenset[str]

    def load(self, path: str | Path) -> Document:
        """Load a document from disk."""
