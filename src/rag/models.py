"""Backward-compatible model exports.

New code should prefer domain modules:

- `rag.documents.models`
- `rag.retrieval.models`
- `rag.generation.models`
"""

from rag.documents.models import Chunk, Document, Metadata, MetadataValue
from rag.generation.models import Answer, Citation, Message
from rag.retrieval.models import SearchResult

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "Message",
    "Metadata",
    "MetadataValue",
    "SearchResult",
]
