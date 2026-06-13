"""Custom exceptions raised by the framework."""


class RAGError(Exception):
    """Base class for framework errors."""


class EmptyDocumentError(RAGError):
    """Raised when a document has no usable text."""


class DocumentLoadError(RAGError):
    """Raised when a document cannot be loaded from disk."""


class UnsupportedFileTypeError(RAGError):
    """Raised when no loader exists for the requested file type."""


class LLMNotConfiguredError(RAGError):
    """Raised when answer generation is requested without an LLM."""


class VectorStoreError(RAGError):
    """Raised when a vector store operation cannot be completed."""
