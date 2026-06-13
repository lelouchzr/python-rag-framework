"""Small protocols that define framework extension points."""

from rag.embeddings.base import EmbeddingModel
from rag.generation.llms import LLM, TokenStream
from rag.indexes.base import VectorStore

__all__ = ["EmbeddingModel", "LLM", "TokenStream", "VectorStore"]
