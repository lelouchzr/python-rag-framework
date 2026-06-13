"""Public API for the Python RAG framework."""

from rag._version import __version__
from rag.chunking import ChunkingConfig, TextChunker
from rag.client import RAG
from rag.embeddings import HashingEmbeddingModel, NgramHashingEmbeddingModel
from rag.llms import AnthropicLLM, CallableLLM, OpenAILLM
from rag.models import Answer, Chunk, Citation, Document, Message, SearchResult
from rag.prompting import PromptBuilder
from rag.retrieval import LexicalReranker
from rag.vector_store import InMemoryVectorStore, SQLiteVectorStore

__all__ = [
    "Answer",
    "AnthropicLLM",
    "CallableLLM",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "Document",
    "HashingEmbeddingModel",
    "InMemoryVectorStore",
    "Message",
    "LexicalReranker",
    "NgramHashingEmbeddingModel",
    "OpenAILLM",
    "PromptBuilder",
    "RAG",
    "SearchResult",
    "SQLiteVectorStore",
    "TextChunker",
    "__version__",
]
