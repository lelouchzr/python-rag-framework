# Architecture

This project is intentionally small. It provides a complete local RAG pipeline
while keeping each major capability replaceable through narrow interfaces.

## Package Layout

- `rag.client`: user-facing `RAG` facade.
- `rag.documents`: document models, corpus state, chunking, and loaders.
- `rag.documents.loaders`: format-specific loaders and file dispatch.
- `rag.embeddings`: embedding interfaces and local embedding implementations.
- `rag.indexes`: vector store interfaces, in-memory storage, and SQLite storage.
- `rag.retrieval`: retrieval models and retrieval orchestration.
- `rag.generation`: prompt construction, LLM adapters, citations, and history.
- `rag.exceptions`: framework-specific exceptions.

The older flat modules remain as compatibility exports:

- `rag.rag`
- `rag.models`
- `rag.loaders`
- `rag.chunking`
- `rag.llms`
- `rag.prompting`
- `rag.vector_store`
- `rag.protocols`

## Design Decisions

- `RAG` is a facade. It coordinates ingestion, retrieval, and generation but no
  longer owns every implementation detail directly.
- `Corpus` owns loaded documents and chunks.
- `Retriever` owns indexing and query retrieval.
- `ConversationHistory` owns chat messages.
- `CitationBuilder` owns conversion from search results to citations.
- Format-specific loaders are isolated behind `FileLoader`, making future
  loaders straightforward without growing one monolithic file.
- The default embedding model is local and deterministic. It is useful for
  development and tests, while stronger models can be injected.
- The default vector store is in-memory. SQLite is available when persistence is
  needed without adding an external service.
- `ask()` requires a configured LLM. The framework does not fabricate answers
  when no model is available.
- Provider adapters and OCR dependencies are optional. The core install remains
  small.

## Extension Points

- `DocumentLoader`: add a new source format.
- `EmbeddingModel`: replace hashing embeddings with local or hosted embeddings.
- `VectorStore`: replace the in-memory store with persistent storage.
- `Reranker`: adjust final result ordering after vector search.
- `LLM`: connect OpenAI, Anthropic, local models, or application-specific model
  gateways.
- `PromptBuilder`: customize prompt format and citation instructions.
