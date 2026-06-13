# Roadmap

The planned roadmap for the lightweight RAG framework is complete. The project
now covers the full local RAG pipeline while keeping provider integrations and
heavier capabilities optional.

## Completed Milestones

### 1. Project Foundation

- Python package configuration with `pyproject.toml`.
- `src/` package layout.
- Ruff, mypy, and pytest configuration.
- Concise documentation in `README.md` and `docs/`.
- Public API tests for the complete local pipeline.

### 2. Architecture Hardening

- Domain package organization:
  - `rag.documents`
  - `rag.embeddings`
  - `rag.indexes`
  - `rag.retrieval`
  - `rag.generation`
- Public `RAG` facade for ingestion, retrieval, and answer generation.
- Extracted responsibilities:
  - `Corpus`
  - `Retriever`
  - `CitationBuilder`
  - `ConversationHistory`
- Backward-compatible flat-module exports:
  - `rag.rag`
  - `rag.models`
  - `rag.loaders`
  - `rag.chunking`
  - `rag.llms`
  - `rag.prompting`
  - `rag.vector_store`
  - `rag.protocols`

### 3. Document Ingestion

- Inline text ingestion with `RAG.add_text()`.
- TXT loading.
- Markdown loading.
- PDF loading.
- DOCX loading.
- Optional OCR image loading with `RAG.add_image()`.
- Format-specific loader modules behind a `FileLoader` dispatcher.

### 4. Chunking And Models

- Deterministic text chunking.
- Configurable chunk size and overlap.
- Pydantic models for:
  - documents
  - chunks
  - search results
  - citations
  - messages
  - answers

### 5. Embeddings And Indexes

- `EmbeddingModel` protocol.
- Local hashing embedding model.
- Local n-gram hashing embedding model.
- `VectorStore` protocol.
- In-memory vector store.
- SQLite-backed persistent vector store.

### 6. Retrieval

- Semantic retrieval over embedded chunks.
- Exact metadata filtering.
- Operator-based metadata filtering:
  - `$eq`
  - `$ne`
  - `$in`
  - `$nin`
  - `$contains`
  - `$exists`
  - `$gt`
  - `$gte`
  - `$lt`
  - `$lte`
- Optional lexical reranking with `LexicalReranker`.

### 7. Generation

- Prompt construction from retrieved context.
- `LLM` protocol.
- `CallableLLM` adapter.
- `RAG.ask()`.
- Streaming responses with `RAG.stream()`.
- Source citations.
- Conversation history.
- Provider adapters:
  - `OpenAILLM`
  - `AnthropicLLM`

### 8. Optional Integrations

- `openai` extra for OpenAI integration.
- `anthropic` extra for Anthropic integration.
- `providers` extra for both provider SDKs.
- `ocr` extra for OCR image loading.

## Current Status

The framework is feature-complete for its intended scope: a modern, lightweight,
well-structured Python RAG library suitable for a portfolio project.

Future work should be driven by concrete use cases rather than feature parity
with larger frameworks. Good candidates would be packaging for PyPI, retrieval
quality benchmarks, examples, or a small demo application.
