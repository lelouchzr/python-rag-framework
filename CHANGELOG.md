# Changelog

All notable changes to this project will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## Retrieval pipeline baseline

### Added

- Public `RAG` facade for ingestion, retrieval, and answer generation.
- TXT, Markdown, PDF, DOCX, and optional OCR image loading.
- Deterministic chunking.
- Local hashing and n-gram hashing embedding models.
- In-memory and SQLite vector stores.
- Exact and operator-based metadata filtering.
- Optional lexical reranking.
- Prompt construction, source citations, streaming, and conversation history.
- Callable, OpenAI, and Anthropic LLM adapters.
- Domain-oriented package structure with backward-compatible flat-module exports.
- Initial documentation, examples, CLI, benchmarks, and CI configuration.
