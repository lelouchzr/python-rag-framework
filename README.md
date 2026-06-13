# Python RAG Framework

A lightweight Python framework for building retrieval-augmented generation
applications with clean architecture, strong typing, and minimal dependencies.

The project is intentionally focused. It does not try to be another LangChain
clone; it provides a small set of composable primitives that are easy to read,
test, and extend.

## Features

- TXT and Markdown loading in the core package
- optional PDF, DOCX, and OCR image loading
- deterministic chunking
- local hashing and n-gram hashing embeddings
- in-memory and SQLite vector stores
- exact and operator-based metadata filtering
- optional lexical reranking
- prompt construction
- pluggable LLM interface
- callable, OpenAI, and Anthropic LLM adapters
- streaming responses
- source citations
- conversation history
- small CLI for local inspection, chunking, and retrieval

## Installation

Core install:

```bash
uv add python-rag-framework
```

Development install:

```bash
uv sync --extra dev
```

Optional integrations:

```bash
uv add "python-rag-framework[pdf]"
uv add "python-rag-framework[docx]"
uv add "python-rag-framework[ocr]"
uv add "python-rag-framework[openai]"
uv add "python-rag-framework[anthropic]"
```

## Quick Start

```python
from rag import RAG

rag = RAG()
rag.add_text(
    "In Korean, 은/는 are topic particles and often introduce contrast.",
    source="note",
)

results = rag.retrieve("What do topic particles mark?", limit=2)

for result in results:
    print(result.score)
    print(result.chunk.text)
```

Use a callable for answer generation:

```python
from collections.abc import Sequence

from rag import CallableLLM, Message, RAG


def generate(prompt: str, history: Sequence[Message]) -> str:
    return call_your_model(prompt, history)


rag = RAG(llm=CallableLLM(generate))
rag.add_markdown("notes.md")
answer = rag.ask("Explain the difference between 은/는 and 이/가.")
print(answer.text)
```

## CLI

```bash
rag inspect examples/sample_notes.md
rag chunk examples/sample_notes.md --json
rag search examples/sample_notes.md "topic particles"
```

The CLI is intentionally local-only. It is useful for inspecting loaders,
chunking behavior, and retrieval results without configuring an LLM provider.

## Examples

Runnable examples live in [`examples/`](examples/):

- `local_search.py`
- `callable_llm.py`
- `persistent_sqlite_index.py`
- `openai_adapter.py`

## Quality

```bash
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

The repository also includes GitHub Actions for linting, type checking, tests,
package builds, docs builds, and optional PyPI publishing.

## Documentation

- [Installation](docs/installation.md)
- [Quick Start](docs/quick-start.md)
- [Examples](docs/examples.md)
- [API Reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Contributing](docs/contributing.md)
- [Roadmap](docs/roadmap.md)

## Status

The framework is feature-complete for its current scope and suitable as a
portfolio-quality open-source Python library. Future work should focus on
release hardening, examples, benchmarks, and real-world feedback rather than
feature sprawl.
