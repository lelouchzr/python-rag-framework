# Public API

The main entry point is `rag.RAG`.

## Version

```python
import rag

print(rag.__version__)
```

## Ingestion

```python
from rag import RAG

rag = RAG()
rag.add_text("Inline source text", source="notes")
rag.add_txt("notes.txt")
rag.add_markdown("notes.md")
rag.add_pdf("paper.pdf")
rag.add_docx("handout.docx")
rag.add_image("scan.png")  # requires the `ocr` extra
```

Each ingestion method returns the chunks created from that source. The full
corpus is available through `rag.documents` and `rag.chunks`.

## Retrieval

```python
results = rag.retrieve(
    "Korean topic particles",
    limit=4,
    metadata_filter={"language": "en"},
)
```

Results are returned as `SearchResult` models containing a `Chunk` and a
similarity score.

Metadata filters also support simple operators:

```python
results = rag.retrieve(
    "advanced grammar",
    metadata_filter={
        "year": {"$gte": 2020},
        "language": {"$in": ["en", "fr"]},
        "title": {"$contains": "grammar"},
    },
)
```

Supported operators are `$eq`, `$ne`, `$in`, `$nin`, `$contains`, `$exists`,
`$gt`, `$gte`, `$lt`, and `$lte`.

## Answer Generation

`ask()` and `stream()` require an LLM. Use `CallableLLM` for small integrations
or pass any object implementing the `LLM` protocol. Provider adapters are
available as optional integrations.

```python
from collections.abc import Sequence

from rag import CallableLLM, Message, RAG


def generate(prompt: str, history: Sequence[Message]) -> str:
    return call_my_model(prompt, history)


rag = RAG(llm=CallableLLM(generate))
answer = rag.ask("What does this source say?")
```

```python
from rag import OpenAILLM, RAG

rag = RAG(llm=OpenAILLM(model="gpt-4.1-mini"))
```

`answer` contains:

- `text`: generated answer
- `citations`: retrieved sources used for the prompt
- `results`: retrieved chunks and scores
- `question`: original question

Streaming yields text parts and updates conversation history when the stream is
fully consumed:

```python
for part in rag.stream("Summarize the document"):
    print(part, end="")
```

## Custom Components

`RAG` accepts replacement components:

```python
rag = RAG(
    chunker=my_chunker,
    file_loader=my_loader,
    embedding_model=my_embedding_model,
    vector_store=my_vector_store,
    reranker=my_reranker,
    prompt_builder=my_prompt_builder,
    llm=my_llm,
)
```

Built-in alternatives include `NgramHashingEmbeddingModel`,
`SQLiteVectorStore`, and `LexicalReranker`.

## Import Stability

The preferred import for application code is the root package:

```python
from rag import RAG, CallableLLM
```

Advanced users can import from domain packages such as `rag.documents`,
`rag.embeddings`, `rag.indexes`, `rag.retrieval`, and `rag.generation`.

Existing flat-module imports remain supported for compatibility:

```python
from rag.rag import RAG
from rag.models import Document
from rag.vector_store import InMemoryVectorStore
```

## CLI

The package installs a small `rag` command:

```bash
rag version
rag inspect notes.md
rag chunk notes.md --json
rag search notes.md "topic particles" --limit 3
```

The CLI uses the same loaders, chunker, embedding model, and retriever as the
Python API.
