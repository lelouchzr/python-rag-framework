# Quick Start

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

Use a callable when you want answer generation without writing an adapter class:

```python
from collections.abc import Sequence

from rag import CallableLLM, Message, RAG


def generate(prompt: str, history: Sequence[Message]) -> str:
    return call_your_model(prompt, history)


rag = RAG(llm=CallableLLM(generate))
answer = rag.ask("Explain the retrieved context.")
```
