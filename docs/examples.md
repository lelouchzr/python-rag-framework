# Examples

Runnable examples live in `examples/`:

- `local_search.py`: local retrieval without an LLM
- `callable_llm.py`: answer generation with a Python callable
- `persistent_sqlite_index.py`: persistent local indexing
- `openai_adapter.py`: optional OpenAI integration

Run an example from the repository root:

```bash
uv run python examples/local_search.py
```

The CLI also supports quick local inspection:

```bash
uv run rag inspect examples/sample_notes.md
uv run rag chunk examples/sample_notes.md --json
uv run rag search examples/sample_notes.md "topic particles"
```
