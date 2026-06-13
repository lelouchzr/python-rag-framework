# Benchmarks

The repository includes a lightweight benchmark script in `benchmarks/`.

```bash
uv run python benchmarks/benchmark_local.py --documents 200 --queries 50
```

It measures:

- indexing time
- retrieval latency
- embedding time
- peak memory usage

The benchmark is intentionally simple. Its purpose is to make future changes
measurable, not to claim broad performance superiority over other frameworks.
