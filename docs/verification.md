# Verification and Packaging

The project follows Semantic Versioning:

- `MAJOR`: breaking public API changes
- `MINOR`: backward-compatible features
- `PATCH`: backward-compatible fixes

Before a release:

```bash
uv run ruff check .
uv run mypy
uv run pytest
uv build
uv run mkdocs build --strict
```

Update:

- `CHANGELOG.md`
- `pyproject.toml` version
- `src/rag/_version.py`

The checked-in code is installed from Git. Run the checks above before
building distribution artifacts locally.
