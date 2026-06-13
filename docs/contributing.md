# Contributing

This repository is built as a modern Python library.

## Local setup

```bash
uv sync --extra dev
```

Run checks before opening a change:

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv build
uv run mkdocs build --strict
```

## Engineering standards

- Keep public APIs small and documented.
- Prefer composition over inheritance.
- Add tests for every public feature.
- Raise explicit framework exceptions for expected user-facing errors.
- Avoid placeholder implementations. It is better to leave a feature out than to
  ship a fake version of it.
- Keep provider-specific integrations behind protocols or small adapters.

## Pull Requests

- Keep changes focused.
- Preserve backward compatibility unless a breaking change is explicitly planned.
- Update tests for public behavior changes.
- Update docs when public APIs, CLI behavior, or packaging changes.

## Release Discipline

The project follows Semantic Versioning:

- Patch releases fix bugs without changing public APIs.
- Minor releases add backward-compatible capabilities.
- Major releases may include breaking public API changes.
