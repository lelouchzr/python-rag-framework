# Installation

Install the core package:

```bash
uv add python-rag-framework
```

For local development:

```bash
uv sync --extra dev
```

Optional extras:

```bash
uv add "python-rag-framework[pdf]"
uv add "python-rag-framework[docx]"
uv add "python-rag-framework[ocr]"
uv add "python-rag-framework[openai]"
uv add "python-rag-framework[anthropic]"
```

The core install intentionally stays small. PDF, DOCX, OCR, and provider SDKs
are loaded only when their integrations are used.
