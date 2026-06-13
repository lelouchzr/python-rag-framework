"""Loaders for UTF-8 text-based documents."""

from pathlib import Path

from rag.documents.models import Document
from rag.exceptions import DocumentLoadError, EmptyDocumentError


class TextLoader:
    """Load UTF-8 text-like files into framework documents."""

    supported_suffixes = frozenset({".txt", ".md", ".markdown"})

    def load(self, path: str | Path) -> Document:
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Could not read document at '{file_path}'."
            raise DocumentLoadError(msg) from exc
        except UnicodeDecodeError as exc:
            msg = f"Could not decode '{file_path}' as UTF-8 text."
            raise DocumentLoadError(msg) from exc

        return _document_from_text(text, file_path)


class TxtLoader(TextLoader):
    """Load `.txt` files."""

    supported_suffixes = frozenset({".txt"})


class MarkdownLoader(TextLoader):
    """Load Markdown files as plain text."""

    supported_suffixes = frozenset({".md", ".markdown"})


def _document_from_text(text: str, file_path: Path) -> Document:
    if not text.strip():
        msg = f"Document '{file_path}' is empty."
        raise EmptyDocumentError(msg)

    return Document(
        text=text,
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower().lstrip("."),
        },
    )
