"""PDF document loader."""

import importlib
from pathlib import Path
from typing import Protocol, cast

from rag.documents.loaders.text import _document_from_text
from rag.documents.models import Document
from rag.exceptions import DocumentLoadError


class _PdfPage(Protocol):
    def extract_text(self) -> str | None: ...


class _PdfReader(Protocol):
    pages: list[_PdfPage]


class PdfLoader:
    """Extract text from PDF files."""

    supported_suffixes = frozenset({".pdf"})

    def load(self, path: str | Path) -> Document:
        file_path = Path(path)
        try:
            reader = _load_pdf_reader(file_path)
            pages: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append(page_text.strip())
        except Exception as exc:
            msg = f"Could not extract text from PDF '{file_path}'."
            raise DocumentLoadError(msg) from exc

        return _document_from_text("\n\n".join(pages), file_path)


def _load_pdf_reader(file_path: Path) -> _PdfReader:
    try:
        pypdf_module = importlib.import_module("pypdf")
    except ImportError as exc:
        msg = "Install the 'pdf' extra to load PDF files."
        raise DocumentLoadError(msg) from exc

    reader_factory = getattr(pypdf_module, "PdfReader", None)
    if reader_factory is None:
        msg = "The installed pypdf package does not expose PdfReader."
        raise DocumentLoadError(msg)
    return cast(_PdfReader, reader_factory(file_path))
