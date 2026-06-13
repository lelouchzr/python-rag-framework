"""DOCX document loader."""

import importlib
from pathlib import Path
from typing import Protocol, cast

from rag.documents.loaders.text import _document_from_text
from rag.documents.models import Document
from rag.exceptions import DocumentLoadError


class _DocxParagraph(Protocol):
    text: str


class _DocxCell(Protocol):
    text: str


class _DocxRow(Protocol):
    cells: list[_DocxCell]


class _DocxTable(Protocol):
    rows: list[_DocxRow]


class _DocxDocument(Protocol):
    paragraphs: list[_DocxParagraph]
    tables: list[_DocxTable]


class DocxLoader:
    """Extract text from DOCX files."""

    supported_suffixes = frozenset({".docx"})

    def load(self, path: str | Path) -> Document:
        file_path = Path(path)
        try:
            doc = _load_docx_document(file_path)
            paragraphs = [
                paragraph.text.strip()
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            ]
            table_cells = [
                cell.text.strip()
                for table in doc.tables
                for row in table.rows
                for cell in row.cells
                if cell.text.strip()
            ]
        except Exception as exc:
            msg = f"Could not extract text from DOCX '{file_path}'."
            raise DocumentLoadError(msg) from exc

        return _document_from_text("\n\n".join([*paragraphs, *table_cells]), file_path)


def _load_docx_document(file_path: Path) -> _DocxDocument:
    try:
        docx_module = importlib.import_module("docx")
    except ImportError as exc:
        msg = "Install the 'docx' extra to load DOCX files."
        raise DocumentLoadError(msg) from exc

    document_factory = getattr(docx_module, "Document", None)
    if document_factory is None:
        msg = "The installed python-docx package does not expose Document."
        raise DocumentLoadError(msg)
    return cast(_DocxDocument, document_factory(str(file_path)))
