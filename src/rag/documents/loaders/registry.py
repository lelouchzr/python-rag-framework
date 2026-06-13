"""Dispatch file loading to format-specific loaders."""

from pathlib import Path

from rag.documents.loaders.base import DocumentLoader
from rag.documents.loaders.docx import DocxLoader
from rag.documents.loaders.ocr import OcrImageLoader
from rag.documents.loaders.pdf import PdfLoader
from rag.documents.loaders.text import MarkdownLoader, TxtLoader
from rag.documents.models import Document
from rag.exceptions import UnsupportedFileTypeError


class FileLoader:
    """Load supported files into framework documents."""

    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders = loaders or [
            TxtLoader(),
            MarkdownLoader(),
            PdfLoader(),
            DocxLoader(),
            OcrImageLoader(),
        ]

    @property
    def supported_suffixes(self) -> frozenset[str]:
        suffixes: set[str] = set()
        for loader in self._loaders:
            suffixes.update(loader.supported_suffixes)
        return frozenset(suffixes)

    def load(self, path: str | Path) -> Document:
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        for loader in self._loaders:
            if suffix in loader.supported_suffixes:
                return loader.load(file_path)

        supported = ", ".join(sorted(self.supported_suffixes))
        msg = f"Unsupported file type '{suffix}'. Supported types: {supported}."
        raise UnsupportedFileTypeError(msg)
