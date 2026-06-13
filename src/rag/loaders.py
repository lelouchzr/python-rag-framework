"""Backward-compatible loader exports."""

from rag.documents.loaders import (
    DocumentLoader,
    DocxLoader,
    FileLoader,
    MarkdownLoader,
    OcrImageLoader,
    PdfLoader,
    TextFileLoader,
    TextLoader,
    TxtLoader,
)

__all__ = [
    "DocumentLoader",
    "DocxLoader",
    "FileLoader",
    "MarkdownLoader",
    "OcrImageLoader",
    "PdfLoader",
    "TextFileLoader",
    "TextLoader",
    "TxtLoader",
]
