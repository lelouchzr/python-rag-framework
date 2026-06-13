"""Document loader implementations."""

from rag.documents.loaders.base import DocumentLoader
from rag.documents.loaders.docx import DocxLoader
from rag.documents.loaders.ocr import OcrImageLoader
from rag.documents.loaders.pdf import PdfLoader
from rag.documents.loaders.registry import FileLoader
from rag.documents.loaders.text import MarkdownLoader, TextLoader, TxtLoader

TextFileLoader = FileLoader

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
