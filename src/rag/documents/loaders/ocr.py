"""OCR document loader for image files."""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

from rag.documents.loaders.text import _document_from_text
from rag.documents.models import Document
from rag.exceptions import DocumentLoadError


class OcrImageLoader:
    """Extract text from images using Pillow and pytesseract."""

    supported_suffixes = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})

    def load(self, path: str | Path) -> Document:
        file_path = Path(path)
        image_open, image_to_string = _load_ocr_dependencies()
        try:
            image = image_open(file_path)
            text = image_to_string(image)
        except Exception as exc:
            msg = f"Could not extract OCR text from image '{file_path}'."
            raise DocumentLoadError(msg) from exc

        return _document_from_text(text, file_path)


def _load_ocr_dependencies() -> tuple[
    Callable[[Path], object],
    Callable[[object], str],
]:
    try:
        image_module = importlib.import_module("PIL.Image")
        pytesseract_module = importlib.import_module("pytesseract")
    except ImportError as exc:
        msg = "Install the 'ocr' extra to use OCR image loading."
        raise DocumentLoadError(msg) from exc

    image_open = getattr(image_module, "open", None)
    image_to_string = getattr(pytesseract_module, "image_to_string", None)
    if image_open is None or image_to_string is None:
        msg = "OCR dependencies are installed but do not expose the expected API."
        raise DocumentLoadError(msg)

    return (
        cast(Callable[[Path], object], image_open),
        cast(Callable[[object], str], image_to_string),
    )
