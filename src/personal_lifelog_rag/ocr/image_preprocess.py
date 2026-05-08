"""Small local image preprocessing helpers for OCR."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator

from PIL import Image, ImageOps


@contextmanager
def preprocessed_image_path(
    image_path: Path,
    *,
    max_side: int = 1800,
    grayscale: bool = True,
) -> Iterator[Path]:
    """Yield a temporary OCR-friendly image path without modifying the source."""

    temp_path: Path | None = None
    try:
        with Image.open(image_path) as image:
            processed = ImageOps.exif_transpose(image)
            processed.thumbnail((max_side, max_side))
            if grayscale:
                processed = processed.convert("L")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temp_path = Path(handle.name)
            processed.save(temp_path)
        yield temp_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
