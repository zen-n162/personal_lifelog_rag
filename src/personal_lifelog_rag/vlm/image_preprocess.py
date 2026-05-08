"""VLM image preprocessing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile

from PIL import Image, ImageOps


@contextmanager
def preprocessed_vlm_image_path(image_path: Path, *, max_side: int = 1024):
    """Yield a temporary low-resolution image path without modifying the source."""

    tmp_path: Path | None = None
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((max_side, max_side))
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            handle = tempfile.NamedTemporaryFile(prefix="lifelog_vlm_", suffix=".jpg", delete=False)
            tmp_path = Path(handle.name)
            handle.close()
            image.save(tmp_path, format="JPEG", quality=88)
        yield tmp_path
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

