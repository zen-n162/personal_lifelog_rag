"""Ingest local photo files into the lifelog database."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from personal_lifelog_rag.ingest.exif_reader import (
    MEDIA_EXTENSIONS,
    MediaMetadata,
    read_media_metadata,
)


@dataclass(frozen=True)
class PhotoIngestReport:
    scanned: int
    imported: int
    duplicates: int
    skipped: int


def iter_media_files(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).expanduser()
    if root.is_file():
        return [root] if root.suffix.lower() in MEDIA_EXTENSIONS else []
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def ingest_photo_directory(
    root_dir: str | Path,
    repository,
    *,
    thumbnails_dir: str | Path = "data/thumbnails",
) -> int:
    """Store local image metadata and thumbnails; return newly inserted rows."""

    return ingest_photo_directory_with_report(
        root_dir,
        repository,
        thumbnails_dir=thumbnails_dir,
    ).imported


def ingest_photo_directory_with_report(
    root_dir: str | Path,
    repository,
    *,
    thumbnails_dir: str | Path = "data/thumbnails",
) -> PhotoIngestReport:
    """Store local image metadata and thumbnails with safe summary counts."""

    files = iter_media_files(root_dir)
    count = 0
    duplicates = 0
    skipped = 0
    for path in files:
        try:
            metadata = read_media_metadata(path)
            thumbnail_path = create_thumbnail(metadata.file_path, metadata.file_hash, thumbnails_dir)
        except Exception:
            skipped += 1
            continue

        metadata = replace(metadata, thumbnail_path=thumbnail_path)
        before = repository.stats()["media_items"]
        repository.add_media_item(
            id=metadata.id,
            file_path=str(metadata.file_path),
            file_name=metadata.file_name,
            file_hash=metadata.file_hash,
            media_type=metadata.media_type,
            captured_at=metadata.captured_at,
            fallback_captured_at=metadata.fallback_captured_at,
            gps_lat=metadata.gps_lat,
            gps_lon=metadata.gps_lon,
            camera_model=metadata.camera_model,
            width=metadata.width,
            height=metadata.height,
            thumbnail_path=str(metadata.thumbnail_path) if metadata.thumbnail_path else None,
        )
        after = repository.stats()["media_items"]
        inserted = max(after - before, 0)
        count += inserted
        if inserted == 0:
            duplicates += 1
    return PhotoIngestReport(
        scanned=len(files),
        imported=count,
        duplicates=duplicates,
        skipped=skipped,
    )


def create_thumbnail(
    image_path: str | Path,
    file_hash: str,
    thumbnails_dir: str | Path,
    *,
    size: tuple[int, int] = (256, 256),
) -> Path:
    """Create a local JPEG thumbnail under data/thumbnails."""

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow is required for thumbnail generation") from exc

    thumbnail_dir = Path(thumbnails_dir).expanduser()
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = thumbnail_dir / f"{file_hash[:24]}.jpg"
    if thumbnail_path.exists():
        return thumbnail_path

    with Image.open(Path(image_path).expanduser()) as image:
        image.thumbnail(size)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(thumbnail_path, format="JPEG", quality=85)

    return thumbnail_path
