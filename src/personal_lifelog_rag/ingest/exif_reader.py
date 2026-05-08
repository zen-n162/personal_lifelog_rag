"""Read local image metadata without sending data anywhere."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.utils import sha256_file

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UNSUPPORTED_IMAGE_EXTENSIONS = {".heic"}
MEDIA_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class MediaMetadata:
    id: str
    file_path: Path
    file_name: str
    file_hash: str
    media_type: str
    captured_at: str | None
    fallback_captured_at: str
    gps_lat: float | None
    gps_lon: float | None
    camera_model: str | None
    width: int | None
    height: int | None
    thumbnail_path: Path | None = None
    exif: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_media_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in UNSUPPORTED_IMAGE_EXTENSIONS:
        return "unsupported_heic"
    return "unknown"


def read_media_metadata(path: str | Path) -> MediaMetadata:
    """Read file, image, and EXIF metadata locally with Pillow."""

    resolved = Path(path).expanduser()
    stat = resolved.stat()
    file_hash = sha256_file(resolved)
    fallback_captured_at = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()
    metadata: dict[str, Any] = {
        "file_size": stat.st_size,
        "mime_type": mimetypes.guess_type(resolved.name)[0],
    }

    pillow_data = _read_with_pillow(resolved)
    exif = pillow_data.get("exif", {})

    return MediaMetadata(
        id=f"media_{file_hash}",
        file_path=resolved,
        file_name=resolved.name,
        file_hash=file_hash,
        media_type=detect_media_type(resolved),
        captured_at=pillow_data.get("captured_at"),
        fallback_captured_at=fallback_captured_at,
        gps_lat=pillow_data.get("gps_lat"),
        gps_lon=pillow_data.get("gps_lon"),
        camera_model=pillow_data.get("camera_model"),
        width=pillow_data.get("width"),
        height=pillow_data.get("height"),
        exif=exif,
        metadata=metadata,
    )


def _read_with_pillow(path: Path) -> dict[str, Any]:
    try:
        from PIL import ExifTags, Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow is required for image ingestion") from exc

    exif_ifd: dict[int, Any] = {}
    gps_info: dict[int, Any] | None = None
    with Image.open(path) as image:
        width, height = image.size
        raw_exif = image.getexif()
        try:
            exif_ifd = dict(raw_exif.get_ifd(ExifTags.IFD.Exif))
        except Exception:
            exif_ifd = {}
        try:
            gps_info = dict(raw_exif.get_ifd(ExifTags.IFD.GPSInfo))
        except Exception:
            gps_info = None

    if not raw_exif:
        return {
            "width": width,
            "height": height,
            "exif": {},
            "captured_at": None,
            "gps_lat": None,
            "gps_lon": None,
            "camera_model": None,
        }

    tag_names = {tag_id: name for tag_id, name in ExifTags.TAGS.items()}
    exif = {tag_names.get(key, str(key)): value for key, value in raw_exif.items()}
    exif.update({tag_names.get(key, str(key)): value for key, value in exif_ifd.items()})
    captured_at = _parse_exif_datetime(exif.get("DateTimeOriginal") or exif.get("DateTime"))
    camera_model = _first_text(exif.get("Model"))

    gps_lat = None
    gps_lon = None
    gps_info = gps_info or exif.get("GPSInfo")
    if gps_info:
        gps_lat, gps_lon = _parse_gps(gps_info, ExifTags)

    return {
        "width": width,
        "height": height,
        "exif": _json_safe_exif(exif),
        "captured_at": captured_at,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "camera_model": camera_model,
    }


def _parse_exif_datetime(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).isoformat()
        except ValueError:
            continue
    return None


def _parse_gps(gps_info: Any, exif_tags: Any) -> tuple[float | None, float | None]:
    if not hasattr(gps_info, "items"):
        return None, None
    gps_tag_names = {tag_id: name for tag_id, name in exif_tags.GPSTAGS.items()}
    named = {gps_tag_names.get(key, key): value for key, value in dict(gps_info).items()}
    lat = _gps_decimal(named.get("GPSLatitude"), named.get("GPSLatitudeRef"))
    lon = _gps_decimal(named.get("GPSLongitude"), named.get("GPSLongitudeRef"))
    return lat, lon


def _gps_decimal(value: Any, ref: Any) -> float | None:
    if not value or len(value) != 3:
        return None
    degrees, minutes, seconds = (float(item) for item in value)
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in {"S", "W"}:
        decimal *= -1
    return decimal


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_safe_exif(exif: dict[str, Any]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in exif.items():
        if key == "MakerNote":
            continue
        safe[str(key)] = str(value)
    return safe
