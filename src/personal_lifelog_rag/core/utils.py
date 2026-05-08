"""Small shared utilities."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def iso_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value[:10]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
