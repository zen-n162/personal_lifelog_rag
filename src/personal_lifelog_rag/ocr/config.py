"""Small local OCR runtime config reader.

This intentionally uses the repo's tiny YAML parser instead of adding a YAML
dependency. OCR config is local-only and never triggers installation or model
downloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from personal_lifelog_rag.benchmark.schemas import _parse_nested_mapping


@dataclass(frozen=True)
class OcrRuntimeConfig:
    engine: str | None = None
    languages: str = "jpn+eng"
    tesseract_cmd: str = "tesseract"
    psm: int | None = 6
    oem: int | None = 1
    min_confidence: float = 0.0
    max_text_length: int = 5000
    redact_sensitive: bool = True
    local_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ocr_runtime_config(path: str | Path | None = None) -> OcrRuntimeConfig:
    """Load optional `ocr:` config from a local YAML/JSON-like file."""

    if path is None:
        return OcrRuntimeConfig()
    source = Path(path).expanduser()
    if not source.exists():
        return OcrRuntimeConfig()
    payload = _parse_nested_mapping(source.read_text(encoding="utf-8"))
    row = payload.get("ocr", {}) if isinstance(payload, dict) else {}
    if not isinstance(row, dict):
        row = {}
    return OcrRuntimeConfig(
        engine=_none_or_str(row.get("engine")),
        languages=_none_or_str(row.get("languages")) or "jpn+eng",
        tesseract_cmd=_none_or_str(row.get("tesseract_cmd")) or "tesseract",
        psm=_none_or_int(row.get("psm")) if row.get("psm") is not None else 6,
        oem=_none_or_int(row.get("oem")) if row.get("oem") is not None else 1,
        min_confidence=_none_or_float(row.get("min_confidence")) or 0.0,
        max_text_length=_none_or_int(row.get("max_text_length")) or 5000,
        redact_sensitive=_bool_or_default(row.get("redact_sensitive"), True),
        local_only=_bool_or_default(row.get("local_only"), True),
    )


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return default
