"""Report-specific redaction for public portfolio output."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from personal_lifelog_rag.ocr.redaction import redact_ocr_text


GPS_RE = re.compile(r"(?<!\d)(?:[+-]?\d{1,3}\.\d{3,})\s*,\s*(?:[+-]?\d{1,3}\.\d{3,})(?!\d)")
MEDIA_ID_RE = re.compile(r"\bmedia_[A-Za-z0-9_-]{8,}\b")
PATH_RE = re.compile(r"(?:/[^\s]+)+|(?:[A-Za-z]:\\[^\s]+)")


class ReportRedactor:
    """Stable, local-only anonymizer for generated reports."""

    def __init__(self, *, public: bool = True) -> None:
        self.public = public
        self._people: dict[str, str] = {}
        self._senders: dict[str, str] = {}
        self._places: dict[str, str] = {}

    def text(self, value: Any, *, max_chars: int = 120) -> str:
        if value is None:
            return ""
        text = str(value)
        text = GPS_RE.sub("[GPS]", text)
        text = MEDIA_ID_RE.sub(lambda match: self.media_id(match.group(0)), text)
        text = PATH_RE.sub("[PATH]", text)
        text = redact_ocr_text(text, max_chars=None)
        compact = " ".join(text.split())
        if len(compact) > max_chars:
            compact = compact[: max(max_chars - 1, 0)].rstrip() + "…"
        return compact

    def person(self, value: Any) -> str:
        if not value:
            return ""
        raw = str(value).strip()
        if not self.public:
            return self.text(raw, max_chars=40)
        return self._people.setdefault(raw, f"PERSON_{len(self._people) + 1}")

    def sender(self, value: Any) -> str:
        if not value:
            return ""
        raw = str(value).strip()
        if not self.public:
            return self.text(raw, max_chars=40)
        return self._senders.setdefault(raw, f"SENDER_{len(self._senders) + 1}")

    def place(self, value: Any, *, sensitive: bool = False) -> str:
        if not value:
            return ""
        if sensitive:
            return "SENSITIVE_PLACE"
        raw = str(value).strip()
        if not self.public:
            return self.text(raw, max_chars=60)
        return self._places.setdefault(raw, f"PLACE_{len(self._places) + 1}")

    def date(self, value: Any) -> str:
        text = str(value or "")
        if not text:
            return ""
        return text[:7] if self.public and len(text) >= 7 else text[:10]

    def media_id(self, value: Any) -> str:
        text = str(value or "")
        if not text:
            return ""
        if self.public:
            return "MEDIA_ID_REDACTED"
        return text[:16] + "…" if len(text) > 16 else text

    def file_path(self, value: Any) -> str:
        if not value:
            return ""
        if self.public:
            return "[PATH]"
        return Path(str(value)).name
