"""Privacy-friendly OCR preview redaction."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s])?(?:\d{2,4}[-\s]){2,4}\d{2,4}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")


def redact_ocr_text(text: str | None, *, max_chars: int | None = None) -> str:
    """Redact obvious sensitive tokens for logs and previews."""

    if not text:
        return ""
    redacted = EMAIL_RE.sub("[EMAIL]", str(text))
    redacted = PHONE_RE.sub("[PHONE]", redacted)
    redacted = LONG_NUMBER_RE.sub("[NUMBER]", redacted)
    redacted = _squash_long_whitespace(redacted)
    if max_chars is not None and len(redacted) > max_chars:
        return redacted[: max(max_chars - 1, 0)].rstrip() + "…"
    return redacted


def _squash_long_whitespace(value: str) -> str:
    return re.sub(r"\s{3,}", "\n", value.strip())
