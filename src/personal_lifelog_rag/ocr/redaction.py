"""Privacy-friendly OCR preview redaction."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s])?(?:\d{2,4}[-\s]){2,4}\d{2,4}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
POSTAL_CODE_RE = re.compile(r"(?<!\d)(?:〒\s*)?\d{3}-\d{4}(?!\d)")
ADDRESS_LIKE_RE = re.compile(
    r"(?:東京都|北海道|京都府|大阪府|.{2,3}県).{0,24}?(?:市|区|町|村).{0,24}?\d{1,4}(?:[-ー−]\d{1,4}){0,3}"
)


def redact_ocr_text(text: str | None, *, max_chars: int | None = None) -> str:
    """Redact obvious sensitive tokens for logs and previews."""

    if not text:
        return ""
    redacted = URL_RE.sub("[URL]", str(text))
    redacted = EMAIL_RE.sub("[EMAIL]", redacted)
    redacted = PHONE_RE.sub("[PHONE]", redacted)
    redacted = POSTAL_CODE_RE.sub("[POSTAL_CODE]", redacted)
    redacted = ADDRESS_LIKE_RE.sub("[ADDRESS]", redacted)
    redacted = LONG_NUMBER_RE.sub("[NUMBER]", redacted)
    redacted = _squash_long_whitespace(redacted)
    if max_chars is not None and len(redacted) > max_chars:
        return redacted[: max(max_chars - 1, 0)].rstrip() + "…"
    return redacted


def _squash_long_whitespace(value: str) -> str:
    return re.sub(r"\s{3,}", "\n", value.strip())
