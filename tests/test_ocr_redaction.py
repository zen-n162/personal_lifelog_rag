from __future__ import annotations

from personal_lifelog_rag.ocr.redaction import redact_ocr_text


def test_redaction_masks_email_phone_and_long_numbers() -> None:
    text = "contact test@example.com phone 090-1234-5678 ticket 123456789012"

    redacted = redact_ocr_text(text)

    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted
    assert "[NUMBER]" in redacted
    assert "test@example.com" not in redacted


def test_redaction_truncates_preview() -> None:
    redacted = redact_ocr_text("a" * 100, max_chars=12)

    assert len(redacted) <= 13
    assert redacted.endswith("…")
