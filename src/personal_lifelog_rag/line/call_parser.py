"""Parse LINE call-log messages into structured local records."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from personal_lifelog_rag.core.privacy import redact_text


CallStatus = Literal["completed", "missed", "unanswered", "canceled", "unknown"]

CALL_MARKERS = ("通話", "不在着信", "着信", "電話", "☎")
CALL_DURATION_RE = re.compile(r"通話時間\s+([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)")


@dataclass(frozen=True)
class ParsedLineCall:
    call_status: CallStatus
    duration_sec: int | None = None
    raw_text_short: str = ""
    warnings: list[str] = field(default_factory=list)


def parse_line_call_text(text: str | None, *, max_raw_chars: int = 60) -> ParsedLineCall | None:
    """Return a structured call event if a LINE message looks call-related."""

    raw_text = text or ""
    compact = " ".join(raw_text.split())
    if not _looks_like_call(compact):
        return None

    duration_match = CALL_DURATION_RE.search(compact)
    if duration_match:
        duration = parse_call_duration(duration_match.group(1))
        warnings = [] if duration is not None else [f"unparseable duration: {duration_match.group(1)}"]
        return ParsedLineCall(
            call_status="completed",
            duration_sec=duration,
            raw_text_short=redact_text(compact, max_chars=max_raw_chars),
            warnings=warnings,
        )

    if "不在着信" in compact:
        status: CallStatus = "missed"
    elif "応答がありませんでした" in compact:
        status = "unanswered"
    elif "通話をキャンセル" in compact or "☎ キャンセル" in compact or compact.strip() == "キャンセル":
        status = "canceled"
    else:
        status = "unknown"

    return ParsedLineCall(
        call_status=status,
        duration_sec=None,
        raw_text_short=redact_text(compact, max_chars=max_raw_chars),
    )


def parse_call_duration(value: str) -> int | None:
    """Parse LINE call duration values such as MM:SS or H:MM:SS into seconds."""

    parts = value.strip().split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return None

    try:
        resolved_hours = int(hours)
        resolved_minutes = int(minutes)
        resolved_seconds = int(seconds)
    except ValueError:
        return None
    if resolved_hours < 0 or not 0 <= resolved_minutes <= 59 or not 0 <= resolved_seconds <= 59:
        return None
    return resolved_hours * 3600 + resolved_minutes * 60 + resolved_seconds


def _looks_like_call(text: str) -> bool:
    return any(marker in text for marker in CALL_MARKERS)

