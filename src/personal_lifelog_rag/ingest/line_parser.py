"""Parser for locally exported LINE chat history text files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


@dataclass
class LineMessage:
    message_id: str
    chat_id: str
    source_path: Path
    source_file: str
    sent_at: str
    chat_name: str | None
    sender_name: str | None
    message_text: str
    message_type: str = "text"
    timezone: str | None = "Asia/Tokyo"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LineParseWarning:
    source_path: Path
    line_number: int
    reason: str


@dataclass(frozen=True)
class LineParseResult:
    messages: list[LineMessage]
    warnings: list[LineParseWarning]


GENERIC_CHAT_DIR_NAMES = {"line", "raw", "data", "fixtures", "exports", "talk", "talks"}
MESSAGE_TYPES = {"text", "image", "video", "sticker", "file", "system", "unknown"}
DATE_PATTERNS = [
    re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})(?:\(.+\))?$"),
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+.+)?$"),
    re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\s+.+)?$"),
]
MESSAGE_PATTERN = re.compile(
    r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})\t(?P<sender>[^\t]+)\t(?P<text>.*)$"
)


def parse_line_chat_file(
    path: str | Path,
    *,
    chat_name: str | None = None,
    timezone: str = "Asia/Tokyo",
) -> list[LineMessage]:
    return parse_line_chat_file_with_warnings(
        path,
        chat_name=chat_name,
        timezone=timezone,
    ).messages


def parse_line_chat_file_with_warnings(
    path: str | Path,
    *,
    chat_name: str | None = None,
    timezone: str = "Asia/Tokyo",
) -> LineParseResult:
    source_path = Path(path).expanduser()
    text = source_path.read_text(encoding="utf-8-sig")
    resolved_chat_name = chat_name or source_path.stem
    return parse_line_chat_text_with_warnings(
        text,
        source_path=source_path,
        chat_name=resolved_chat_name,
        timezone=timezone,
    )


def parse_line_chat_text(
    text: str,
    *,
    source_path: str | Path = "line_export.txt",
    chat_name: str | None = None,
    timezone: str = "Asia/Tokyo",
) -> list[LineMessage]:
    return parse_line_chat_text_with_warnings(
        text,
        source_path=source_path,
        chat_name=chat_name,
        timezone=timezone,
    ).messages


def parse_line_chat_text_with_warnings(
    text: str,
    *,
    source_path: str | Path = "line_export.txt",
    chat_name: str | None = None,
    timezone: str = "Asia/Tokyo",
) -> LineParseResult:
    messages: list[LineMessage] = []
    warnings: list[LineParseWarning] = []
    current_date: tuple[int, int, int] | None = None
    current_message: LineMessage | None = None
    source = Path(source_path)
    chat_id = derive_chat_id(source)
    source_file = source.name
    tzinfo = ZoneInfo(timezone)

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        parsed_date = _parse_date_line(line.strip())
        if parsed_date:
            _append_message(messages, current_message)
            current_message = None
            current_date = parsed_date
            continue

        match = MESSAGE_PATTERN.match(line)
        if match and current_date:
            _append_message(messages, current_message)
            year, month, day = current_date
            try:
                sent_at = datetime(
                    year,
                    month,
                    day,
                    int(match.group("hour")),
                    int(match.group("minute")),
                    tzinfo=tzinfo,
                ).isoformat()
            except ValueError:
                warnings.append(LineParseWarning(source, line_number, "invalid datetime"))
                current_message = None
                continue
            sender = match.group("sender").strip()
            message_text = match.group("text").strip()
            message_type = classify_message_type(message_text, sender_name=sender)
            message_id = generate_message_id(
                source_file=source_file,
                sent_at=sent_at,
                sender=sender,
                text=message_text,
            )
            current_message = LineMessage(
                message_id=message_id,
                chat_id=chat_id,
                source_path=source,
                source_file=source_file,
                chat_name=chat_name,
                sender_name=sender,
                message_text=message_text,
                message_type=message_type,
                sent_at=sent_at,
                timezone=timezone,
                metadata={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "source_file": source_file,
                    "source_line": line_number,
                },
            )
            continue

        if current_message and line:
            if current_message.message_type != "text":
                warnings.append(LineParseWarning(source, line_number, "unparsed line after special message"))
                continue
            current_message.message_text = f"{current_message.message_text}\n{line.rstrip()}"
            current_message.message_type = classify_message_type(
                current_message.message_text,
                sender_name=current_message.sender_name,
            )
            current_message.message_id = generate_message_id(
                source_file=current_message.source_file,
                sent_at=current_message.sent_at,
                sender=current_message.sender_name or "",
                text=current_message.message_text,
            )
            current_message.metadata["message_id"] = current_message.message_id
            continue

        warnings.append(LineParseWarning(source, line_number, "unparsed line"))

    _append_message(messages, current_message)
    return LineParseResult(messages=messages, warnings=warnings)


def _parse_date_line(line: str) -> tuple[int, int, int] | None:
    for pattern in DATE_PATTERNS:
        match = pattern.match(line)
        if match:
            year, month, day = (int(part) for part in match.groups())
            try:
                date(year, month, day)
            except ValueError:
                return None
            return year, month, day
    return None


def _append_message(messages: list[LineMessage], message: LineMessage | None) -> None:
    if message is None:
        return
    message.message_type = classify_message_type(
        message.message_text,
        sender_name=message.sender_name,
    )
    message.metadata["message_type"] = message.message_type
    messages.append(message)


def derive_chat_id(source_path: str | Path) -> str:
    path = Path(source_path).expanduser()
    parent_name = path.parent.name.strip()
    if parent_name and parent_name.lower() not in GENERIC_CHAT_DIR_NAMES:
        seed = parent_name
    else:
        seed = path.stem
    return f"line_{_stable_hash(seed)[:16]}"


def generate_message_id(
    *,
    source_file: str,
    sent_at: str,
    sender: str,
    text: str,
) -> str:
    seed = "\u001f".join([source_file, sent_at, sender, text])
    return f"line_msg_{_stable_hash(seed)}"


def classify_message_type(text: str, *, sender_name: str | None = None) -> str:
    normalized = text.strip()
    sender = (sender_name or "").strip().lower()

    if sender in {"line", "system"}:
        return "system"
    if not normalized:
        return "unknown"
    if normalized in {"[写真]", "[画像]", "[photo]", "[image]"}:
        return "image"
    if normalized in {"[動画]", "[video]"}:
        return "video"
    if normalized in {"[スタンプ]", "[sticker]"}:
        return "sticker"
    if normalized.startswith("[ファイル]") or normalized in {"[file]", "[ファイル]"}:
        return "file"
    if _looks_like_system_message(normalized):
        return "system"
    if normalized.startswith("[") and normalized.endswith("]"):
        return "unknown"
    return "text"


def _looks_like_system_message(text: str) -> bool:
    system_markers = (
        "メッセージの送信を取り消しました",
        "通話をキャンセルしました",
        "不在着信",
        "招待しました",
        "退出しました",
        "参加しました",
    )
    return any(marker in text for marker in system_markers)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
