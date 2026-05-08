"""Inspect one local date without sending private data outside the machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.places.matcher import match_place, summarize_place_matches
from personal_lifelog_rag.places.schemas import Place


ALL_RECORD_LIMIT = 1_000_000
DEFAULT_SNIPPET_LIMIT = 20
SNIPPET_TEXT_CHARS = 60


@dataclass(frozen=True)
class TimeRange:
    first_photo_at: str | None = None
    last_photo_at: str | None = None
    first_line_at: str | None = None
    last_line_at: str | None = None


@dataclass(frozen=True)
class GPSSummary:
    photo_count: int = 0
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    registered_places: list[dict[str, Any]] = field(default_factory=list)
    hide_coordinates: bool = False


@dataclass(frozen=True)
class LineSample:
    time: str
    sender: str
    text: str


@dataclass(frozen=True)
class PhotoSample:
    time: str
    file_name: str
    has_gps: bool
    thumbnail_path: str | None = None


@dataclass(frozen=True)
class DateInspection:
    target_date: str
    photo_count: int
    gps_photo_count: int
    line_message_count: int
    event_count: int
    event_evidence_count: int
    time_range: TimeRange
    photo_hourly_counts: list[int] = field(default_factory=lambda: [0] * 24)
    line_hourly_counts: list[int] = field(default_factory=lambda: [0] * 24)
    line_samples: list[LineSample] = field(default_factory=list)
    photo_samples: list[PhotoSample] = field(default_factory=list)
    gps_summary: GPSSummary = field(default_factory=GPSSummary)
    snippets_enabled: bool = True


def inspect_date(
    repository,
    target_date: str,
    *,
    limit: int = DEFAULT_SNIPPET_LIMIT,
    include_snippets: bool = True,
    place_dictionary: list[Place] | None = None,
) -> DateInspection:
    """Collect safe diagnostics for one date from the local database."""

    normalized_date = _normalize_date(target_date)
    safe_limit = max(limit, 0)
    media_items = repository.list_media_items(
        start_date=normalized_date,
        end_date=normalized_date,
        limit=ALL_RECORD_LIMIT,
    )
    line_messages = repository.list_line_messages(
        start_date=normalized_date,
        end_date=normalized_date,
        limit=ALL_RECORD_LIMIT,
    )
    events = repository.list_events(
        start_date=normalized_date,
        end_date=normalized_date,
        limit=ALL_RECORD_LIMIT,
    )
    evidence_count = sum(len(repository.list_event_evidence(event["id"])) for event in events)
    photo_times = [_media_time(item) for item in media_items]
    line_times = [_parse_datetime(message.get("sent_at")) for message in line_messages]
    gps_items = [item for item in media_items if _has_gps(item)]

    return DateInspection(
        target_date=normalized_date,
        photo_count=len(media_items),
        gps_photo_count=len(gps_items),
        line_message_count=len(line_messages),
        event_count=len(events),
        event_evidence_count=evidence_count,
        time_range=TimeRange(
            first_photo_at=_time_label(_min_datetime(photo_times)),
            last_photo_at=_time_label(_max_datetime(photo_times)),
            first_line_at=_time_label(_min_datetime(line_times)),
            last_line_at=_time_label(_max_datetime(line_times)),
        ),
        photo_hourly_counts=_hourly_counts(photo_times),
        line_hourly_counts=_hourly_counts(line_times),
        line_samples=_line_samples(line_messages, safe_limit) if include_snippets else [],
        photo_samples=_photo_samples(media_items, safe_limit) if include_snippets else [],
        gps_summary=_gps_summary(gps_items, place_dictionary=place_dictionary or []),
        snippets_enabled=include_snippets,
    )


def format_date_inspection(inspection: DateInspection) -> str:
    """Format date diagnostics for CLI output with privacy-conscious snippets."""

    lines = [f"Inspect date: {inspection.target_date}", ""]
    lines.extend(
        [
            "基本統計:",
            f"- 写真件数: {inspection.photo_count}",
            f"- GPS付き写真件数: {inspection.gps_photo_count}",
            f"- LINEメッセージ件数: {inspection.line_message_count}",
            f"- events件数: {inspection.event_count}",
            f"- event_evidence件数: {inspection.event_evidence_count}",
            "",
            "時間範囲:",
            f"- 最初の写真時刻: {inspection.time_range.first_photo_at or 'なし'}",
            f"- 最後の写真時刻: {inspection.time_range.last_photo_at or 'なし'}",
            f"- 最初のLINE時刻: {inspection.time_range.first_line_at or 'なし'}",
            f"- 最後のLINE時刻: {inspection.time_range.last_line_at or 'なし'}",
            "",
            "時間帯分布:",
            "hour  photos  line_messages",
        ]
    )
    for hour in range(24):
        lines.append(
            f"{hour:02d}    {inspection.photo_hourly_counts[hour]:>6}  "
            f"{inspection.line_hourly_counts[hour]:>13}"
        )

    lines.append("")
    lines.extend(_format_line_samples(inspection))
    lines.append("")
    lines.extend(_format_photo_samples(inspection))
    lines.append("")
    lines.extend(_format_gps_summary(inspection.gps_summary))
    return "\n".join(lines)


def _format_line_samples(inspection: DateInspection) -> list[str]:
    if not inspection.snippets_enabled:
        return ["LINEサンプル:", "- --no-snippets により非表示"]
    if not inspection.line_samples:
        return ["LINEサンプル:", "- なし"]
    lines = ["LINEサンプル:"]
    for sample in inspection.line_samples:
        lines.append(f"- {sample.time} {sample.sender}: {sample.text}")
    return lines


def _format_photo_samples(inspection: DateInspection) -> list[str]:
    if not inspection.snippets_enabled:
        return ["写真サンプル:", "- --no-snippets により非表示"]
    if not inspection.photo_samples:
        return ["写真サンプル:", "- なし"]
    lines = ["写真サンプル:"]
    for sample in inspection.photo_samples:
        gps = "GPSあり" if sample.has_gps else "GPSなし"
        thumbnail = f" thumbnail={sample.thumbnail_path}" if sample.thumbnail_path else ""
        lines.append(f"- {sample.time} {sample.file_name} {gps}{thumbnail}")
    return lines


def _format_gps_summary(summary: GPSSummary) -> list[str]:
    if summary.photo_count == 0:
        return ["GPS概要:", "- GPS付き写真はありません。"]
    lines = [
        "GPS概要:",
        f"- GPS付き写真: {summary.photo_count}枚",
    ]
    if summary.registered_places:
        lines.append("- 登録済み場所候補:")
        for row in summary.registered_places[:10]:
            lines.append(f"  - {row['display_name']}: {row['count']}枚")
    if summary.hide_coordinates:
        lines.append("- lat/lon: 登録場所のprivacy設定により非表示")
    else:
        lines.extend(
            [
                f"- lat min/max: {summary.lat_min:.3f} / {summary.lat_max:.3f}",
                f"- lon min/max: {summary.lon_min:.3f} / {summary.lon_max:.3f}",
            ]
        )
    return lines


def _line_samples(line_messages: list[dict[str, Any]], limit: int) -> list[LineSample]:
    samples: list[LineSample] = []
    for message in line_messages[:limit]:
        samples.append(
            LineSample(
                time=_time_label(_parse_datetime(message.get("sent_at"))) or "時刻不明",
                sender=redact_text(str(message.get("sender") or message.get("sender_name") or "unknown"), max_chars=24),
                text=redact_text(message.get("text") or message.get("message_text"), max_chars=SNIPPET_TEXT_CHARS),
            )
        )
    return samples


def _photo_samples(media_items: list[dict[str, Any]], limit: int) -> list[PhotoSample]:
    samples: list[PhotoSample] = []
    for item in media_items[:limit]:
        samples.append(
            PhotoSample(
                time=_time_label(_media_time(item)) or "時刻不明",
                file_name=redact_text(str(item.get("file_name") or "image"), max_chars=80),
                has_gps=_has_gps(item),
                thumbnail_path=item.get("thumbnail_path"),
            )
        )
    return samples


def _gps_summary(gps_items: list[dict[str, Any]], *, place_dictionary: list[Place]) -> GPSSummary:
    if not gps_items:
        return GPSSummary()
    lats = [_float_value(item.get("gps_lat")) for item in gps_items]
    lons = [_float_value(item.get("gps_lon")) for item in gps_items]
    lats = [value for value in lats if value is not None]
    lons = [value for value in lons if value is not None]
    if not lats or not lons:
        return GPSSummary()
    matches = [
        match
        for item in gps_items
        if (match := match_place(item.get("gps_lat"), item.get("gps_lon"), place_dictionary)) is not None
    ]
    hide_coordinates = any(
        match.privacy_level == "sensitive" and not match.show_exact_location
        for match in matches
    )
    return GPSSummary(
        photo_count=len(gps_items),
        lat_min=round(min(lats), 3),
        lat_max=round(max(lats), 3),
        lon_min=round(min(lons), 3),
        lon_max=round(max(lons), 3),
        registered_places=summarize_place_matches(matches),
        hide_coordinates=hide_coordinates,
    )


def _hourly_counts(values: list[datetime | None]) -> list[int]:
    counts = [0] * 24
    for value in values:
        if value is not None:
            counts[value.hour] += 1
    return counts


def _media_time(item: dict[str, Any]) -> datetime | None:
    return _parse_datetime(item.get("captured_at") or item.get("fallback_captured_at"))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return None


def _min_datetime(values: list[datetime | None]) -> datetime | None:
    parsed = [value for value in values if value is not None]
    return min(parsed) if parsed else None


def _max_datetime(values: list[datetime | None]) -> datetime | None:
    parsed = [value for value in values if value is not None]
    return max(parsed) if parsed else None


def _time_label(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%H:%M")


def _has_gps(item: dict[str, Any]) -> bool:
    return item.get("gps_lat") is not None and item.get("gps_lon") is not None


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_date(value: str) -> str:
    return date.fromisoformat(value).isoformat()
