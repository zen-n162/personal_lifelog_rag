"""Extractive answer generation without cloud LLM calls."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.retrieval.confidence import calculate_confidence
from personal_lifelog_rag.retrieval.temporal_search import TimelineSearchResult


def build_answer(question: str, result: TimelineSearchResult) -> str:
    """Build a local-only, evidence-backed answer from retrieved records."""

    total = len(result.events) + len(result.media_items) + len(result.line_messages)
    if total == 0:
        lines = [f"{_scope_label(result)}の記録が見つかりませんでした。"]
        lines.extend(_confidence_section(result))
        return "\n".join(lines)

    lines = [f"{_scope_label(result)}の記録を確認しました。"]
    if result.date_range and result.date_range.ambiguous:
        lines.append("年が省略されていたため、default_yearで補完しています。")
    if result.keyword:
        lines.append(f"キーワード: {result.keyword}")

    if result.events:
        lines.extend(_event_answer_sections(result))
    else:
        lines.extend(_fallback_answer_sections(result))

    lines.extend(_evidence_section(result))
    lines.extend(_confidence_section(result))
    return "\n".join(lines)


def summarize_records(records: list[dict[str, Any]], *, text_key: str) -> list[str]:
    return [redact_text(record.get(text_key), max_chars=60) for record in records]


def _event_answer_sections(result: TimelineSearchResult) -> list[str]:
    events = _sorted_events(result.events)
    lines = ["", f"この日は{len(events)}件の出来事候補があります。", ""]
    for index, event in enumerate(events, start=1):
        title = _display_event_title(event)
        confidence = _float_or_none(event.get("confidence"))
        confidence_text = f" / confidence {confidence:.2f}" if confidence is not None else ""
        lines.append(f"{index}. {_event_time_range(event)} {title}{confidence_text}")
        status_parts: list[str] = []
        if event.get("is_verified"):
            status_parts.append("手動確認済み")
        if event.get("is_pinned"):
            status_parts.append("優先表示")
        if status_parts:
            lines.append(" / ".join(status_parts))

        summary = redact_text(event.get("summary"), max_chars=140)
        if summary:
            lines.append(summary)
        location = redact_text(event.get("location_name"), max_chars=40)
        evidence_count = int(event.get("event_evidence_count") or 0)
        detail_parts: list[str] = []
        if location:
            detail_parts.append(f"場所候補: {location}")
        detail_parts.append(f"根拠: {evidence_count}件")
        lines.append(" / ".join(detail_parts))
        lines.append("")

    lines.extend(
        [
            "推定:",
            "上記はLINE・写真・GPSなどのローカル記録から自動生成した出来事候補です。",
            "明確な本文、caption、OCR、またはGPS根拠がない活動や感情は断定しません。",
        ]
    )
    return lines


def _display_event_title(event: dict[str, Any]) -> str:
    title = redact_text(event.get("title"), max_chars=80) or "記録された出来事"
    summary = str(event.get("summary") or "")
    if title == "外出・写真撮影":
        if "GPS付き写真" in summary or _has_gps(event):
            return "位置情報付き写真の記録"
        return "写真撮影の記録"
    if title == "写真が残る出来事":
        return "写真撮影の記録"
    return title


def _fallback_answer_sections(result: TimelineSearchResult) -> list[str]:
    lines = [""]
    lines.extend(_summary_lines(result))
    lines.extend(_minimal_timeline_section(result))
    lines.extend(_fallback_inference_section(result))
    return lines


def _summary_lines(result: TimelineSearchResult) -> list[str]:
    line_count = len(result.line_messages)
    photo_count = len(result.media_items)
    gps_count = _gps_photo_count(result.media_items)
    lines: list[str] = []

    if line_count:
        highlights = _line_highlights(result.line_messages)
        if highlights:
            lines.append("LINEでは、" + "、".join(highlights) + "。")
        else:
            lines.append(f"LINEメッセージが{line_count}件あります。")
    else:
        lines.append("LINEメッセージは見つかりませんでした。")

    if photo_count:
        lines.append(
            f"この日に撮影された写真が{photo_count}枚あります。"
            f"そのうち{gps_count}枚には位置情報があります。"
        )
    else:
        lines.append("この日に撮影された写真は見つかりませんでした。")
    return lines


def _minimal_timeline_section(result: TimelineSearchResult) -> list[str]:
    if not result.timeline_items:
        return []

    lines = ["", "根拠の抜粋:"]
    for item in result.timeline_items[:5]:
        if item["kind"] == "line_message":
            message = item["record"]
            sender = redact_text(str(message.get("sender") or message.get("sender_name") or "unknown"), max_chars=24)
            text = redact_text(message.get("text") or message.get("message_text"), max_chars=60)
            lines.append(f"- {_time_label(item.get('at'))} LINE {sender}: {text}")
        else:
            media = item["record"]
            gps_suffix = " GPSあり" if _has_gps(media) else " GPSなし"
            lines.append(f"- {_time_label(item.get('at'))} 写真{gps_suffix}")
    return lines


def _fallback_inference_section(result: TimelineSearchResult) -> list[str]:
    hints: list[str] = []
    joined_text = "\n".join((message.get("text") or message.get("message_text") or "") for message in result.line_messages)
    if any(word in joined_text for word in ("待ち合わせ", "着く", "駅", "東口", "西口", "待って")):
        hints.append("待ち合わせや移動に関する会話があった可能性があります")
    if any(word in joined_text for word in ("ご飯", "食事", "ラーメン", "カフェ", "食べ", "おいしかった")):
        hints.append("食事やカフェに関する話題があった可能性があります")
    if result.media_items and result.line_messages:
        hints.append("写真とLINEの両方に同日の記録があります")

    if not hints:
        if len(result.line_messages) + len(result.media_items) <= 2:
            return ["", "推定:", "記録が少ないため、この日の行動は断定できません。"]
        return ["", "推定:", "残っている記録からは、この日の詳細な行動までは断定できません。"]

    return [
        "",
        "推定:",
        "、".join(hints) + "。ただし、根拠が明確でない活動や感情は断定できません。",
    ]


def _evidence_section(result: TimelineSearchResult) -> list[str]:
    return [
        "",
        "根拠:",
        f"- イベント候補: {len(result.events)}件",
        f"- event_evidence: {_event_evidence_count(result.events)}件",
        f"- LINEメッセージ: {len(result.line_messages)}件",
        f"- 写真: {len(result.media_items)}枚",
        f"- GPS付き写真: {_gps_photo_count(result.media_items)}枚",
    ]


def _confidence_section(result: TimelineSearchResult) -> list[str]:
    confidence = calculate_confidence(result)
    return [
        "",
        "信頼度:",
        f"- 日付に記録が存在すること: {confidence['date']}",
        f"- 誰と連絡していたか: {confidence['contact']}",
        f"- どこにいたか: {confidence['place']}",
        f"- 何をしていたか: {confidence['activity']}",
    ]


def _line_highlights(line_messages: list[dict[str, Any]]) -> list[str]:
    waiting: str | None = None
    meal: str | None = None
    for message in line_messages:
        text = message.get("text") or message.get("message_text") or ""
        time_text = _time_label(message.get("sent_at"))
        if waiting is None and any(word in text for word in ("待ち合わせ", "着く", "駅", "東口", "西口", "待って")):
            waiting = f"{time_text}頃に待ち合わせや移動に関する会話があります"
        if meal is None and any(word in text for word in ("ご飯", "食事", "ラーメン", "カフェ", "食べ", "おいしかった")):
            meal = f"{time_text}頃に食事やカフェに関する話題があります"
        if waiting and meal:
            break
    return [highlight for highlight in (waiting, meal) if highlight]


def _scope_label(result: TimelineSearchResult) -> str:
    if result.date_range:
        if result.date_range.start_date == result.date_range.end_date:
            return _japanese_date(result.date_range.start_date.isoformat())
        return f"{result.date_range.start_iso} から {result.date_range.end_iso}"
    return "全期間"


def _japanese_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return f"{parsed.year}年{parsed.month}月{parsed.day}日"


def _time_label(value: str | None) -> str:
    if not value:
        return "時刻不明"
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value[:16]


def _event_time_range(event: dict[str, Any]) -> str:
    start = event.get("start_time")
    end = event.get("end_time")
    if start and end and start != end:
        return f"{start[:5]}〜{end[:5]}"
    if start:
        return start[:5]
    return "時刻不明"


def _sorted_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda event: ((event.get("date") or ""), (event.get("start_time") or ""), event.get("id") or ""))


def _event_evidence_count(events: list[dict[str, Any]]) -> int:
    return sum(int(event.get("event_evidence_count") or 0) for event in events)


def _gps_photo_count(media_items: list[dict[str, Any]]) -> int:
    return sum(1 for item in media_items if _has_gps(item))


def _has_gps(item: dict[str, Any]) -> bool:
    return item.get("gps_lat") is not None and item.get("gps_lon") is not None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
