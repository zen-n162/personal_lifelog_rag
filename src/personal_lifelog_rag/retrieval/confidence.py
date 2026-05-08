"""Conservative confidence labels for local lifelog answers."""

from __future__ import annotations

import json
import re
from typing import Any

from personal_lifelog_rag.retrieval.temporal_search import TimelineSearchResult


CONFIDENCE_UNKNOWN = "不明"
CONFIDENCE_LOW = "低"
CONFIDENCE_MEDIUM = "中"
CONFIDENCE_HIGH = "高"

LOCATION_WORDS = {
    "新宿",
    "渋谷",
    "池袋",
    "東京",
    "横浜",
    "大阪",
    "京都",
    "名古屋",
    "東口",
    "西口",
    "南口",
    "北口",
}
LOCATION_SUFFIX_RE = re.compile(
    r"([一-龥ァ-ヶA-Za-z0-9ー]{1,16}(?:駅|口|公園|空港|ホテル|店|カフェ|レストラン|神社|寺|港|海|山|温泉|美術館|博物館|大学|会場))"
)
ACTIVITY_CATEGORIES = {
    "meeting": ("待ち合わせ", "待って", "待ってる", "合流", "集合", "着く", "到着", "駅", "東口", "西口"),
    "food": ("ご飯", "食事", "ラーメン", "ランチ", "カフェ", "食べ", "おいしかった", "飲み"),
    "call": ("通話", "電話", "不在着信", "着信"),
    "work": ("仕事", "会議", "打ち合わせ", "出勤"),
    "travel": ("旅行", "観光", "移動", "散歩"),
    "shopping": ("買い物", "購入"),
    "entertainment": ("映画", "ライブ", "展示", "美術館", "博物館"),
}
CAUTIOUS_ACTIVITY_MARKERS = ("可能性", "候補", "断定")


def calculate_confidence(result: TimelineSearchResult) -> dict[str, str]:
    """Return user-facing confidence labels for date/contact/place/activity.

    The activity label is intentionally conservative: event confidence, photo
    count, GPS count, and LINE volume do not make it high by themselves.
    """

    line_count = len(result.line_messages)
    photo_count = len(result.media_items)
    event_count = len(result.events)
    record_count = line_count + photo_count + event_count
    if record_count == 0:
        return {
            "date": CONFIDENCE_UNKNOWN,
            "contact": CONFIDENCE_UNKNOWN,
            "place": CONFIDENCE_UNKNOWN,
            "activity": CONFIDENCE_UNKNOWN,
        }

    return {
        "date": _date_confidence(line_count=line_count, photo_count=photo_count, event_count=event_count),
        "contact": _contact_confidence(result.line_messages),
        "place": _place_confidence(result),
        "activity": _activity_confidence(result),
    }


def _date_confidence(*, line_count: int, photo_count: int, event_count: int) -> str:
    if event_count or line_count + photo_count >= 3:
        return CONFIDENCE_HIGH
    if line_count + photo_count:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_UNKNOWN


def _contact_confidence(line_messages: list[dict[str, Any]]) -> str:
    if not line_messages:
        return CONFIDENCE_UNKNOWN
    senders = {
        str(message.get("sender") or message.get("sender_name"))
        for message in line_messages
        if message.get("sender") or message.get("sender_name")
    }
    if senders and len(line_messages) >= 2:
        return CONFIDENCE_HIGH
    if senders:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _place_confidence(result: TimelineSearchResult) -> str:
    gps_count = sum(1 for item in result.media_items if _has_gps(item))
    named_places = _named_places(result)
    photo_count = len(result.media_items)

    if named_places and gps_count:
        return CONFIDENCE_HIGH
    if named_places:
        return CONFIDENCE_MEDIUM
    if gps_count:
        return CONFIDENCE_MEDIUM
    if photo_count:
        return CONFIDENCE_LOW
    return CONFIDENCE_UNKNOWN


def _activity_confidence(result: TimelineSearchResult) -> str:
    line_text = _line_text(result.line_messages)
    event_text = _event_text(result.events)
    media_content_text = _media_content_text(result.media_items)
    line_categories, line_occurrences = _activity_categories(line_text)
    event_categories, _event_occurrences = _activity_categories(event_text)
    media_categories, media_occurrences = _activity_categories(media_content_text)
    has_media_content = bool(media_content_text.strip())
    has_activity_signal = bool(line_categories or event_categories or media_categories)

    if has_media_content and media_categories and (line_categories or event_categories or len(media_categories) >= 2):
        return CONFIDENCE_HIGH
    if _has_strong_event_activity(result.events) and line_occurrences >= 3 and len(line_categories) >= 2:
        return CONFIDENCE_HIGH
    if has_media_content and media_occurrences:
        return CONFIDENCE_MEDIUM
    if has_activity_signal:
        return CONFIDENCE_MEDIUM
    if len(result.line_messages) + len(result.media_items) + len(result.events):
        return CONFIDENCE_LOW
    return CONFIDENCE_UNKNOWN


def _has_strong_event_activity(events: list[dict[str, Any]]) -> bool:
    for event in events:
        text = " ".join(str(event.get(key) or "") for key in ("title", "summary"))
        if not text or any(marker in text for marker in CAUTIOUS_ACTIVITY_MARKERS):
            continue
        categories, occurrences = _activity_categories(text)
        if categories and occurrences >= 2:
            return True
    return False


def _activity_categories(text: str) -> tuple[set[str], int]:
    categories: set[str] = set()
    occurrences = 0
    for category, words in ACTIVITY_CATEGORIES.items():
        for word in words:
            count = text.count(word)
            if count:
                categories.add(category)
                occurrences += count
    return categories, occurrences


def _named_places(result: TimelineSearchResult) -> set[str]:
    text_parts: list[str] = []
    for event in result.events:
        if event.get("location_name"):
            text_parts.append(str(event["location_name"]))
        text_parts.append(str(event.get("title") or ""))
        text_parts.append(str(event.get("summary") or ""))
    for message in result.line_messages:
        text_parts.append(str(message.get("text") or message.get("message_text") or ""))
    text_parts.append(_media_content_text(result.media_items))
    text = "\n".join(text_parts)
    found = {word for word in LOCATION_WORDS if word in text}
    found.update(match.group(1) for match in LOCATION_SUFFIX_RE.finditer(text))
    return found


def _line_text(line_messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("text") or message.get("message_text") or "") for message in line_messages)


def _event_text(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        parts.append(str(event.get("title") or ""))
        parts.append(str(event.get("summary") or ""))
        parts.append(str(event.get("location_name") or ""))
    return "\n".join(parts)


def _media_content_text(media_items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in media_items:
        parts.extend(_media_content_parts(item))
    return "\n".join(parts)


def _media_content_parts(item: dict[str, Any]) -> list[str]:
    parts = [
        str(item.get("caption") or ""),
        str(item.get("ocr_text") or ""),
    ]
    analysis = item.get("analysis_json")
    if isinstance(analysis, str) and analysis.strip():
        parts.append(_analysis_json_text(analysis))
    elif isinstance(analysis, dict):
        parts.append(_analysis_mapping_text(analysis))
    return [part for part in parts if part and part != "未解析"]


def _analysis_json_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict):
        return _analysis_mapping_text(payload)
    return str(payload)


def _analysis_mapping_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("scene", "objects", "possible_activity", "text_in_image"):
        value = payload.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return "\n".join(parts)


def _has_gps(item: dict[str, Any]) -> bool:
    return item.get("gps_lat") is not None and item.get("gps_lon") is not None

