"""Rule-based expansion for local visual/VLM SQL fallback search."""

from __future__ import annotations

import re


VISUAL_QUERY_SYNONYMS: dict[str, list[str]] = {
    "ご飯": [
        "ご飯",
        "meal",
        "food",
        "rice",
        "dining",
        "dish",
        "plate",
        "bowl",
        "cafe",
        "restaurant",
        "meal_possible",
        "food_possible",
        "rice_possible",
        "fried_food",
        "fried_food_possible",
        "salad_possible",
        "soup_possible",
        "soup",
        "salad",
        "crispy_fried_items",
        "dark_sauce_dish",
        "dining_area_possible",
        "rice_with_green_onions",
        "tray",
        "food_cues",
    ],
    "食事": [
        "食事",
        "meal",
        "food",
        "dining",
        "restaurant",
        "cafe",
        "meal_possible",
        "food_possible",
        "dining_area_possible",
    ],
    "料理": ["料理", "meal", "food", "dish", "plate", "bowl", "meal_possible", "food_possible"],
    "カフェ": ["カフェ", "cafe", "cafe_possible", "coffee", "glass_cup", "wooden_table", "dining_area_possible"],
    "ラーメン": ["ラーメン", "ramen", "noodles", "soup", "bowl", "ramen_possible", "noodle_possible"],
    "車": ["車", "車内", "vehicle", "car", "vehicle_interior_possible", "car_seat_possible", "headrest"],
    "車内": ["車内", "vehicle", "car", "vehicle_interior_possible", "car_seat_possible", "headrest"],
    "室内": ["室内", "indoor", "indoor_possible", "indoor_room_possible", "living_room_possible"],
    "アルバム": ["アルバム", "photo_album", "media content", "handwritten labels", "photo collage"],
    "駅": ["駅", "station", "station_possible", "train", "platform", "sign"],
    "新宿": ["新宿", "shinjuku", "station", "city", "urban", "sign", "train", "street", "station_possible", "city_possible"],
    # Keep photo generic terms narrow; expanding to "photo" or "image" makes
    # food queries match generic captions rather than food evidence.
    "写真": ["写真"],
}


def expand_visual_query_terms(query: str) -> list[str]:
    """Return privacy-safe local search terms for visual SQL fallback."""

    normalized = (query or "").strip()
    terms: list[str] = []
    _append_unique(terms, normalized)
    for token in _rough_tokens(normalized):
        _append_unique(terms, token)
    for trigger, synonyms in VISUAL_QUERY_SYNONYMS.items():
        if trigger and trigger in normalized:
            for synonym in synonyms:
                _append_unique(terms, synonym)
    if any(trigger in normalized for trigger in ("食べ", "食っ", "食う", "食べた")):
        for synonym in VISUAL_QUERY_SYNONYMS["ご飯"] + VISUAL_QUERY_SYNONYMS["食事"]:
            _append_unique(terms, synonym)
    return [term for term in terms if term]


def visual_query_terms_for_display(query: str, *, limit: int = 20) -> list[str]:
    """Compact expansion preview for reports and tests."""

    return expand_visual_query_terms(query)[:limit]


def _rough_tokens(text: str) -> list[str]:
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    jp_tokens = [token for token in ("ご飯", "食事", "料理", "カフェ", "ラーメン", "車内", "車", "室内", "アルバム", "駅", "新宿", "写真") if token in text]
    return jp_tokens + ascii_tokens


def _append_unique(items: list[str], value: str) -> None:
    value = str(value or "").strip()
    if value and value not in items:
        items.append(value)
