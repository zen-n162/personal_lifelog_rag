"""Small local entity extraction helpers for natural-language queries."""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from personal_lifelog_rag.retrieval.date_parser import DateRange, parse_date_query


FOOD_TERMS = ("ご飯", "食事", "食べ", "ラーメン", "カフェ", "ランチ", "夜ご飯", "昼ご飯", "おいしかった", "店")
CALL_TERMS = ("通話", "電話", "長電話", "不在着信", "着信", "応答", "キャンセル")
PHOTO_TERMS = ("写真", "GPS", "撮った", "撮影")
SUMMARY_TERMS = ("まとめ", "まとめて", "主な", "思い出", "出来事", "イベント")
VISIT_STOPWORDS = {
    "いつ",
    "どこ",
    "何",
    "なに",
    "記録",
    "探して",
    "行った",
    "行く",
    "いた",
    "いる",
    "日",
    "場所",
    "周辺",
}


def extract_query_entities(query: str, *, today: date | None = None) -> dict[str, Any]:
    text = query.strip()
    date_range = parse_date_query(text, today=today)
    entities: dict[str, Any] = {
        "raw_terms": extract_raw_terms(text),
    }
    if date_range:
        entities.update(_date_entities(date_range))

    person = extract_person(text)
    if person:
        entities["person"] = person

    place = extract_place(text)
    if place:
        entities["place"] = place

    topic = extract_topic(text)
    if topic:
        entities["topic"] = topic

    food_terms = [term for term in FOOD_TERMS if term in text]
    if food_terms:
        entities["food_terms"] = food_terms
        entities["activity"] = "food"

    call_status = extract_call_status(text)
    if call_status:
        entities["call_status"] = call_status
        entities["activity"] = "call"

    if any(term in text for term in PHOTO_TERMS):
        entities["activity"] = "photo"

    return entities


def extract_raw_terms(query: str) -> list[str]:
    cleaned = re.sub(r"[?？。!！]", " ", query)
    cleaned = re.sub(r"(のはいつ|はいつ|いつ|ですか|でしたか|教えて|探して|まとめて)", " ", cleaned)
    parts = re.split(r"[\s,、/]+|(?:で|に|を|は|が|と)", cleaned)
    terms: list[str] = []
    for value in parts:
        term = value.strip(" 　")
        if term and term not in terms:
            terms.append(term)
    return terms


def extract_person(query: str) -> str | None:
    match = re.search(r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,20})(?:さん)?と", query)
    if not match:
        return None
    person = match.group(1).strip()
    if person in {"誰", "どこ", "いつ", "何"}:
        return None
    return person


def extract_place(query: str) -> str | None:
    patterns = [
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9ー]{1,24})(?:に|へ)(?:行った|行く|行った記録|いた|いる|着いた|向かった)",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9ー]{1,24})周辺",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9ー]{1,24})(?:駅|公園|空港|店|カフェ|大学)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            candidate = _clean_entity(match.group(1))
            if candidate and candidate not in VISIT_STOPWORDS:
                return candidate
    return None


def extract_topic(query: str) -> str | None:
    match = re.search(r"(.+?)の(?:話|話題)", query)
    if match:
        return _clean_entity(match.group(1))
    match = re.search(r"([一-龥ぁ-んァ-ヶA-Za-z0-9ー]{1,30})(?:のこと|について)", query)
    if match:
        return _clean_entity(match.group(1))
    for hint in ("アルバム", "研究", "旅行"):
        if hint in query:
            return hint
    if "話" not in query and "話題" not in query:
        return None
    terms = [term for term in extract_raw_terms(query) if term not in {"話", "話題", "した", "していた"}]
    return _clean_entity(terms[0]) if terms else None


def extract_call_status(query: str) -> str | None:
    if "不在着信" in query:
        return "missed"
    if "応答" in query:
        return "unanswered"
    if "キャンセル" in query:
        return "canceled"
    if "長電話" in query or "通話" in query or "電話" in query:
        return "completed"
    return None


def is_summary_query(query: str) -> bool:
    return any(term in query for term in SUMMARY_TERMS)


def _date_entities(date_range: DateRange) -> dict[str, str]:
    if date_range.start_date == date_range.end_date:
        return {"date": date_range.start_iso, "date_from": date_range.start_iso, "date_to": date_range.end_iso}
    return {"date_from": date_range.start_iso, "date_to": date_range.end_iso}


def _clean_entity(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[「」『』\"' 　?？。!！]", "", value)
    cleaned = re.sub(r"(いつ|どこ|何|なに|行った|行く|した|していた|日は|日)$", "", cleaned)
    return cleaned or None
