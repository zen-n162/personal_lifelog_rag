"""Rule-based query intent detection for local-only routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

from personal_lifelog_rag.retrieval.query_entities import (
    CALL_TERMS,
    FOOD_TERMS,
    PHOTO_TERMS,
    extract_query_entities,
    is_summary_query,
)


QueryIntent = Literal[
    "date_qa",
    "place_visit",
    "food_activity",
    "call_activity",
    "topic_mention",
    "person_interaction",
    "photo_activity",
    "event_summary",
    "time_range_summary",
    "location_summary",
    "unknown",
    "generic",
]

SEARCH_QUERY_INTENTS = {"place_visit", "food_activity", "call_activity", "topic_mention", "generic"}
VALID_QUERY_INTENTS = {
    "date_qa",
    "place_visit",
    "food_activity",
    "call_activity",
    "topic_mention",
    "person_interaction",
    "photo_activity",
    "event_summary",
    "time_range_summary",
    "location_summary",
    "unknown",
    "generic",
}

PLACE_HINTS = ("行った", "行く", "着いた", "着く", "到着", "どこ", "場所", "駅", "会った", "待ち合わせ")
FOOD_HINTS = ("ご飯", "食事", "食べ", "ラーメン", "カフェ", "ランチ", "夜ご飯", "昼ご飯", "店")
CALL_HINTS = ("通話", "電話", "不在着信", "着信", "話した")
MENTION_HINTS = ("アルバム", "写真送って", "話題", "言ってた")
DATE_QA_HINTS = ("何して", "何があった", "なにして", "この日")
TIME_RANGE_HINTS = ("去年", "今年", "今月", "先月", "最近", "この1か月", "この一か月")
PHOTO_ACTIVITY_HINTS = ("写真が多", "写真を撮", "撮った日", "GPS付き写真", "写真ある")
LOCATION_SUMMARY_HINTS = ("よく行った場所", "行った場所", "場所をまとめ", "場所まとめ")
PERSON_ACTIVITY_HINTS = ("会った", "出かけた", "会う")


@dataclass(frozen=True)
class QueryIntentResult:
    intent: QueryIntent
    confidence: float
    normalized_query: str
    entities: dict[str, Any]
    routing_hint: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_query_intent(query: str, *, override: str | None = None) -> QueryIntent:
    """Infer a small local-only search intent."""

    if override:
        if override not in SEARCH_QUERY_INTENTS:
            raise ValueError(f"Unsupported search intent: {override}")
        return override  # type: ignore[return-value]

    text = query.strip()
    if any(word in text for word in CALL_HINTS):
        return "call_activity"
    if any(word in text for word in FOOD_HINTS):
        return "food_activity"
    if any(word in text for word in PLACE_HINTS):
        return "place_visit"
    if any(word in text for word in MENTION_HINTS):
        return "topic_mention"
    return "generic"


def classify_query_intent(
    query: str,
    *,
    today: date | None = None,
) -> QueryIntentResult:
    """Classify a natural-language query and produce a local routing hint."""

    normalized = " ".join(query.strip().split())
    if not normalized:
        return _result(
            intent="unknown",
            confidence=0.0,
            normalized_query="",
            entities={},
            routing_hint="unsupported",
            reasons=["空のクエリです"],
        )

    entities = extract_query_entities(normalized, today=today)
    reasons: list[str] = []
    has_single_date = "date" in entities
    has_range = "date_from" in entities and "date_to" in entities and not has_single_date

    if has_single_date and any(hint in normalized for hint in DATE_QA_HINTS):
        reasons.append("日付と「何して」系の表現を検出")
        return _result("date_qa", 0.92, normalized, entities, "ask", reasons)

    if has_range and (is_summary_query(normalized) or any(hint in normalized for hint in DATE_QA_HINTS)):
        reasons.append("期間とsummary/何して系の表現を検出")
        return _result("time_range_summary", 0.86, normalized, entities, "event_summary", reasons)

    if any(hint in normalized for hint in LOCATION_SUMMARY_HINTS):
        reasons.append("場所の集計・要約表現を検出")
        return _result("location_summary", 0.78, normalized, entities, "event_summary", reasons)

    if any(hint in normalized for hint in PHOTO_ACTIVITY_HINTS) or (
        any(term in normalized for term in PHOTO_TERMS) and any(word in normalized for word in ("多", "撮", "ある"))
    ):
        reasons.append("写真/GPSに関する検索表現を検出")
        return _result("photo_activity", 0.8, normalized, entities, "photo_activity", reasons)

    if any(term in normalized for term in CALL_TERMS):
        if entities.get("person"):
            reasons.append(f"人物候補「{entities['person']}」と通話表現を検出")
        else:
            reasons.append("通話表現を検出")
        return _result("call_activity", 0.9, normalized, entities, "call_search", reasons)

    if any(term in normalized for term in FOOD_TERMS):
        reasons.append("食事・カフェ系の語を検出")
        return _result("food_activity", 0.86, normalized, entities, "search", reasons)

    if entities.get("person") and any(hint in normalized for hint in PERSON_ACTIVITY_HINTS):
        reasons.append(f"人物候補「{entities['person']}」と交流表現を検出")
        return _result("person_interaction", 0.78, normalized, entities, "search", reasons)

    if "話" in normalized or "話題" in normalized or any(hint in normalized for hint in MENTION_HINTS):
        reasons.append("話題・言及検索の表現を検出")
        return _result("topic_mention", 0.82, normalized, entities, "search", reasons)

    if any(hint in normalized for hint in PLACE_HINTS) and entities.get("place"):
        reasons.append(f"地名候補「{entities['place']}」と移動/滞在表現を検出")
        return _result("place_visit", 0.84, normalized, entities, "search", reasons)

    if is_summary_query(normalized):
        reasons.append("イベント要約表現を検出")
        return _result("event_summary", 0.65, normalized, entities, "event_summary", reasons)

    if has_single_date:
        reasons.append("日付を検出")
        return _result("date_qa", 0.72, normalized, entities, "ask", reasons)

    reasons.append("対応済みIntentを十分な確度で判定できませんでした")
    return _result("unknown", 0.25, normalized, entities, "unsupported", reasons)


def _result(
    intent: QueryIntent,
    confidence: float,
    normalized_query: str,
    entities: dict[str, Any],
    routing_hint: str,
    reasons: list[str],
) -> QueryIntentResult:
    return QueryIntentResult(
        intent=intent,
        confidence=round(confidence, 2),
        normalized_query=normalized_query,
        entities=entities,
        routing_hint=routing_hint,
        reasons=reasons,
    )
