"""Route classified natural-language queries to local retrieval paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.local_search import (
    LocalSearchOptions,
    format_local_search_report,
    local_text_search,
)
from personal_lifelog_rag.retrieval.query_intent import QueryIntentResult, classify_query_intent
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.line.call_index import format_search_calls_report, search_calls
from personal_lifelog_rag.timeline.event_reports import format_event_list, list_events_report


@dataclass(frozen=True)
class RoutedQueryResult:
    query: str
    intent: str
    intent_confidence: float
    entities: dict[str, Any]
    routing: str
    answer: str
    results: list[dict[str, Any]]
    intent_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_query_intent(result: QueryIntentResult) -> str:
    lines = [
        "Query intent",
        "",
        f"query: {result.normalized_query}",
        f"intent: {result.intent}",
        f"confidence: {result.confidence:.2f}",
        f"routing: {result.routing_hint}",
        "entities:",
    ]
    if result.entities:
        for key, value in result.entities.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.append("reasons:")
    for reason in result.reasons:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def route_query(
    repository,
    query: str,
    *,
    today: date | None = None,
    limit: int = 5,
    include_hidden: bool = False,
) -> RoutedQueryResult:
    intent_result = classify_query_intent(query, today=today)
    intent = intent_result.intent

    if intent == "date_qa":
        date_range = parse_date_query(query, today=today)
        result = search_timeline(repository, query, date_range=date_range, include_hidden=include_hidden)
        return _routed(
            intent_result,
            routing="ask",
            answer=build_answer(query, result),
            results=[
                {
                    "events": len(result.events),
                    "line_messages": len(result.line_messages),
                    "media_items": len(result.media_items),
                }
            ],
        )

    if intent in {"place_visit", "food_activity", "topic_mention", "person_interaction"}:
        search_intent = _search_intent_for(intent)
        search_query = _search_query_for(intent_result)
        report = local_text_search(
            repository,
            LocalSearchOptions(
                query=search_query,
                date_from=intent_result.entities.get("date_from"),
                date_to=intent_result.entities.get("date_to"),
                limit=limit,
                mode="all",
                intent=search_intent,
                include_hidden=include_hidden,
            ),
        )
        return _routed(
            intent_result,
            routing="search",
            answer=format_local_search_report(report),
            results=report["results"],
        )

    if intent == "call_activity":
        call_status = intent_result.entities.get("call_status")
        if call_status in {"missed", "unanswered", "canceled"}:
            report = search_calls(
                repository,
                statuses=[call_status],
                start_date=intent_result.entities.get("date_from"),
                end_date=intent_result.entities.get("date_to"),
                limit=limit,
            )
            return _routed(
                intent_result,
                routing="search-calls",
                answer=format_search_calls_report(report),
                results=report["results"],
            )
        report = local_text_search(
            repository,
            LocalSearchOptions(
                query="通話",
                date_from=intent_result.entities.get("date_from"),
                date_to=intent_result.entities.get("date_to"),
                limit=limit,
                mode="all",
                intent="call_activity",
                include_hidden=include_hidden,
            ),
        )
        return _routed(
            intent_result,
            routing="search",
            answer=format_local_search_report(report),
            results=report["results"],
        )

    if intent in {"time_range_summary", "event_summary", "location_summary"}:
        rows = list_events_report(
            repository,
            start_date=intent_result.entities.get("date_from"),
            end_date=intent_result.entities.get("date_to"),
            with_evidence=False,
            include_hidden=include_hidden,
        )[:limit]
        answer = format_event_list(rows, with_evidence=False) if rows else _no_records_message(intent_result)
        return _routed(
            intent_result,
            routing="list-events",
            answer=answer,
            results=rows,
        )

    if intent == "photo_activity":
        rows = _photo_activity_results(
            repository,
            start_date=intent_result.entities.get("date_from"),
            end_date=intent_result.entities.get("date_to"),
            limit=limit,
        )
        return _routed(
            intent_result,
            routing="photo-activity",
            answer=_format_photo_activity(rows),
            results=rows,
        )

    return _routed(
        intent_result,
        routing="unsupported",
        answer=_unsupported_message(query),
        results=[],
    )


def format_routed_query_result(result: RoutedQueryResult) -> str:
    return "\n".join(
        [
            f"質問: {result.query}",
            f"意図: {result.intent} (confidence={result.intent_confidence:.2f})",
            f"routing: {result.routing}",
            "",
            result.answer,
        ]
    )


def _routed(
    intent_result: QueryIntentResult,
    *,
    routing: str,
    answer: str,
    results: list[dict[str, Any]],
) -> RoutedQueryResult:
    return RoutedQueryResult(
        query=intent_result.normalized_query,
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        entities=intent_result.entities,
        routing=routing,
        answer=answer,
        results=results,
        intent_reasons=intent_result.reasons,
    )


def _search_intent_for(intent: str) -> str:
    if intent == "food_activity":
        return "food_activity"
    if intent == "place_visit":
        return "place_visit"
    if intent == "topic_mention":
        return "topic_mention"
    return "generic"


def _search_query_for(intent_result: QueryIntentResult) -> str:
    entities = intent_result.entities
    if intent_result.intent == "place_visit" and entities.get("place"):
        return str(entities["place"])
    if intent_result.intent == "topic_mention" and entities.get("topic"):
        return str(entities["topic"])
    if intent_result.intent == "person_interaction" and entities.get("person"):
        return str(entities["person"])
    if intent_result.intent == "food_activity":
        terms = [str(term) for term in (entities.get("food_terms") or [])]
        for term in terms:
            if term not in {"食べ", "食事", "店"}:
                return term
        return terms[0] if terms else "ご飯"
    raw_terms = entities.get("raw_terms") or []
    return str(raw_terms[0]) if raw_terms else intent_result.normalized_query


def _photo_activity_results(
    repository,
    *,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    media = repository.list_media_items(start_date=start_date, end_date=end_date, limit=1_000_000)
    by_date: dict[str, dict[str, Any]] = {}
    for item in media:
        timestamp = item.get("captured_at") or item.get("fallback_captured_at")
        if not timestamp:
            continue
        key = str(timestamp)[:10]
        row = by_date.setdefault(key, {"date": key, "photos": 0, "gps_photos": 0})
        row["photos"] += 1
        if item.get("gps_lat") is not None and item.get("gps_lon") is not None:
            row["gps_photos"] += 1
    return sorted(by_date.values(), key=lambda row: (-row["photos"], row["date"]))[: max(limit, 0)]


def _format_photo_activity(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "写真の記録は見つかりませんでした。"
    lines = ["写真が多かった日:"]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}. {row['date']} 写真={row['photos']} GPS付き={row['gps_photos']}")
    return "\n".join(lines)


def _no_records_message(intent_result: QueryIntentResult) -> str:
    range_label = intent_result.entities.get("date_from") or ""
    if intent_result.entities.get("date_to") and intent_result.entities.get("date_to") != range_label:
        range_label = f"{range_label}..{intent_result.entities['date_to']}"
    return f"対象期間のイベントは見つかりませんでした: {range_label or intent_result.normalized_query}"


def _unsupported_message(query: str) -> str:
    return "\n".join(
        [
            "この質問はまだ十分な確度で分類できませんでした。",
            "対応しやすい聞き方の例:",
            "- 2024年12月24日は何していた？",
            "- 新宿に行ったのはいつ？",
            "- ご飯を食べた日は？",
            "- 通話した日は？",
            "- アルバムの話をしたのはいつ？",
            f"元の質問: {query}",
        ]
    )
