"""Route classified natural-language queries to local retrieval paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from personal_lifelog_rag.embeddings.base import MultimodalEmbeddingEngine
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.local_search import (
    LocalSearchOptions,
    format_local_search_report,
    local_text_search,
)
from personal_lifelog_rag.embeddings.multimodal_search import format_multimodal_search, multimodal_search
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.line.person_links import search_person_line_days
from personal_lifelog_rag.retrieval.monthly_summary import build_monthly_summary_report, format_monthly_summary
from personal_lifelog_rag.retrieval.person_place_qa import route_person_place_query
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
    metadata: dict[str, Any] | None = None

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
    multimodal_config: dict[str, Any] | None = None,
    multimodal_engine: MultimodalEmbeddingEngine | None = None,
    public_mode: bool = False,
) -> RoutedQueryResult:
    intent_result = classify_query_intent(query, today=today)
    intent = intent_result.intent

    person_place_payload = route_person_place_query(
        repository,
        query,
        intent_result.entities,
        limit=limit,
        include_hidden=include_hidden,
        public_mode=public_mode,
    )
    if person_place_payload is not None:
        return _routed(
            intent_result,
            routing=str(person_place_payload["routing"]),
            answer=str(person_place_payload["answer"]),
            results=person_place_payload["results"],
            intent_override=str(person_place_payload["intent"]),
            metadata=person_place_payload.get("metadata"),
        )

    if intent == "date_qa":
        date_range = parse_date_query(query, today=today)
        result = search_timeline(repository, query, date_range=date_range, include_hidden=include_hidden, public_mode=public_mode)
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
        if intent == "person_interaction" and _looks_like_line_person_query(query) and intent_result.entities.get("person"):
            report = search_person_line_days(repository, person_name=str(intent_result.entities["person"]), limit=limit)
            return _routed(
                intent_result,
                routing="line-speaker-search",
                answer=str(report["answer"]),
                results=report["results"],
            )
        if intent in {"place_visit", "food_activity"} and _looks_like_image_query(query):
            report = multimodal_search(
                repository,
                _multimodal_options(
                    query=search_query if search_query != "ご飯" else query,
                    date_from=intent_result.entities.get("date_from"),
                    date_to=intent_result.entities.get("date_to"),
                    limit=limit,
                    backend="hybrid",
                    include_hidden=include_hidden,
                    config=multimodal_config,
                ),
                engine=multimodal_engine,
            )
            return _routed(
                intent_result,
                routing="multimodal-search",
                answer=_format_multimodal_image_answer(query, report),
                results=report["results"],
            )
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

    if intent in {"monthly_summary", "time_range_summary"}:
        start_date = intent_result.entities.get("date_from")
        end_date = intent_result.entities.get("date_to")
        if start_date and end_date:
            report = build_monthly_summary_report(
                repository,
                start_date=str(start_date),
                end_date=str(end_date),
                include_hidden=include_hidden,
                top_days_limit=5,
                events_per_day=5,
                public=public_mode,
            )
            return _routed(
                intent_result,
                routing="monthly-summary" if intent == "monthly_summary" else "time-range-summary",
                answer=format_monthly_summary(report),
                results=[report],
            )

    if intent in {"event_summary", "location_summary"}:
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
        if _looks_like_image_query(query):
            report = multimodal_search(
                repository,
                _multimodal_options(
                    query=query,
                    date_from=intent_result.entities.get("date_from"),
                    date_to=intent_result.entities.get("date_to"),
                    limit=limit,
                    backend="hybrid",
                    include_hidden=include_hidden,
                    config=multimodal_config,
                ),
                engine=multimodal_engine,
            )
            return _routed(
                intent_result,
                routing="multimodal-search",
                answer=_format_multimodal_image_answer(query, report),
                results=report["results"],
            )
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

    if intent in {"image_search", "multimodal_image_search", "specific_food_search", "food_photo_search", "place_photo_search"}:
        report = multimodal_search(
            repository,
            _multimodal_options(
                query=query,
                date_from=intent_result.entities.get("date_from"),
                date_to=intent_result.entities.get("date_to"),
                limit=limit,
                backend="hybrid",
                include_hidden=include_hidden,
                config=multimodal_config,
            ),
            engine=multimodal_engine,
        )
        return _routed(
            intent_result,
            routing="multimodal-search",
            answer=_format_multimodal_image_answer(query, report),
            results=report["results"],
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
    intent_override: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RoutedQueryResult:
    return RoutedQueryResult(
        query=intent_result.normalized_query,
        intent=intent_override or intent_result.intent,
        intent_confidence=intent_result.confidence,
        entities=intent_result.entities,
        routing=routing,
        answer=answer,
        results=results,
        intent_reasons=intent_result.reasons,
        metadata=metadata,
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


def _multimodal_options(
    *,
    query: str,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    backend: str,
    include_hidden: bool,
    config: dict[str, Any] | None,
) -> MultimodalSearchOptions:
    config = config or {}
    return MultimodalSearchOptions(
        query=query,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        backend=backend,  # type: ignore[arg-type]
        include_hidden=include_hidden,
        engine_name=config.get("engine"),
        model_name=config.get("model_name"),
        model_path=config.get("model_path"),
        device=config.get("device", "auto"),
        dtype=config.get("dtype"),
        local_files_only=config.get("local_files_only", True),
        embedding_dim=config.get("embedding_dim"),
        batch_size=config.get("batch_size"),
    )


def _looks_like_image_query(query: str) -> bool:
    return any(term in query for term in ("写真", "写って", "画像", "撮った", "撮影"))


def _looks_like_line_person_query(query: str) -> bool:
    return any(term in query for term in ("LINE", "ライン", "話した", "話して", "やりとり"))


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


def _format_multimodal_image_answer(query: str, report: dict[str, Any]) -> str:
    if not report.get("results"):
        return format_multimodal_search(report)
    ranked_dates = _ranked_unique_dates(report["results"])
    primary_date = ranked_dates[0] if ranked_dates else "該当日"
    if len(ranked_dates) > 1:
        summary_line = f"主な候補は {', '.join(ranked_dates[:3])} です。"
    else:
        summary_line = f"主な候補は {primary_date} です。"
    lines = [
        f"質問: {query}",
        "",
        f"画像解析では、{primary_date}に「{query}」に関連する可能性がある写真が見つかりました。",
        summary_line,
        "これはQwen3-VLなどによるローカル画像解析の推定です。必要に応じて写真を確認してください。",
        "",
        "候補:",
    ]
    for row in report["results"][:5]:
        cues = row.get("food_cues") or row.get("activity_tags") or row.get("matched_terms") or []
        cue_text = ", ".join(cues[:6]) if cues else "VLM caption/tags"
        lines.append(f"- {row.get('date')} {str(row.get('captured_at') or '')[11:16]} / cues: {cue_text} / evidence_strength={row.get('evidence_strength')}")
    lines.extend(["", format_multimodal_search(report)])
    return "\n".join(lines)


def _ranked_unique_dates(results: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for row in results:
        value = str(row.get("date") or "")
        if value and value not in dates:
            dates.append(value)
    return dates


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
