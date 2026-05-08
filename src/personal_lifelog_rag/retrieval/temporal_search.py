"""Local temporal search over the SQLite repository."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from personal_lifelog_rag.retrieval.date_parser import DateRange


@dataclass(frozen=True)
class TimelineSearchResult:
    question: str
    date_range: DateRange | None
    keyword: str | None
    events: list[dict[str, Any]]
    media_items: list[dict[str, Any]]
    line_messages: list[dict[str, Any]]
    timeline_items: list[dict[str, Any]] = field(default_factory=list)


def search_timeline(
    repository,
    question: str,
    *,
    date_range: DateRange | None = None,
    keyword: str | None = None,
    limit: int = 50,
    include_hidden: bool = False,
) -> TimelineSearchResult:
    extracted_keyword = keyword or extract_keyword(question)
    start_date = date_range.start_iso if date_range else None
    end_date = date_range.end_iso if date_range else None

    events = repository.list_events(
        start_date=start_date,
        end_date=end_date,
        keyword=extracted_keyword,
        limit=limit,
        include_hidden=include_hidden,
    )
    media_items = repository.list_media_items(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    line_messages = repository.list_line_messages(
        start_date=start_date,
        end_date=end_date,
        keyword=extracted_keyword,
        limit=limit,
    )

    return TimelineSearchResult(
        question=question,
        date_range=date_range,
        keyword=extracted_keyword,
        events=events,
        media_items=media_items,
        line_messages=line_messages,
        timeline_items=build_timeline_items(line_messages=line_messages, media_items=media_items),
    )


def build_timeline_items(
    *,
    line_messages: list[dict[str, Any]],
    media_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in line_messages:
        items.append(
            {
                "kind": "line_message",
                "at": message.get("sent_at"),
                "record": message,
            }
        )
    for media_item in media_items:
        items.append(
            {
                "kind": "media_item",
                "at": media_item.get("captured_at") or media_item.get("fallback_captured_at"),
                "record": media_item,
            }
        )
    return sorted(items, key=lambda item: ((item.get("at") or ""), item["kind"]))


def extract_keyword(question: str) -> str | None:
    cleaned = question.strip()
    cleaned = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日", " ", cleaned)
    cleaned = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", " ", cleaned)
    cleaned = re.sub(r"(今日|昨日|去年の夏|去年)", " ", cleaned)

    for pattern in (
        r"(.+?)を食べた",
        r"(.+?)に行った",
        r"(.+?)さんと",
    ):
        match = re.search(pattern, cleaned)
        if match:
            keyword = _strip_noise(match.group(1))
            return keyword or None

    for phrase in ("何していた", "何してた", "いつ", "行ったのは", "まとめて", "この日に", "撮った写真", "LINEの会話"):
        cleaned = cleaned.replace(phrase, " ")
    cleaned = _strip_noise(cleaned)
    if cleaned in {"は", "に", "で", "を", "の", "日"}:
        return None
    return cleaned or None


def _strip_noise(value: str) -> str:
    cleaned = value.strip(" ?？。　")
    cleaned = re.sub(r"^[はにでをのが]+", "", cleaned)
    cleaned = re.sub(r"[はにでをのが]+$", "", cleaned)
    return cleaned.strip(" ?？。　")
