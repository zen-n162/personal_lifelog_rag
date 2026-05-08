"""SQLite-backed vector embedding build and search."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from personal_lifelog_rag.embeddings.adapter import EmbeddingAdapter


@dataclass(frozen=True)
class EmbeddingBuildReport:
    model_name: str
    line_messages_seen: int
    media_items_seen: int
    embedded: int


@dataclass(frozen=True)
class SemanticSearchResult:
    source_type: str
    source_id: str
    date: str | None
    score: float
    text: str
    record: dict[str, Any]


def build_embeddings(repository, adapter: EmbeddingAdapter) -> EmbeddingBuildReport:
    """Embed line messages and available media text into SQLite."""

    sources = repository.embedding_sources()
    texts = [source["text"] for source in sources]
    if not texts:
        return EmbeddingBuildReport(
            model_name=adapter.model_name,
            line_messages_seen=0,
            media_items_seen=0,
            embedded=0,
        )

    vectors = adapter.embed_texts(texts)
    rows = []
    for source, vector in zip(sources, vectors, strict=True):
        rows.append(
            {
                **source,
                "embedding_json": json.dumps(vector, separators=(",", ":")),
                "embedding_model": adapter.model_name,
                "embedding_dim": len(vector),
                "content_hash": _content_hash(source["text"]),
            }
        )
    embedded = repository.upsert_embeddings(rows)
    return EmbeddingBuildReport(
        model_name=adapter.model_name,
        line_messages_seen=sum(1 for source in sources if source["source_type"] == "line_message"),
        media_items_seen=sum(1 for source in sources if source["source_type"] == "media_item"),
        embedded=embedded,
    )


def semantic_search(
    repository,
    adapter: EmbeddingAdapter,
    query: str,
    *,
    limit: int = 5,
) -> list[SemanticSearchResult]:
    """Return nearest local records for a query using cosine similarity."""

    query_vector = adapter.embed_texts([query])[0]
    rows = repository.list_embeddings(model_name=adapter.model_name)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        vector = json.loads(row["embedding_json"])
        score = _cosine_similarity(query_vector, vector)
        scored.append((score, row))

    results: list[SemanticSearchResult] = []
    for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
        record = repository.get_embedding_record(row["source_type"], row["source_id"]) or {}
        results.append(
            SemanticSearchResult(
                source_type=row["source_type"],
                source_id=row["source_id"],
                date=_record_date(record, row["source_type"]),
                score=score,
                text=row["text"],
                record=record,
            )
        )
    return results


def format_search_results(results: list[SemanticSearchResult]) -> str:
    if not results:
        return "候補は見つかりませんでした。build-embeddings を先に実行してください。"

    lines = ["検索候補:"]
    for result in results:
        date_text = result.date or "date unknown"
        preview = " ".join(result.text.split())
        if len(preview) > 100:
            preview = preview[:100] + "..."
        lines.append(
            f"- {date_text} score={result.score:.3f} "
            f"{result.source_type}: {preview}"
        )
    return "\n".join(lines)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_date(record: dict[str, Any], source_type: str) -> str | None:
    if source_type == "line_message":
        sent_at = record.get("sent_at")
        return sent_at[:10] if sent_at else None
    if source_type == "media_item":
        captured_at = record.get("captured_at") or record.get("fallback_captured_at")
        return captured_at[:10] if captured_at else None
    return None
