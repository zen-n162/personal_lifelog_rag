"""Hybrid local multimodal search over embeddings, OCR, VLM, LINE, and events."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.base import MultimodalEmbeddingEngine
from personal_lifelog_rag.embeddings.engines import (
    get_cached_multimodal_embedding_engine,
    infer_query_engine_from_records,
)
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository, embedding_vector
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.embeddings.similarity import cosine_similarity
from personal_lifelog_rag.retrieval.evidence_strength import (
    confidence_label_for_score,
    compute_evidence_strength,
)
from personal_lifelog_rag.retrieval.local_search import extract_search_terms
from personal_lifelog_rag.retrieval.multimodal_ranker import (
    matched_fields_for_row,
    matched_terms_for_row,
    score_multimodal_components,
)
from personal_lifelog_rag.retrieval.visual_query_expansion import expand_visual_query_terms
from personal_lifelog_rag.vlm.review_service import apply_vlm_override_to_result, should_use_vlm_for_search
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import image_search


def multimodal_search(
    repository: LifelogRepository,
    options: MultimodalSearchOptions,
    *,
    engine: MultimodalEmbeddingEngine | None = None,
) -> dict[str, Any]:
    embedding_repository = MediaEmbeddingRepository(repository.db_path)
    embedding_rows = (
        [
            row
            for row in embedding_repository.list_embeddings(
                start_date=options.date_from,
                end_date=options.date_to,
                statuses=["success"],
                limit=50_000,
            )
            if not _is_fake_embedding_row(row)
            and str(row.get("status") or "") != "engine_unavailable"
        ]
        if options.backend in {"embedding", "hybrid"}
        else []
    )
    resolved_engine = engine or _query_engine(options, embedding_rows)
    embedding_status: dict[str, Any] = {"available": False, "reason": ""}
    embedding_candidates = (
        _embedding_candidates(repository, embedding_rows, options, resolved_engine, embedding_status)
        if options.backend in {"embedding", "hybrid"}
        else {}
    )
    sql_candidates = (
        _sql_candidates(repository, options)
        if options.backend in {"sql", "vlm_sql", "hybrid"}
        else {}
    )
    media_ids = set(embedding_candidates) | set(sql_candidates)
    results = []
    for media_id in media_ids:
        row = _merge_candidate(
            repository,
            media_id,
            embedding_candidates.get(media_id, {}),
            sql_candidates.get(media_id, {}),
            query=options.query,
            include_hidden=options.include_hidden,
        )
        if row:
            results.append(row)
    results.sort(key=lambda row: (-float(row["score_components"]["final_score"]), row["date"], row["media_id"]))
    limited = results[: max(options.limit, 0)]
    return {
        "query": options.query,
        "backend": options.backend,
        "embedding_engine": getattr(resolved_engine, "name", None),
        "embedding_model": getattr(resolved_engine, "model_name", None),
        "date_from": options.date_from,
        "date_to": options.date_to,
        "total": len(results),
        "results": limited,
        "embedding_status": embedding_status if options.backend in {"embedding", "hybrid"} else {},
        "caution": "画像解析とembeddingによる推定です。必要に応じて写真を確認してください。",
    }


def format_multimodal_search(report: dict[str, Any]) -> str:
    if not report["results"]:
        embedding_status = report.get("embedding_status") or {}
        if report.get("backend") == "embedding" and embedding_status.get("reason"):
            return "\n".join(
                [
                    f"Multimodal Search embedding result unavailable: {report['query']}",
                    f"- reason: {embedding_status['reason']}",
                    "- fallback: use --backend hybrid or --backend vlm_sql to search VLM/OCR text.",
                ]
            )
        return f"Multimodal Search results were not found: {report['query']}"
    lines = [
        f"Multimodal Search: {report['query']}",
        f"- backend: {report['backend']}",
        f"- embedding_engine: {report.get('embedding_engine') or ''}",
        f"- results: {report['total']}",
        "- caution: 画像解析とembeddingによる推定です。必要に応じて写真を確認してください。",
    ]
    for index, row in enumerate(report["results"], start=1):
        lines.append(
            f"{index}. {row['date']} score={row['score_components']['final_score']:.2f} "
            f"confidence={row['confidence_label']} evidence_strength={row['evidence_strength']}"
        )
        lines.append(f"   media: {row['media_id']} {row['file_name']}")
        if row.get("captured_at"):
            lines.append(f"   captured_at: {row['captured_at']}")
        if row.get("caption"):
            lines.append(f"   caption: {row['caption']}")
        if "vlm" in set(row.get("evidence_types") or []):
            lines.append("   VLM evidence: caption/tags available")
        if row.get("ocr_preview"):
            lines.append(f"   OCR: {row['ocr_preview']}")
        if row.get("food_cues"):
            lines.append(f"   food_cues: {', '.join(row['food_cues'])}")
        if row.get("matched_terms"):
            lines.append(f"   matched_terms: {', '.join(row['matched_terms'][:8])}")
        if row.get("related_event"):
            lines.append(f"   event: {row['related_event']}")
        if row.get("review_status") and row.get("review_status") != "unreviewed":
            lines.append(f"   VLM review: {row['review_status']}")
        if row.get("is_verified"):
            lines.append("   VLM review: 手動確認済み")
        if row.get("reasons"):
            lines.append("   reasons:")
            for reason in row["reasons"]:
                lines.append(f"   - {reason}")
        lines.append(f"   score_components: {json.dumps(row['score_components'], ensure_ascii=False, sort_keys=True)}")
        lines.append("   caution: embedding/VLM-only evidence is treated as weak and not definitive.")
    return "\n".join(lines)


def _query_engine(
    options: MultimodalSearchOptions,
    embedding_rows: list[dict[str, Any]],
) -> MultimodalEmbeddingEngine:
    if options.engine_name or options.model_name or options.model_path:
        return get_cached_multimodal_embedding_engine(
            options.engine_name,
            model_name=options.model_name,
            model_path=options.model_path,
            device=options.device,
            dtype=options.dtype,
            local_files_only=options.local_files_only,
            embedding_dim=options.embedding_dim,
            batch_size=options.batch_size,
        )
    return infer_query_engine_from_records(embedding_rows)


def _is_fake_embedding_row(row: dict[str, Any]) -> bool:
    return "fake" in str(row.get("embedding_model") or "").lower()


def _embedding_candidates(
    repository: LifelogRepository,
    rows: list[dict[str, Any]],
    options: MultimodalSearchOptions,
    engine: MultimodalEmbeddingEngine,
    status: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not rows:
        status.update({"available": False, "reason": "no successful non-fake media_embeddings are available"})
        return {}
    if not engine.is_available():
        reporter = getattr(engine, "availability_error", None)
        reason = reporter() if callable(reporter) else f"embedding engine '{engine.name}' is not available"
        status.update({"available": False, "reason": reason})
        return {}
    query_result = engine.embed_text(options.query)
    if query_result.status != "success" or not query_result.vector:
        status.update({"available": False, "reason": query_result.error_message or f"query embedding failed with status={query_result.status}"})
        return {}
    status.update({"available": True, "reason": ""})
    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        vector = embedding_vector(row)
        if not vector:
            continue
        score = cosine_similarity(query_result.vector, vector)
        media_id = str(row.get("media_id") or "")
        if not media_id:
            continue
        existing = candidates.get(media_id)
        if existing is None or score > existing["embedding_score"]:
            candidates[media_id] = {
                "embedding_score": round(max(0.0, min(score, 1.0)), 4),
                "embedding_type": row.get("embedding_type"),
                "embedding_model": row.get("embedding_model"),
                "embedding_row": row,
            }
    return candidates


def _sql_candidates(repository: LifelogRepository, options: MultimodalSearchOptions) -> dict[str, dict[str, Any]]:
    report = image_search(
        repository,
        ImageSearchOptions(
            query=options.query,
            date_from=options.date_from,
            date_to=options.date_to,
            limit=50_000,
            include_hidden=options.include_hidden,
        ),
    )
    candidates: dict[str, dict[str, Any]] = {}
    for row in report.get("results") or []:
        media_id = str(row.get("media_id") or "")
        if media_id:
            candidates[media_id] = {"sql_score": float(row.get("score") or 0.0), "sql_row": row, "expanded_terms": report.get("expanded_terms") or []}
    return candidates


def _merge_candidate(
    repository: LifelogRepository,
    media_id: str,
    embedding_candidate: dict[str, Any],
    sql_candidate: dict[str, Any],
    *,
    query: str,
    include_hidden: bool,
) -> dict[str, Any] | None:
    row = _media_row_from_candidates(repository, media_id, embedding_candidate, sql_candidate)
    if not row:
        return None
    row = apply_vlm_override_to_result(row)
    if not should_use_vlm_for_search(row, include_hidden=include_hidden):
        return None
    timestamp = str(row.get("captured_at") or row.get("fallback_captured_at") or "")
    date_value = timestamp[:10]
    expanded_terms = expand_visual_query_terms(query)
    matching_terms = _matched_terms(row, sql_candidate.get("expanded_terms") or expanded_terms)
    related_event = _related_event(repository, media_id, date_value, include_hidden=include_hidden)
    line_matches = _line_matches(repository, query, date_value, expanded_terms=expanded_terms)
    evidence_types = _evidence_types(row, embedding_candidate, related_event=related_event, line_matches=line_matches)
    score_components = score_multimodal_components(
        row,
        expanded_terms=expanded_terms,
        embedding_score=float(embedding_candidate.get("embedding_score") or 0.0),
        related_event=related_event,
        line_matches=line_matches,
        sql_score=float(sql_candidate.get("sql_score") or 0.0),
        matched_terms=matching_terms,
    )
    visual_match = bool(score_components.get("visual_match"))
    strength = compute_multimodal_evidence_strength(
        evidence_types=evidence_types,
        verified_event=bool((related_event or {}).get("is_verified")),
        pinned_event=bool((related_event or {}).get("is_pinned")),
        visual_match=visual_match,
    )
    safety_flags = _json_list(row.get("safety_flags_json"))
    return {
        "date": date_value,
        "media_id": media_id,
        "file_name": redact_text(row.get("file_name"), max_chars=80),
        "captured_at": timestamp,
        "thumbnail_path": row.get("thumbnail_path") or "",
        "caption": redact_text(row.get("short_caption") or row.get("caption"), max_chars=140),
        "ocr_preview": redact_text(row.get("ocr_text_redacted") or row.get("ocr_text"), max_chars=120),
        "matched_fields": _matched_fields(
            row,
            query,
            has_embedding=bool(embedding_candidate),
            expanded_terms=sql_candidate.get("expanded_terms") or expanded_terms,
        ),
        "matched_terms": matching_terms,
        "related_event": _event_label(related_event),
        "related_event_id": (related_event or {}).get("id"),
        "line_samples": line_matches[:5],
        "evidence_types": evidence_types,
        "evidence_strength": strength,
        "scene_tags": _json_list(row.get("scene_tags_json")),
        "activity_tags": _json_list(row.get("activity_tags_json")),
        "food_cues": _json_list(row.get("food_cues_json")),
        "location_cues": _json_list(row.get("location_cues_json")),
        "confidence_label": _confidence_label(
            score_components["final_score"],
            evidence_types=evidence_types,
            safety_flags=safety_flags,
            visual_match=visual_match,
        ),
        "score_components": score_components,
        "reasons": _reasons(row, score_components, related_event, line_matches, evidence_types),
        "review_status": row.get("review_status") or "unreviewed",
        "is_verified": int(row.get("is_verified") or 0),
        "is_hidden": int(row.get("is_hidden") or 0),
        "is_wrong": int(row.get("is_wrong") or 0),
        "is_searchable": int(row.get("is_searchable") if row.get("is_searchable") is not None else 1),
        "is_event_usable": int(row.get("is_event_usable") if row.get("is_event_usable") is not None else 1),
    }


def compute_multimodal_evidence_strength(
    *,
    evidence_types: list[str],
    verified_event: bool = False,
    pinned_event: bool = False,
    visual_match: bool = True,
) -> str:
    if not visual_match:
        return "weak"
    return compute_evidence_strength(
        evidence_types,
        verified_event=verified_event,
        pinned_event=pinned_event,
    )


def _media_row_from_candidates(
    repository: LifelogRepository,
    media_id: str,
    embedding_candidate: dict[str, Any],
    sql_candidate: dict[str, Any],
) -> dict[str, Any] | None:
    row = dict((embedding_candidate.get("embedding_row") or {}))
    row.update(sql_candidate.get("sql_row") or {})
    vlm = repository.get_media_vlm(media_id) or {}
    ocr = repository.get_media_ocr(media_id) or {}
    if vlm:
        row.update(vlm)
    elif ocr:
        row.update(ocr)
    override = repository.get_media_vlm_override(media_id)
    if override:
        row.update(
            {
                "caption_override": override.get("caption_override"),
                "short_caption_override": override.get("short_caption_override"),
                "scene_tags_override_json": override.get("scene_tags_override_json"),
                "object_tags_override_json": override.get("object_tags_override_json"),
                "activity_tags_override_json": override.get("activity_tags_override_json"),
                "food_cues_override_json": override.get("food_cues_override_json"),
                "location_cues_override_json": override.get("location_cues_override_json"),
                "vlm_is_verified": override.get("is_verified", 0),
                "vlm_is_hidden": override.get("is_hidden", 0),
                "vlm_is_wrong": override.get("is_wrong", 0),
                "vlm_is_searchable": override.get("is_searchable", 1),
                "vlm_is_event_usable": override.get("is_event_usable", 1),
                "vlm_review_status": override.get("review_status") or "unreviewed",
                "vlm_review_note": override.get("review_note"),
            }
        )
    if not row:
        media = repository.list_media_items(limit=1_000_000)
        row = next((item for item in media if str(item.get("id")) == media_id), {})
    return row or None


def _score_components(
    row: dict[str, Any],
    *,
    embedding_score: float,
    sql_score: float,
    related_event: dict[str, Any] | None,
    line_matches: list[dict[str, Any]],
) -> dict[str, float]:
    vlm_text_score = max(
        sql_score,
        _text_match_score(row, ("caption", "short_caption", "scene_tags_json", "object_tags_json", "activity_tags_json", "food_cues_json", "location_cues_json")),
    )
    ocr_score = _text_match_score(row, ("ocr_text", "ocr_text_redacted"))
    line_score = min(0.4, len(line_matches) * 0.08)
    event_score = 0.0
    place_score = 0.0
    override_boost = 0.0
    safety_penalty = -0.1 if row.get("safety_flags_json") and "forbidden" in str(row.get("safety_flags_json")) else 0.0
    if related_event:
        event_score = min(0.6, 0.25 + float(related_event.get("confidence") or 0.0) * 0.3)
        if related_event.get("location_name"):
            place_score = 0.1
        if related_event.get("is_pinned"):
            override_boost += 0.12
        if related_event.get("is_verified"):
            override_boost += 0.08
    if row.get("is_verified") or row.get("review_status") == "accepted":
        override_boost += 0.08
    if row.get("is_wrong") or row.get("review_status") in {"rejected", "wrong"}:
        safety_penalty -= 0.5
    final = (
        embedding_score * 0.38
        + vlm_text_score * 0.18
        + sql_score * 0.12
        + ocr_score * 0.15
        + line_score
        + event_score * 0.2
        + place_score
        + override_boost
        + safety_penalty
    )
    if embedding_score and not any([vlm_text_score, ocr_score, line_score, event_score]):
        final = min(final, 0.44)
    if vlm_text_score and not any([embedding_score, ocr_score, line_score, event_score]):
        final = min(final, 0.44)
    return {
        "sql_score": round(sql_score, 3),
        "embedding_score": round(embedding_score, 3),
        "vlm_text_score": round(vlm_text_score, 3),
        "ocr_score": round(ocr_score, 3),
        "line_score": round(line_score, 3),
        "event_score": round(event_score, 3),
        "place_score": round(place_score, 3),
        "override_boost": round(override_boost, 3),
        "safety_penalty": round(safety_penalty, 3),
        "final_score": round(max(0.0, min(final, 0.95)), 3),
    }


def _text_match_score(row: dict[str, Any], keys: tuple[str, ...]) -> float:
    values = [str(row.get(key) or "") for key in keys]
    return min(0.5, sum(0.12 for value in values if value.strip()))


def _related_event(repository: LifelogRepository, media_id: str, date_value: str, *, include_hidden: bool) -> dict[str, Any] | None:
    if not media_id or not date_value:
        return None
    for event in repository.list_events(start_date=date_value, end_date=date_value, include_hidden=include_hidden, limit=1000):
        evidence = repository.list_event_evidence(str(event["id"]))
        if any(row.get("evidence_type") in {"photo", "vlm", "ocr"} and row.get("evidence_id") == media_id for row in evidence):
            return event
    return None


def _line_matches(repository: LifelogRepository, query: str, date_value: str, *, expanded_terms: list[str]) -> list[dict[str, Any]]:
    if not date_value:
        return []
    terms = list(dict.fromkeys(extract_search_terms(query) + expanded_terms))
    records = repository.search_text_records(terms=terms, start_date=date_value, end_date=date_value, limit=20, include_hidden=False)
    return [
        {
            "time": str(row.get("sent_at") or "")[11:16],
            "sender": redact_text(row.get("sender"), max_chars=24),
            "text": redact_text(row.get("text"), max_chars=70),
        }
        for row in records.get("line_messages", [])[:5]
    ]


def _evidence_types(
    row: dict[str, Any],
    embedding_candidate: dict[str, Any],
    *,
    related_event: dict[str, Any] | None,
    line_matches: list[dict[str, Any]],
) -> list[str]:
    types: list[str] = ["photo"]
    if embedding_candidate:
        types.append("embedding")
    if row.get("caption") or row.get("short_caption") or row.get("food_cues_json"):
        types.append("vlm")
    if row.get("ocr_text") or row.get("ocr_text_redacted"):
        types.append("ocr")
    if related_event:
        types.append("event")
        if related_event.get("location_name"):
            types.append("place")
    if row.get("gps_lat") is not None and row.get("gps_lon") is not None:
        types.append("gps")
    if line_matches:
        types.append("line")
    return list(dict.fromkeys(types))


def _matched_fields(row: dict[str, Any], query: str, *, has_embedding: bool, expanded_terms: list[str] | None = None) -> list[str]:
    return matched_fields_for_row(row, expanded_terms or expand_visual_query_terms(query), has_embedding=has_embedding)


def _matched_terms(row: dict[str, Any], terms: list[str]) -> list[str]:
    return matched_terms_for_row(row, terms)


def _event_label(event: dict[str, Any] | None) -> str | None:
    if not event:
        return None
    return redact_text(f"{event.get('start_time') or ''} {event.get('title') or ''}".strip(), max_chars=100)


def _reasons(
    row: dict[str, Any],
    score_components: dict[str, float],
    event: dict[str, Any] | None,
    line_matches: list[dict[str, Any]],
    evidence_types: list[str],
) -> list[str]:
    reasons: list[str] = []
    if score_components["embedding_score"] > 0:
        reasons.append("embedding similarity candidate")
    if "vlm" in evidence_types:
        reasons.append("VLM caption/tags available")
    if "ocr" in evidence_types:
        reasons.append("OCR text available")
    if line_matches:
        reasons.append("same-day LINE mention exists")
    if event:
        reasons.append("related event evidence exists")
    if score_components["final_score"] <= 0.44 and set(evidence_types).issubset({"photo", "embedding", "vlm"}):
        reasons.append("embedding/VLM-only evidence kept weak")
    return reasons


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [str(raw)] if str(raw).strip() else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def _confidence_label(
    score: float,
    *,
    evidence_types: list[str],
    safety_flags: list[str] | None = None,
    visual_match: bool = True,
) -> str:
    label = confidence_label_for_score(score, evidence_types=evidence_types, safety_flags=safety_flags)
    if not visual_match:
        return "低"
    return label
