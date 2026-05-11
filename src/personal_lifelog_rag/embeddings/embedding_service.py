"""Batch builders and stats for local media embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.base import MultimodalEmbeddingEngine
from personal_lifelog_rag.embeddings.engines import get_multimodal_embedding_engine
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.embeddings.schemas import (
    BuildMediaEmbeddingsOptions,
    BuildMediaEmbeddingsReport,
)


DEFAULT_TEXT_TYPES = {"caption", "ocr", "combined_text"}


def build_image_embeddings(
    repository: LifelogRepository,
    options: BuildMediaEmbeddingsOptions,
    *,
    engine: MultimodalEmbeddingEngine | None = None,
    progress_callback=None,
) -> BuildMediaEmbeddingsReport:
    resolved_engine = engine or get_multimodal_embedding_engine(
        options.engine_name,
        model_name=options.model_name,
        model_path=options.model_path,
        device=options.device,
        dtype=options.dtype,
        local_files_only=options.local_files_only,
        embedding_dim=options.embedding_dim,
        batch_size=options.batch_size,
    )
    embedding_repository = MediaEmbeddingRepository(repository.db_path)
    rows = [
        row
        for row in repository.list_media_items(
            start_date=options.start_date,
            end_date=options.end_date,
            limit=1_000_000,
        )
        if str(row.get("media_type") or "image") == "image"
        and Path(str(row.get("file_path") or "")).expanduser().exists()
    ]
    if options.media_ids:
        allowed_ids = set(options.media_ids)
        rows = [row for row in rows if str(row.get("id")) in allowed_ids]
    rows = rows[: max(options.limit, 0)]
    report = BuildMediaEmbeddingsReport(
        selected=len(rows),
        dry_run=options.dry_run,
        embedding_type="image",
        engine=resolved_engine.name,
        model_name=getattr(resolved_engine, "model_name", None),
    )
    model_name = _model_name(resolved_engine)
    available = resolved_engine.is_available()
    for index, row in enumerate(rows, start=1):
        media_id = str(row["id"])
        if progress_callback is not None and not options.dry_run:
            progress_callback(f"image embedding {index}/{len(rows)} {media_id}")
        if options.skip_existing and not options.force and embedding_repository.existing_success(media_id, "image", model_name):
            _add_row(report, row, "skipped")
            report.skipped += 1
            continue
        if options.dry_run:
            _add_row(report, row, "pending")
            continue
        if not available:
            embedding_repository.upsert_embedding(
                media_id=media_id,
                embedding_type="image",
                embedding_model=model_name,
                status="engine_unavailable",
                error_message=_engine_unavailable_message(resolved_engine),
            )
            _add_row(report, row, "engine_unavailable")
            report.engine_unavailable += 1
            report.processed += 1
            continue
        image_path = Path(str(row.get("file_path") or "")).expanduser()
        if not image_path.exists():
            embedding_repository.upsert_embedding(
                media_id=media_id,
                embedding_type="image",
                embedding_model=model_name,
                status="failed",
                error_message="image file does not exist",
            )
            _add_row(report, row, "failed")
            report.failed += 1
            report.processed += 1
            continue
        result = resolved_engine.embed_image(image_path)
        _save_result(embedding_repository, media_id, "image", model_name, result, source_text=None)
        _count_status(report, result.status)
        _add_row(report, row, result.status, dim=result.embedding_dim)
        report.processed += 1
    return report


def build_text_embeddings(
    repository: LifelogRepository,
    options: BuildMediaEmbeddingsOptions,
    *,
    engine: MultimodalEmbeddingEngine | None = None,
    progress_callback=None,
) -> BuildMediaEmbeddingsReport:
    if options.embedding_type not in DEFAULT_TEXT_TYPES:
        raise ValueError("--type must be one of caption, ocr, combined_text")
    resolved_engine = engine or get_multimodal_embedding_engine(
        options.engine_name,
        model_name=options.model_name,
        model_path=options.model_path,
        device=options.device,
        dtype=options.dtype,
        local_files_only=options.local_files_only,
        embedding_dim=options.embedding_dim,
        batch_size=options.batch_size,
    )
    embedding_repository = MediaEmbeddingRepository(repository.db_path)
    candidate_rows = repository.list_media_items(
        start_date=options.start_date,
        end_date=options.end_date,
        limit=1_000_000,
    )
    candidate_rows = [
        row
        for row in candidate_rows
        if Path(str(row.get("file_path") or "")).expanduser().exists()
    ]
    if options.media_ids:
        allowed_ids = set(options.media_ids)
        candidate_rows = [row for row in candidate_rows if str(row.get("id")) in allowed_ids]
    rows = []
    for row in candidate_rows:
        text = _media_text(repository, str(row["id"]), options.embedding_type)
        if text:
            rows.append({**row, "source_text": text})
        if len(rows) >= max(options.limit, 0):
            break
    report = BuildMediaEmbeddingsReport(
        selected=len(rows),
        dry_run=options.dry_run,
        embedding_type=options.embedding_type,
        engine=resolved_engine.name,
        model_name=getattr(resolved_engine, "model_name", None),
    )
    model_name = _model_name(resolved_engine)
    available = resolved_engine.is_available()
    for index, row in enumerate(rows, start=1):
        media_id = str(row["id"])
        source_text = str(row["source_text"])
        if progress_callback is not None and not options.dry_run:
            progress_callback(f"{options.embedding_type} embedding {index}/{len(rows)} {media_id}")
        if options.skip_existing and not options.force and embedding_repository.existing_success(media_id, options.embedding_type, model_name):
            _add_row(report, row, "skipped")
            report.skipped += 1
            continue
        if options.dry_run:
            _add_row(report, row, "pending", source_text=source_text)
            continue
        if not available:
            embedding_repository.upsert_embedding(
                media_id=media_id,
                embedding_type=options.embedding_type,
                embedding_model=model_name,
                source_text=source_text,
                status="engine_unavailable",
                error_message=_engine_unavailable_message(resolved_engine),
            )
            _add_row(report, row, "engine_unavailable", source_text=source_text)
            report.engine_unavailable += 1
            report.processed += 1
            continue
        result = resolved_engine.embed_text(source_text)
        _save_result(embedding_repository, media_id, options.embedding_type, model_name, result, source_text=source_text)
        _count_status(report, result.status)
        _add_row(report, row, result.status, dim=result.embedding_dim, source_text=source_text)
        report.processed += 1
    return report


def embedding_stats(repository: LifelogRepository, *, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    return MediaEmbeddingRepository(repository.db_path).stats(start_date=start_date, end_date=end_date)


def format_embedding_build_report(report: BuildMediaEmbeddingsReport) -> str:
    lines = [
        "Media embedding summary:",
        f"- type: {report.embedding_type}",
        f"- selected: {report.selected}",
        f"- processed: {report.processed}",
        f"- success: {report.success}",
        f"- skipped: {report.skipped}",
        f"- failed: {report.failed}",
        f"- engine_unavailable: {report.engine_unavailable}",
        f"- engine: {report.engine}",
        f"- model: {report.model_name or ''}",
        f"- dry_run: {report.dry_run}",
    ]
    return "\n".join(lines)


def format_embedding_stats(report: dict[str, Any]) -> str:
    lines = [
        "Embedding Stats",
        f"- range: {report['range']['from'] or 'all'}..{report['range']['to'] or 'all'}",
        f"- total media_embeddings: {report['total']}",
        "type counts:",
        *_counts(report["by_type"]),
        "model counts:",
        *_counts(report["by_model"]),
        "status counts:",
        *_counts(report["by_status"]),
        "embedding_dim distribution:",
        *_counts(report["embedding_dim_distribution"]),
    ]
    return "\n".join(lines)


def _save_result(
    embedding_repository: MediaEmbeddingRepository,
    media_id: str,
    embedding_type: str,
    model_name: str,
    result,
    *,
    source_text: str | None,
) -> None:
    embedding_repository.upsert_embedding(
        media_id=media_id,
        embedding_type=embedding_type,
        embedding_model=model_name,
        vector=result.vector if result.status == "success" else None,
        embedding_dim=result.embedding_dim,
        source_text=source_text,
        status=result.status,
        error_message=result.error_message,
    )


def _media_text(repository: LifelogRepository, media_id: str, embedding_type: str) -> str:
    vlm = repository.get_media_vlm(media_id) or {}
    ocr = repository.get_media_ocr(media_id) or {}
    if embedding_type == "caption":
        return " ".join(str(vlm.get(key) or "") for key in ("short_caption", "caption")).strip()
    if embedding_type == "ocr":
        return str(ocr.get("ocr_text") or ocr.get("ocr_text_redacted") or "").strip()
    parts = [
        vlm.get("short_caption"),
        vlm.get("caption"),
        vlm.get("scene_tags_json"),
        vlm.get("object_tags_json"),
        vlm.get("activity_tags_json"),
        vlm.get("food_cues_json"),
        vlm.get("location_cues_json"),
        ocr.get("ocr_text"),
        ocr.get("ocr_text_redacted"),
    ]
    return _clean_join(parts)


def _clean_join(parts: list[Any]) -> str:
    values: list[str] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, (list, tuple)):
            values.extend(str(item) for item in part if str(item).strip())
            continue
        text = str(part)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values.extend(str(item) for item in parsed if str(item).strip())
        elif text.strip():
            values.append(text.strip())
    return " ".join(values)


def _model_name(engine: MultimodalEmbeddingEngine) -> str:
    return str(getattr(engine, "model_name", None) or getattr(engine, "name", "unknown") or "unknown")


def _engine_unavailable_message(engine: MultimodalEmbeddingEngine) -> str:
    reporter = getattr(engine, "availability_error", None)
    if callable(reporter):
        try:
            message = reporter()
        except Exception as exc:
            message = f"availability_error failed: {exc.__class__.__name__}: {exc}"
        if message:
            return str(message)
    return f"embedding engine '{engine.name}' is not available"


def _count_status(report: BuildMediaEmbeddingsReport, status: str) -> None:
    if status == "success":
        report.success += 1
    elif status == "failed":
        report.failed += 1
    elif status == "engine_unavailable":
        report.engine_unavailable += 1
    else:
        report.skipped += 1


def _add_row(
    report: BuildMediaEmbeddingsReport,
    row: dict[str, Any],
    status: str,
    *,
    dim: int | None = None,
    source_text: str | None = None,
) -> None:
    report.rows.append(
        {
            "media_id": row.get("id"),
            "file_name": row.get("file_name"),
            "status": status,
            "embedding_dim": dim,
            "source_text_preview": redact_text(source_text, max_chars=80),
        }
    )


def _counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in counts.items()]
