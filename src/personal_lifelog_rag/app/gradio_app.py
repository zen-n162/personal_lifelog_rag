"""Optional localhost-only Gradio UI."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import ensure_localhost, redact_text
from personal_lifelog_rag.db.repository import LifelogRepository, resolve_db_path
from personal_lifelog_rag.embeddings.multimodal_search import format_multimodal_search, multimodal_search
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.ingest.line_parser import parse_line_chat_file_with_warnings
from personal_lifelog_rag.ingest.photo_ingest import ingest_photo_directory_with_report
from personal_lifelog_rag.ocr.engines import get_ocr_engine
from personal_lifelog_rag.ocr.ocr_service import (
    OcrImagesOptions,
    format_ocr_report,
    format_ocr_stats,
    ocr_stats,
    run_ocr_images,
)
from personal_lifelog_rag.places.place_dictionary import DEFAULT_PRIVATE_PLACES_PATH, load_place_dictionary
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.ui.event_review import (
    clear_event_review_override,
    event_detail,
    event_review_overview,
    save_event_review_override,
)
from personal_lifelog_rag.ui.event_review_service import (
    ReviewQueueFilters,
    bulk_update_events,
    hide_low_confidence_line_only,
    make_eval_case_yaml,
    review_queue,
    review_rows_for_dataframe,
)
from personal_lifelog_rag.ui.face_review_service import (
    add_person_alias_for_ui,
    create_person_for_ui,
    face_detail_for_ui,
    face_cluster_action_for_ui,
    face_cluster_detail_for_ui,
    face_cluster_review_for_ui,
    face_review_queue_for_ui,
    label_face_cluster_for_ui,
    link_cluster_to_person_for_ui,
    link_person_cluster_for_ui,
    line_speakers_for_ui,
    link_line_speaker_for_ui,
    merge_cluster_into_target_person_for_ui,
    next_face_id,
    next_cluster_id,
    next_person_id,
    person_cluster_overview_for_ui,
    person_form_for_ui,
    persons_for_ui,
    previous_face_id,
    previous_cluster_id,
    previous_person_id,
    unlink_cluster_from_person_for_ui,
    unlink_person_cluster_for_ui,
    unlink_line_speaker_for_ui,
    update_person_for_ui,
    update_face_review_for_ui,
)
from personal_lifelog_rag.ui.image_search_service import image_search_for_ui
from personal_lifelog_rag.ui.monthly_summary_service import monthly_summary_for_ui
from personal_lifelog_rag.ui.multimodal_search_service import (
    detail_values as multimodal_detail_values,
    mark_search_result_for_review,
    multimodal_search_for_ui,
    search_result_detail_for_ui,
)
from personal_lifelog_rag.ui.place_review_service import (
    add_place_alias_for_ui,
    create_place_for_ui,
    link_cluster_for_ui,
    place_cluster_detail_for_ui,
    place_review_queue_for_ui,
    reassign_places_for_ui,
    set_cluster_status_for_ui,
    unlink_cluster_for_ui,
    update_place_for_ui,
)
from personal_lifelog_rag.ui.report_viewer_service import (
    generate_report_for_ui,
    list_reports_for_ui,
    load_report_for_ui,
)
from personal_lifelog_rag.ui.thumbnail_gallery import media_thumbnail_gallery_items, sort_media_by_capture_date
from personal_lifelog_rag.ui.vlm_review_service import (
    VlmOverrideUpdate,
    VlmReviewFilters,
    bulk_update_vlm_overrides,
    clear_vlm_override,
    generate_vlm_eval_case,
    get_vlm_review_detail,
    list_vlm_review_items,
    parse_tag_text,
    review_rows_for_dataframe as vlm_review_rows_for_dataframe,
    save_vlm_override,
)
from personal_lifelog_rag.vlm.engines import get_vlm_engine
from personal_lifelog_rag.vlm.vlm_service import (
    VlmImagesOptions,
    format_vlm_report,
    format_vlm_stats,
    run_vlm_images,
    vlm_stats,
)

GRADIO_INSTALL_MESSAGE = (
    "Gradio is not installed. Install the local UI dependencies with "
    '`pip install -e ".[ui]"`, then run `python -m personal_lifelog_rag.app.cli ui` again.'
)

PHOTOS_INSPIRED_CSS = """
:root {
  --lifelog-bg: #000000;
  --lifelog-bg-elevated: #101014;
  --lifelog-surface: rgba(28, 28, 30, 0.86);
  --lifelog-surface-strong: rgba(44, 44, 46, 0.96);
  --lifelog-surface-soft: rgba(58, 58, 60, 0.56);
  --lifelog-border: rgba(255, 255, 255, 0.12);
  --lifelog-border-strong: rgba(255, 255, 255, 0.22);
  --lifelog-text: #f5f5f7;
  --lifelog-text-strong: #ffffff;
  --lifelog-muted: #b8bbc6;
  --lifelog-muted-2: #8e8e93;
  --lifelog-blue: #0a84ff;
  --lifelog-blue-soft: rgba(10, 132, 255, 0.22);
  --lifelog-red: #ff453a;
  --lifelog-shadow: 0 18px 50px rgba(0, 0, 0, 0.46);
  --lifelog-shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.34);
}

html {
  background: var(--lifelog-bg) !important;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at 18% 0%, rgba(10, 132, 255, 0.18), transparent 30%),
    radial-gradient(circle at 82% 8%, rgba(118, 118, 128, 0.16), transparent 28%),
    linear-gradient(180deg, #09090c 0%, var(--lifelog-bg) 56%, #050507 100%) !important;
  color: var(--lifelog-text) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
  letter-spacing: 0 !important;
}

.gradio-container,
.gradio-container p,
.gradio-container span,
.gradio-container div,
.gradio-container label,
.gradio-container li,
.gradio-container td,
.gradio-container th,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  color: var(--lifelog-text) !important;
}

.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding: 18px 22px 34px !important;
}

.photos-hero {
  margin: 4px 0 18px !important;
  padding: 22px 24px !important;
  border: 1px solid var(--lifelog-border) !important;
  border-radius: 8px !important;
  background:
    linear-gradient(135deg, rgba(44, 44, 46, 0.96), rgba(16, 16, 20, 0.86)) !important;
  box-shadow: var(--lifelog-shadow) !important;
  backdrop-filter: blur(22px) saturate(1.25);
}

.photos-hero h1 {
  margin: 0 !important;
  color: var(--lifelog-text-strong) !important;
  font-size: 34px !important;
  font-weight: 760 !important;
  line-height: 1.05 !important;
  letter-spacing: 0 !important;
}

.photos-hero p {
  margin: 8px 0 0 !important;
  color: var(--lifelog-muted) !important;
  font-size: 14px !important;
}

.photos-hero .kicker {
  margin-bottom: 8px;
  color: var(--lifelog-blue);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.tabs {
  border-radius: 8px !important;
}

.tab-nav,
.tabitem {
  background: transparent !important;
}

.tab-nav button {
  border-radius: 8px !important;
  color: var(--lifelog-muted) !important;
  font-weight: 650 !important;
  background: transparent !important;
}

.tab-nav button.selected {
  background: var(--lifelog-surface-strong) !important;
  color: var(--lifelog-text-strong) !important;
  box-shadow: var(--lifelog-shadow-soft) !important;
}

.block,
.form,
.panel,
.compact,
.prose,
.gradio-accordion {
  border-color: var(--lifelog-border) !important;
  border-radius: 8px !important;
  color: var(--lifelog-text) !important;
}

.block,
.form {
  background: var(--lifelog-surface) !important;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.08) inset, var(--lifelog-shadow-soft) !important;
  backdrop-filter: blur(18px) saturate(1.18);
}

label,
.label-wrap span {
  color: var(--lifelog-muted) !important;
  font-weight: 650 !important;
}

input,
textarea,
select,
.wrap,
.container,
.input-container {
  border-radius: 8px !important;
}

input,
textarea,
select {
  background: rgba(28, 28, 30, 0.98) !important;
  border-color: var(--lifelog-border) !important;
  color: var(--lifelog-text) !important;
}

input::placeholder,
textarea::placeholder {
  color: var(--lifelog-muted-2) !important;
  opacity: 1 !important;
}

input:focus,
textarea:focus,
select:focus {
  border-color: rgba(10, 132, 255, 0.72) !important;
  box-shadow: 0 0 0 3px var(--lifelog-blue-soft) !important;
}

button {
  border-radius: 8px !important;
  border-color: var(--lifelog-border) !important;
  font-weight: 700 !important;
  color: var(--lifelog-text) !important;
  background: rgba(44, 44, 46, 0.92) !important;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.08) inset !important;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease !important;
}

button:hover {
  transform: translateY(-1px);
  box-shadow: var(--lifelog-shadow-soft) !important;
}

button.primary {
  background: var(--lifelog-blue) !important;
  border-color: rgba(0, 122, 255, 0.75) !important;
  color: #ffffff !important;
}

button.secondary {
  background: rgba(58, 58, 60, 0.88) !important;
  color: var(--lifelog-text) !important;
}

button.stop {
  background: rgba(255, 69, 58, 0.18) !important;
  border-color: rgba(255, 69, 58, 0.46) !important;
  color: #ffd8d6 !important;
}

table,
.table-wrap,
.dataframe {
  border-radius: 8px !important;
  overflow: hidden !important;
  background: rgba(28, 28, 30, 0.96) !important;
  color: var(--lifelog-text) !important;
}

table thead,
.dataframe thead {
  background: rgba(44, 44, 46, 0.98) !important;
}

table th,
.dataframe th {
  color: var(--lifelog-muted) !important;
  font-size: 12px !important;
  font-weight: 760 !important;
}

table td,
.dataframe td {
  color: var(--lifelog-text) !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
  background: rgba(28, 28, 30, 0.86) !important;
}

.ag-theme-quartz,
.ag-theme-balham,
.ag-root-wrapper,
.ag-root,
.ag-header,
.ag-row,
.ag-cell {
  background: rgba(28, 28, 30, 0.96) !important;
  color: var(--lifelog-text) !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
}

.ag-header-cell-text,
.ag-cell-value {
  color: var(--lifelog-text) !important;
}

.photo-grid,
.photo-grid .grid-container,
.photo-grid .gallery {
  background: rgba(0, 0, 0, 0.32) !important;
  border-radius: 8px !important;
}

.photo-grid img,
.thumbnail-focus img,
.gallery img {
  border-radius: 8px !important;
  box-shadow: 0 14px 28px rgba(0, 0, 0, 0.48) !important;
}

.thumbnail-focus {
  background: #0b0c10 !important;
  border-radius: 8px !important;
  padding: 8px !important;
}

.thumbnail-focus img {
  object-fit: contain !important;
}

.prose h1,
.prose h2,
.prose h3 {
  color: var(--lifelog-text-strong) !important;
  letter-spacing: 0 !important;
}

.prose,
.prose p,
.prose li,
.markdown,
.markdown p,
.markdown li {
  color: var(--lifelog-text) !important;
}

.prose code,
code {
  border-radius: 6px !important;
  background: rgba(10, 132, 255, 0.18) !important;
  color: #b9dcff !important;
}

a {
  color: #79b8ff !important;
}

@media (max-width: 760px) {
  .gradio-container {
    padding: 12px 12px 24px !important;
  }

  .photos-hero {
    padding: 18px 16px !important;
  }

  .photos-hero h1 {
    font-size: 27px !important;
  }
}

@media print {
  body,
  .gradio-container {
    background: #ffffff !important;
    color: #111111 !important;
  }

  .photos-hero,
  .block,
  .form {
    box-shadow: none !important;
    backdrop-filter: none !important;
  }
}
"""


def answer_question(question: str, db_path: str | Path | None = None) -> str:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    date_range = parse_date_query(question)
    result = search_timeline(repository, question, date_range=date_range)
    return build_answer(question, result)


def create_app(db_path: str | Path | None = None):
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(GRADIO_INSTALL_MESSAGE) from exc
    globals()["gr"] = gr

    blocks_kwargs: dict[str, Any] = {
        "title": "Personal LifeLog RAG",
        "css": PHOTOS_INSPIRED_CSS,
    }
    soft_theme = getattr(getattr(gr, "themes", None), "Soft", None)
    if soft_theme is not None:
        blocks_kwargs["theme"] = soft_theme(
            primary_hue="blue",
            neutral_hue="gray",
            radius_size="sm",
            spacing_size="sm",
            text_size="sm",
        )

    resolved_db_path = resolve_db_path(db_path)

    def _repository() -> LifelogRepository:
        repository = LifelogRepository(resolved_db_path)
        repository.initialize()
        return repository

    def _stats_values() -> tuple[str, int, int, int]:
        stats = _repository().stats()
        return (
            str(resolved_db_path),
            stats.get("media_items", 0),
            stats.get("line_messages", 0),
            stats.get("events", 0),
        )

    def _place_choices() -> list[str]:
        try:
            places = load_place_dictionary(DEFAULT_PRIVATE_PLACES_PATH, required=False)
        except Exception:
            return []
        return [place.display_name for place in places]

    def _ingest_line(path: str) -> str:
        repository = _repository()
        root = Path(path).expanduser()
        files = _line_export_files(root)
        imported = 0
        parsed = 0
        warnings = 0
        for file_path in files:
            result = parse_line_chat_file_with_warnings(file_path)
            parsed += len(result.messages)
            warnings += len(result.warnings)
            imported += repository.add_line_messages(result.messages)
        duplicates = parsed - imported
        return (
            "LINE: "
            f"{imported} new, {duplicates} duplicate, {warnings} warning(s), "
            f"{len(files)} file(s)"
        )

    def _ingest_photos(path: str) -> str:
        report = ingest_photo_directory_with_report(path, _repository())
        return (
            "Photos: "
            f"{report.imported} new, {report.duplicates} duplicate, "
            f"{report.skipped} skipped, {report.scanned} file(s)"
        )

    def _ingest_all(photo_path: str, line_path: str) -> tuple[str, str, str, int, int, int]:
        photo_result = _ingest_photos(photo_path)
        line_result = _ingest_line(line_path)
        db_path_text, media_count, line_count, event_count = _stats_values()
        return (
            photo_result,
            line_result,
            db_path_text,
            media_count,
            line_count,
            event_count,
        )

    def _run_ocr_ui(date_text: str, limit_value: float | None, engine_name: str, dry_run: bool) -> str:
        date_value = _blank_or_none(date_text)
        report = run_ocr_images(
            _repository(),
            OcrImagesOptions(
                start_date=date_value,
                end_date=date_value,
                limit=_int_or_none(limit_value) or 20,
                engine_name=engine_name or "tesseract_cli",
                languages=["jpn", "eng"],
                dry_run=dry_run,
                skip_existing=True,
            ),
            engine=get_ocr_engine(engine_name or "tesseract_cli"),
        )
        return format_ocr_report(report)

    def _ocr_stats_ui(date_text: str) -> str:
        date_value = _blank_or_none(date_text)
        return format_ocr_stats(ocr_stats(_repository(), start_date=date_value, end_date=date_value))

    def _run_vlm_ui(date_text: str, limit_value: float | None, engine_name: str, dry_run: bool) -> str:
        date_value = _blank_or_none(date_text)
        report = run_vlm_images(
            _repository(),
            VlmImagesOptions(
                start_date=date_value,
                end_date=date_value,
                limit=_int_or_none(limit_value) or 10,
                engine_name=engine_name or "noop",
                dry_run=dry_run,
                skip_existing=True,
            ),
            engine=get_vlm_engine(engine_name or "noop"),
        )
        return format_vlm_report(report)

    def _vlm_stats_ui(date_text: str) -> str:
        date_value = _blank_or_none(date_text)
        return format_vlm_stats(vlm_stats(_repository(), start_date=date_value, end_date=date_value))

    def _image_search_ui(query_text: str, limit_value: float | None) -> tuple[str, list[list[Any]], list[tuple[str, str]]]:
        payload = image_search_for_ui(
            _repository(),
            query=query_text,
            limit=_int_or_none(limit_value) or 20,
        )
        return payload["summary_text"], payload["rows"], payload["gallery"]

    def _multimodal_search_ui(
        query_text: str,
        person_filter: str,
        place_filter: str,
        activity_query: str,
        backend_value: str,
        from_text: str,
        to_text: str,
        limit_value: float | None,
        include_hidden: bool = False,
        include_sensitive_vlm: bool = False,
    ) -> tuple[Any, ...]:
        del include_sensitive_vlm
        composed_query = _compose_person_place_query(query_text, person_filter, place_filter, activity_query)
        payload = multimodal_search_for_ui(
            _repository(),
            query=composed_query,
            date_from=_blank_or_none(from_text),
            date_to=_blank_or_none(to_text),
            limit=_int_or_none(limit_value) or 10,
            backend=backend_value,
            include_hidden=include_hidden,
        )
        selected = payload["media_ids"][0] if payload["media_ids"] else None
        details = multimodal_detail_values(search_result_detail_for_ui(_repository(), selected))
        return (
            payload["summary_text"],
            payload["rows"],
            gr.update(choices=payload["media_ids"], value=selected),
            payload["media_ids"],
            *details,
        )

    def _multimodal_detail_ui(media_id: str | None):
        return multimodal_detail_values(search_result_detail_for_ui(_repository(), media_id))

    def _adjacent_loaded_id(ids: list[str] | None, current_id: str | None, *, direction: int) -> str | None:
        item_ids = [str(item_id) for item_id in (ids or []) if item_id]
        if not item_ids:
            return None
        if current_id not in item_ids:
            return item_ids[0] if direction >= 0 else item_ids[-1]
        index = item_ids.index(str(current_id))
        return item_ids[(index + direction) % len(item_ids)]

    def _multimodal_adjacent_ui(media_id: str | None, media_ids: list[str] | None, direction: int):
        next_id = _adjacent_loaded_id(media_ids, media_id, direction=direction)
        details = multimodal_detail_values(search_result_detail_for_ui(_repository(), next_id))
        return (gr.update(value=next_id), *details)

    def _multimodal_quick_action(media_id: str | None, action: str):
        detail = mark_search_result_for_review(_repository(), media_id, action)
        return (*multimodal_detail_values(detail), f"{action} を保存しました。")

    def _monthly_summary_ui(
        month_text: str,
        from_text: str,
        to_text: str,
        mode_text: str,
        include_hidden: bool,
    ):
        try:
            payload = monthly_summary_for_ui(
                _repository(),
                month=_blank_or_none(month_text),
                date_from=_blank_or_none(from_text),
                date_to=_blank_or_none(to_text),
                mode=mode_text,
                include_hidden=include_hidden,
            )
        except ValueError as exc:
            return str(exc), [], [], [], []
        return (
            payload["summary_text"],
            payload["metrics"],
            payload["title_distribution_rows"],
            payload["representative_day_rows"],
            payload["representative_event_rows"],
        )

    def _report_choices() -> Any:
        reports = list_reports_for_ui()
        return gr.update(choices=reports, value=reports[0] if reports else None)

    def _load_report_ui(path: str | None):
        payload = load_report_for_ui(path)
        return payload.get("markdown", ""), payload.get("json_summary", ""), payload.get("path", "")

    def _generate_report_ui(
        from_text: str,
        to_text: str,
        mode_text: str,
        include_examples: bool,
        save_json: bool,
    ):
        payload = generate_report_for_ui(
            _repository(),
            start_date=_blank_or_none(from_text),
            end_date=_blank_or_none(to_text),
            mode=mode_text,
            include_examples=include_examples,
            save_json=save_json,
        )
        reports = list_reports_for_ui()
        return (
            gr.update(choices=reports, value=payload.get("markdown_path") or (reports[0] if reports else None)),
            payload.get("markdown", ""),
            payload.get("json_summary", ""),
            payload.get("markdown_path", ""),
        )

    def _ask(question: str) -> tuple[str, list[list[str]], list[list[str]], list[tuple[str, str]], str, int, int, int]:
        repository = _repository()
        date_range = parse_date_query(question)
        result = search_timeline(repository, question, date_range=date_range)
        answer = build_answer(question, result)
        line_rows = [_line_row(message) for message in result.line_messages[:50]]
        sorted_media = sort_media_by_capture_date(result.media_items)[:50]
        photo_rows = [_photo_row(item) for item in sorted_media]
        photo_gallery = media_thumbnail_gallery_items(sorted_media, limit=50)
        db_path_text, media_count, line_count, event_count = _stats_values()
        return answer, line_rows, photo_rows, photo_gallery, db_path_text, media_count, line_count, event_count

    def _load_events(date_text: str):
        overview = event_review_overview(_repository(), date_text.strip())
        choices = overview["event_ids"]
        selected = choices[0] if choices else None
        detail_values = _event_detail_values(selected)
        summary = (
            f"events={overview['event_count']}, "
            f"photos={overview['photo_count']}, "
            f"LINE={overview['line_count']}, "
            f"event_evidence={overview['event_evidence_count']}"
        )
        return (
            summary,
            overview["rows"],
            gr.update(choices=choices, value=selected),
            *detail_values,
        )

    def _load_review_queue(
        date_text: str,
        from_text: str,
        to_text: str,
        confidence_lte: float | None,
        title_text: str,
        location_text: str,
        modality_value: str,
        verified_value: str,
        hidden_value: str,
        pinned_value: str,
        evidence_min: float | None,
        evidence_max: float | None,
        title_category_value: str,
    ):
        filters = ReviewQueueFilters(
            date=_blank_or_none(date_text),
            date_from=_blank_or_none(from_text),
            date_to=_blank_or_none(to_text),
            confidence_lte=confidence_lte,
            title_contains=_blank_or_none(title_text),
            location_contains=_blank_or_none(location_text),
            modality=None if modality_value == "all" else modality_value,
            verified=_verified_filter(verified_value),
            hidden=_hidden_filter(hidden_value),
            pinned="pinned" if pinned_value == "pinned only" else "all",
            evidence_count_min=_int_or_none(evidence_min),
            evidence_count_max=_int_or_none(evidence_max),
            title_category=None if title_category_value == "all" else title_category_value,
            limit=500,
        )
        report = review_queue(_repository(), filters)
        rows = review_rows_for_dataframe(report)
        choices = [row[0] for row in rows]
        selected = choices[0] if choices else None
        detail_values = _event_detail_values(selected)
        summary = f"review events={report['total']}, shown={len(rows)}"
        return (
            summary,
            rows,
            gr.update(choices=choices, value=selected),
            *detail_values,
        )

    def _event_detail_values(event_id: str | None):
        if not event_id:
            return _empty_event_detail_values()
        detail = event_detail(_repository(), event_id)
        event = detail.get("event") or {}
        tags_text = _tags_text(event.get("tags_json"))
        line_rows = [
            [row["sent_at"], row["sender"], row["message_type"], row["text"]]
            for row in detail["line_evidence"]
        ]
        photo_rows = [
            [row["captured_at"], row["file_name"], row["gps"], row["location_name"], row["thumbnail_path"]]
            for row in detail["photo_evidence"]
        ]
        ocr_rows = [
            [row["media_id"], row["captured_at"], row["file_name"], row["status"], row["engine"], row["text"]]
            for row in detail.get("ocr_evidence", [])
        ]
        vlm_rows = [
            [
                row["media_id"],
                row["captured_at"],
                row["file_name"],
                row["status"],
                row["engine"],
                row["caption"],
                row["scene_tags"],
                row["object_tags"],
                row["activity_tags"],
                row["food_cues"],
                row["location_cues"],
                row["safety_flags"],
            ]
            for row in detail.get("vlm_evidence", [])
        ]
        return (
            str(event.get("title") or ""),
            str(event.get("summary") or ""),
            str(event.get("date") or ""),
            str(event.get("start_time") or ""),
            str(event.get("end_time") or ""),
            str(event.get("location_name") or ""),
            str(event.get("confidence") if event.get("confidence") is not None else ""),
            str(event.get("participants_json") or ""),
            tags_text,
            bool(event.get("is_verified")),
            bool(event.get("is_pinned")),
            bool(event.get("is_hidden")),
            str(detail.get("evidence_summary") or ""),
            line_rows,
            photo_rows,
            ocr_rows,
            vlm_rows,
            detail.get("photo_gallery") or [],
            "",
        )

    def _save_event(
        event_id: str | None,
        title: str,
        summary: str,
        location_name: str,
        location_choice: str | None,
        tags_text: str,
        is_verified: bool,
        is_pinned: bool,
        is_hidden: bool,
    ):
        if not event_id:
            return (*_empty_event_detail_values()[:-1], "イベントが選択されていません。")
        repository = _repository()
        save_event_review_override(
            repository,
            event_id,
            title=title,
            summary=summary,
            location_name=(location_choice or location_name),
            tags=tags_text,
            is_verified=is_verified,
            is_pinned=is_pinned,
            is_hidden=is_hidden,
        )
        values = _event_detail_values(event_id)
        return (*values[:-1], "保存しました。再読み込み後もoverrideが反映されます。")

    def _quick_verified(event_id: str | None):
        if not event_id:
            return (*_empty_event_detail_values()[:-1], "イベントが選択されていません。")
        save_event_review_override(_repository(), event_id, is_verified=True)
        values = _event_detail_values(event_id)
        return (*values[:-1], "手動確認済みにしました。")

    def _toggle_pinned(event_id: str | None):
        return _toggle_event_flag(event_id, "is_pinned")

    def _toggle_hidden(event_id: str | None):
        return _toggle_event_flag(event_id, "is_hidden")

    def _toggle_event_flag(event_id: str | None, key: str):
        if not event_id:
            return (*_empty_event_detail_values()[:-1], "イベントが選択されていません。")
        repository = _repository()
        event = repository.get_event(event_id, include_hidden=True) or {}
        kwargs = {"is_pinned": None, "is_hidden": None}
        kwargs[key] = not bool(event.get(key))
        save_event_review_override(repository, event_id, **kwargs)
        values = _event_detail_values(event_id)
        return (*values[:-1], f"{key} を切り替えました。")

    def _clear_overrides(event_id: str | None):
        if not event_id:
            return (*_empty_event_detail_values()[:-1], "イベントが選択されていません。")
        clear_event_review_override(_repository(), event_id)
        values = _event_detail_values(event_id)
        return (*values[:-1], "overrideを削除しました。")

    def _add_tag(event_id: str | None, tag_text: str):
        if not event_id:
            return (*_empty_event_detail_values()[:-1], "イベントが選択されていません。")
        tags = [tag_text.strip()] if tag_text.strip() else []
        bulk_update_events(_repository(), [event_id], add_tags=tags)
        values = _event_detail_values(event_id)
        return (*values[:-1], "タグを追加しました。")

    def _copy_event_id(event_id: str | None) -> str:
        return event_id or ""

    def _bulk_update(ids_text: str, action: str, tag_text: str) -> str:
        ids = [line.strip() for line in ids_text.splitlines() if line.strip()]
        report = bulk_update_events(
            _repository(),
            ids,
            verified=True if action == "verified" else None,
            hidden=True if action == "hidden" else (False if action == "unhide" else None),
            add_tags=[tag_text.strip()] if action == "tag" and tag_text.strip() else None,
        )
        return f"updated={report['updated_count']}, missing={report['missing_count']}"

    def _bulk_hide_low_line_only(from_text: str, to_text: str, threshold: float | None) -> str:
        report = hide_low_confidence_line_only(
            _repository(),
            ReviewQueueFilters(date_from=_blank_or_none(from_text), date_to=_blank_or_none(to_text)),
            threshold=threshold if threshold is not None else 0.5,
        )
        return f"hidden={report['updated_count']}, missing={report['missing_count']}"

    def _make_eval_case(event_id: str | None, query_text: str, expected_date: str) -> str:
        return make_eval_case_yaml(
            _repository(),
            event_id=event_id or None,
            query=_blank_or_none(query_text),
            expected_date=_blank_or_none(expected_date),
        )

    def _load_vlm_review_queue(
        month_text: str,
        date_text: str,
        from_text: str,
        to_text: str,
        status_value: str,
        unreviewed: bool,
        safety_flags: bool,
        food_cues: bool,
        performance_stage: bool,
        has_ocr: bool,
        has_embedding: bool,
        event_linked: bool,
        failed_only: bool,
        low_confidence: float | None,
        limit_value: float | None,
    ):
        status = None if status_value == "all" else status_value
        resolved_from, resolved_to = _month_or_dates(month_text, from_text, to_text)
        rows = list_vlm_review_items(
            _repository(),
            VlmReviewFilters(
                date=_blank_or_none(date_text),
                date_from=resolved_from,
                date_to=resolved_to,
                review_status=status,
                unreviewed=unreviewed,
                safety_flags=safety_flags,
                food_cues=food_cues,
                has_ocr=has_ocr,
                has_embedding=has_embedding,
                event_linked=True if event_linked else None,
                low_confidence=low_confidence,
                limit=_int_or_none(limit_value) or 100,
            ),
        )
        if performance_stage:
            rows = [
                row
                for row in rows
                if _contains_any(row, ("performance", "stage", "theater", "dancing", "パフォーマンス", "ステージ"))
            ]
        if failed_only:
            rows = [row for row in rows if str(row.get("status") or "") == "failed"]
        choices = [str(row["media_id"]) for row in rows if row.get("media_id")]
        selected = choices[0] if choices else None
        return (
            f"vlm review items={len(rows)}",
            vlm_review_rows_for_dataframe(rows),
            gr.update(choices=choices, value=selected),
            *_vlm_detail_values(selected),
        )

    def _vlm_detail_values(media_id: str | None):
        if not media_id:
            return _empty_vlm_detail_values()
        detail = get_vlm_review_detail(_repository(), media_id)
        if not detail:
            return _empty_vlm_detail_values()
        return (
            str(detail.get("caption") or ""),
            str(detail.get("short_caption") or ""),
            _tags_text(detail.get("scene_tags_json")),
            _tags_text(detail.get("object_tags_json")),
            _tags_text(detail.get("activity_tags_json")),
            _tags_text(detail.get("food_cues_json")),
            _tags_text(detail.get("location_cues_json")),
            str(detail.get("review_status") or "unreviewed"),
            str(detail.get("review_note") or ""),
            bool(detail.get("is_verified")),
            bool(detail.get("is_hidden")),
            bool(detail.get("is_wrong")),
            bool(detail.get("is_searchable", 1)),
            bool(detail.get("is_event_usable", 1)),
            str(detail.get("ocr_text_redacted") or detail.get("ocr_text") or ""),
            _related_events_text(detail.get("related_events") or []),
            "",
        )

    def _save_vlm_review(
        media_id: str | None,
        caption: str,
        short_caption: str,
        scene_tags: str,
        object_tags: str,
        activity_tags: str,
        food_cues: str,
        location_cues: str,
        review_status: str,
        review_note: str,
        is_verified: bool,
        is_hidden: bool,
        is_wrong: bool,
        is_searchable: bool,
        is_event_usable: bool,
    ):
        if not media_id:
            return (*_empty_vlm_detail_values()[:-1], "VLM結果が選択されていません。")
        save_vlm_override(
            _repository(),
            VlmOverrideUpdate(
                media_id=media_id,
                caption_override=caption,
                short_caption_override=short_caption,
                scene_tags_override=parse_tag_text(scene_tags),
                object_tags_override=parse_tag_text(object_tags),
                activity_tags_override=parse_tag_text(activity_tags),
                food_cues_override=parse_tag_text(food_cues),
                location_cues_override=parse_tag_text(location_cues),
                review_status=review_status or "unreviewed",
                review_note=review_note,
                is_verified=is_verified,
                is_hidden=is_hidden,
                is_wrong=is_wrong,
                is_searchable=is_searchable,
                is_event_usable=is_event_usable,
            ),
        )
        values = _vlm_detail_values(media_id)
        return (*values[:-1], "VLM overrideを保存しました。")

    def _quick_vlm_action(media_id: str | None, action: str):
        if not media_id:
            return (*_empty_vlm_detail_values()[:-1], "VLM結果が選択されていません。")
        updates: dict[str, Any] = {"media_id": media_id}
        if action == "accepted":
            updates.update(review_status="accepted", is_verified=True, is_hidden=False, is_wrong=False, is_searchable=True, is_event_usable=True)
        elif action == "rejected":
            updates.update(review_status="rejected", is_searchable=False, is_event_usable=False)
        elif action == "wrong":
            updates.update(review_status="wrong", is_wrong=True, is_searchable=False, is_event_usable=False)
        elif action == "verified":
            updates.update(is_verified=True)
        elif action == "hidden":
            updates.update(is_hidden=True)
        elif action == "search_off":
            updates.update(is_searchable=False)
        elif action == "event_off":
            updates.update(is_event_usable=False)
        save_vlm_override(_repository(), VlmOverrideUpdate(**updates))
        values = _vlm_detail_values(media_id)
        return (*values[:-1], f"{action} を反映しました。")

    def _clear_vlm_review(media_id: str | None):
        if not media_id:
            return (*_empty_vlm_detail_values()[:-1], "VLM結果が選択されていません。")
        clear_vlm_override(_repository(), media_id)
        values = _vlm_detail_values(media_id)
        return (*values[:-1], "VLM overrideを削除しました。")

    def _bulk_vlm_update(ids_text: str, action: str, tag_text: str) -> str:
        ids = [line.strip() for line in ids_text.splitlines() if line.strip()]
        kwargs: dict[str, Any] = {}
        if action == "accepted":
            kwargs.update(review_status="accepted", is_verified=True, is_searchable=True, is_event_usable=True)
        elif action == "rejected":
            kwargs.update(review_status="rejected", is_searchable=False, is_event_usable=False)
        elif action == "wrong":
            kwargs.update(review_status="wrong", is_wrong=True, is_searchable=False, is_event_usable=False)
        elif action == "hidden":
            kwargs.update(is_hidden=True)
        elif action == "not_searchable":
            kwargs.update(is_searchable=False)
        elif action == "not_event_usable":
            kwargs.update(is_event_usable=False)
        elif action == "tag":
            kwargs.update(add_tags=parse_tag_text(tag_text))
        report = bulk_update_vlm_overrides(_repository(), ids, **kwargs)
        return f"updated={report['updated']}"

    def _make_vlm_eval_case(media_id: str | None, query_text: str) -> str:
        return generate_vlm_eval_case(
            media_id=media_id or None,
            query=_blank_or_none(query_text),
            expected_media_id=media_id or None,
        )

    def _load_place_review_queue(
        from_text: str,
        to_text: str,
        min_points_value: float | None,
        status_value: str,
        privacy_value: str,
        category_value: str,
        search_text: str,
        mode_value: str,
        limit_value: float | None,
    ):
        payload = place_review_queue_for_ui(
            _repository(),
            date_from=_blank_or_none(from_text),
            date_to=_blank_or_none(to_text),
            min_points=_int_or_none(min_points_value),
            status=None if status_value == "all" else status_value,
            privacy_level=None if privacy_value == "all" else privacy_value,
            category=None if category_value == "all" else category_value,
            search_text=_blank_or_none(search_text),
            limit=_int_or_none(limit_value) or 50,
            mode=mode_value,
        )
        selected = payload["cluster_ids"][0] if payload["cluster_ids"] else None
        detail = _place_cluster_detail_values(selected, mode_value == "private")
        return (
            f"place clusters={len(payload['rows'])}",
            payload["table"],
            gr.update(choices=payload["cluster_ids"], value=selected),
            *detail,
        )

    def _place_cluster_detail_values(cluster_id: str | None, private_mode: bool = False):
        detail = place_cluster_detail_for_ui(_repository(), cluster_id, private_mode=private_mode)
        return (
            detail.get("summary", ""),
            detail.get("gallery", []),
            detail.get("photos", []),
            detail.get("events", []),
            detail.get("linked_place_id", ""),
            detail.get("linked_display_name", ""),
            detail.get("linked_public_name", ""),
            detail.get("linked_category", ""),
            detail.get("linked_privacy_level", ""),
            detail.get("linked_aliases", ""),
            "",
        )

    def _place_detail_ui(cluster_id: str | None, private_mode: bool):
        return _place_cluster_detail_values(cluster_id, private_mode)

    def _create_place_ui(
        cluster_id: str | None,
        display_name: str,
        public_name: str,
        category: str,
        privacy_level: str,
        aliases_text: str,
        notes: str,
    ):
        if not display_name.strip():
            return "display_name が必要です。"
        return create_place_for_ui(
            _repository(),
            display_name=display_name.strip(),
            public_name=_blank_or_none(public_name),
            category=category,
            privacy_level=privacy_level,
            cluster_id=cluster_id or None,
            aliases_text=aliases_text,
            manual_verified=True,
            notes=_blank_or_none(notes),
        )

    def _update_place_ui(
        place_id: str,
        display_name: str,
        public_name: str,
        category: str,
        privacy_level: str,
        notes: str,
    ):
        if not place_id.strip():
            return "place_id が必要です。"
        return update_place_for_ui(
            _repository(),
            place_id=place_id.strip(),
            display_name=_blank_or_none(display_name),
            public_name=_blank_or_none(public_name),
            category=_blank_or_none(category),
            privacy_level=_blank_or_none(privacy_level),
            manual_verified=True,
            notes=_blank_or_none(notes),
        )

    def _link_place_ui(place_id: str, cluster_id: str | None):
        if not place_id.strip() or not cluster_id:
            return "place_id と cluster_id が必要です。"
        return link_cluster_for_ui(_repository(), place_id=place_id.strip(), cluster_id=cluster_id)

    def _alias_place_ui(place_id: str, alias: str):
        if not place_id.strip() or not alias.strip():
            return "place_id と alias が必要です。"
        return add_place_alias_for_ui(_repository(), place_id=place_id.strip(), alias=alias.strip())

    def _place_cluster_action_ui(cluster_id: str | None, action: str):
        if not cluster_id:
            return "cluster が選択されていません。"
        if action == "accepted":
            return set_cluster_status_for_ui(_repository(), cluster_id=cluster_id, status="accepted")
        if action == "rejected":
            return set_cluster_status_for_ui(_repository(), cluster_id=cluster_id, status="rejected")
        if action == "unlink":
            return unlink_cluster_for_ui(_repository(), cluster_id=cluster_id)
        return "unknown action"

    def _reassign_places_ui(from_text: str, to_text: str) -> str:
        return reassign_places_for_ui(
            _repository(),
            date_from=_blank_or_none(from_text),
            date_to=_blank_or_none(to_text),
        )

    def _load_face_review_queue(
        from_text: str,
        to_text: str,
        status_value: str,
        review_status_value: str,
        limit_value: float | None,
        show_private_crops: bool,
    ):
        payload = face_review_queue_for_ui(
            _repository(),
            date_from=_blank_or_none(from_text),
            date_to=_blank_or_none(to_text),
            status=None if status_value == "all" else status_value,
            review_status=None if review_status_value == "all" else review_status_value,
            limit=_int_or_none(limit_value) or 50,
            show_private_crops=show_private_crops,
        )
        selected = payload["face_ids"][0] if payload["face_ids"] else None
        detail = _face_detail_values(selected, show_private_crops=show_private_crops)
        hint = ""
        if not payload["rows"] and status_value == "success":
            hint = " / success rows are empty; choose no_face_detected/all to inspect non-face media records"
        return (
            f"face detections={len(payload['rows'])}{hint}",
            payload["table"],
            payload["gallery"],
            gr.update(choices=payload["face_ids"], value=selected),
            payload["face_ids"],
            payload["gallery_face_ids"],
            *detail,
        )

    def _face_detail_values(face_id: str | None, show_private_crops: bool = True):
        detail = face_detail_for_ui(_repository(), face_id, show_private_crops=show_private_crops)
        return (
            detail.get("summary", ""),
            detail.get("face_thumbnail"),
            detail.get("original_thumbnail"),
            detail.get("file_name", ""),
            detail.get("crop_note", ""),
            detail.get("review_status", ""),
            "",
        )

    def _face_detail_ui(face_id: str | None, show_private_crops: bool):
        return _face_detail_values(face_id, show_private_crops)

    def _face_next_ui(face_id: str | None, face_ids: list[str] | None, show_private_crops: bool):
        next_id = next_face_id(face_ids, face_id)
        return (gr.update(value=next_id), *_face_detail_values(next_id, show_private_crops))

    def _face_prev_ui(face_id: str | None, face_ids: list[str] | None, show_private_crops: bool):
        prev_id = previous_face_id(face_ids, face_id)
        return (gr.update(value=prev_id), *_face_detail_values(prev_id, show_private_crops))

    def _select_index_from_gallery(evt) -> int | None:
        index = getattr(evt, "index", None)
        if isinstance(index, (list, tuple)):
            index = index[0] if index else None
        try:
            return int(index) if index is not None else None
        except (TypeError, ValueError):
            return None

    def _id_from_gallery(ids: list[str] | None, evt) -> str | None:
        index = _select_index_from_gallery(evt)
        if index is None:
            return None
        item_ids = [str(item_id) for item_id in (ids or []) if item_id]
        if index < 0 or index >= len(item_ids):
            return None
        return item_ids[index]

    def _face_gallery_select(face_gallery_ids: list[str] | None, show_private_crops: bool, evt: gr.SelectData):
        face_id = _id_from_gallery(face_gallery_ids, evt)
        return (gr.update(value=face_id), *_face_detail_values(face_id, show_private_crops))

    def _face_review_action(face_id: str | None, review_status: str):
        message, detail = update_face_review_for_ui(_repository(), face_id, review_status)
        return (
            detail.get("summary", ""),
            detail.get("face_thumbnail"),
            detail.get("original_thumbnail"),
            detail.get("file_name", ""),
            detail.get("crop_note", ""),
            detail.get("review_status", ""),
            message,
        )

    def _face_accept_and_next(face_id: str | None, face_ids: list[str] | None, show_private_crops: bool):
        if not face_id:
            detail_values = _face_detail_values(face_id, show_private_crops)
            return (gr.update(value=face_id), *detail_values[:-1], "face_id is required")
        message, _detail = update_face_review_for_ui(_repository(), face_id, "accepted")
        next_id = next_face_id(face_ids, face_id)
        detail_values = _face_detail_values(next_id, show_private_crops)
        status = f"{message}; moved to next face_id={next_id}" if next_id and next_id != face_id else message
        return (gr.update(value=next_id), *detail_values[:-1], status)

    def _load_face_clusters(status_value: str, limit_value: float | None, public_mode: bool, show_private_crops: bool):
        payload = face_cluster_review_for_ui(
            _repository(),
            status=None if status_value == "all" else status_value,
            limit=_int_or_none(limit_value) or 50,
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
        selected = payload["cluster_ids"][0] if payload["cluster_ids"] else None
        detail = _face_cluster_detail_values(selected, show_private_crops=True, public_mode=public_mode)
        people = persons_for_ui(_repository(), public_mode=public_mode)
        return (
            f"face clusters={len(payload['rows'])}",
            payload["table"],
            payload["gallery"],
            payload["gallery_cluster_ids"],
            gr.update(choices=payload["cluster_ids"], value=selected),
            payload["cluster_ids"],
            gr.update(choices=payload["cluster_ids"], value=selected),
            people["table"],
            gr.update(choices=people["person_ids"], value=people["person_ids"][0] if people["person_ids"] else None),
            people["person_ids"],
            *detail,
        )

    def _face_cluster_detail_values(cluster_id: str | None, show_private_crops: bool, public_mode: bool):
        detail = face_cluster_detail_for_ui(
            _repository(),
            cluster_id,
            show_private_crops=show_private_crops,
            public_mode=public_mode,
        )
        linked = detail.get("linked_people") or []
        linked_text = ", ".join(person.get("display_name") or "" for person in linked) if linked else ""
        return (
            detail.get("summary", ""),
            detail.get("member_table", []),
            detail.get("member_thumbnails", []),
            detail.get("member_thumbnail_face_ids", []),
            linked_text,
            detail.get("privacy_note", ""),
            "",
        )

    def _face_cluster_detail_ui(cluster_id: str | None, show_private_crops: bool, public_mode: bool):
        return _face_cluster_detail_values(cluster_id, show_private_crops, public_mode)

    def _face_cluster_next_ui(cluster_id: str | None, cluster_ids: list[str] | None, show_private_crops: bool, public_mode: bool):
        next_id = next_cluster_id(cluster_ids, cluster_id)
        return (gr.update(value=next_id), *_face_cluster_detail_values(next_id, show_private_crops, public_mode))

    def _face_cluster_prev_ui(cluster_id: str | None, cluster_ids: list[str] | None, show_private_crops: bool, public_mode: bool):
        prev_id = previous_cluster_id(cluster_ids, cluster_id)
        return (gr.update(value=prev_id), *_face_cluster_detail_values(prev_id, show_private_crops, public_mode))

    def _face_cluster_gallery_select(cluster_gallery_ids: list[str] | None, show_private_crops: bool, public_mode: bool, evt: gr.SelectData):
        cluster_id = _id_from_gallery(cluster_gallery_ids, evt)
        return (gr.update(value=cluster_id), *_face_cluster_detail_values(cluster_id, show_private_crops, public_mode))

    def _face_member_gallery_select(member_face_ids: list[str] | None, show_private_crops: bool, evt: gr.SelectData):
        face_id = _id_from_gallery(member_face_ids, evt)
        return (gr.update(value=face_id), *_face_detail_values(face_id, show_private_crops))

    def _face_cluster_action(cluster_id: str | None, status: str):
        message, detail = face_cluster_action_for_ui(_repository(), cluster_id, status)
        linked = detail.get("linked_people") or []
        linked_text = ", ".join(person.get("display_name") or "" for person in linked) if linked else ""
        return (
            detail.get("summary", ""),
            detail.get("member_table", []),
            detail.get("member_thumbnails", []),
            detail.get("member_thumbnail_face_ids", []),
            linked_text,
            detail.get("privacy_note", ""),
            message,
        )

    def _face_cluster_accept_and_next(
        cluster_id: str | None,
        cluster_ids: list[str] | None,
        show_private_crops: bool,
        public_mode: bool,
    ):
        if not cluster_id:
            detail_values = _face_cluster_detail_values(cluster_id, show_private_crops, public_mode)
            return (gr.update(value=cluster_id), *detail_values[:-1], "cluster_id is required")
        message, _detail = face_cluster_action_for_ui(_repository(), cluster_id, "accepted")
        next_id = next_cluster_id(cluster_ids, cluster_id)
        detail_values = _face_cluster_detail_values(next_id, show_private_crops, public_mode)
        status = f"{message}; moved to next cluster_id={next_id}" if next_id and next_id != cluster_id else message
        return (gr.update(value=next_id), *detail_values[:-1], status)

    def _create_face_person(name: str | None, public_name: str | None, privacy_level: str):
        message, table, ids = create_person_for_ui(_repository(), name, public_name, privacy_level)
        return message, table, gr.update(choices=ids, value=ids[0] if ids else None), ids

    def _update_face_person(person_id: str | None, name: str | None, public_name: str | None, privacy_level: str):
        message, table, ids = update_person_for_ui(_repository(), person_id, name, public_name, privacy_level)
        return message, table, gr.update(choices=ids, value=person_id if person_id in ids else (ids[0] if ids else None)), ids

    def _person_form_values(person_id: str | None):
        detail = person_form_for_ui(_repository(), person_id)
        return (
            detail.get("display_name", ""),
            detail.get("public_name", ""),
            gr.update(value=detail.get("privacy_level", "private")),
        )

    def _person_next_ui(person_id: str | None, person_ids: list[str] | None):
        next_id = next_person_id(person_ids, person_id)
        return (gr.update(value=next_id), *_person_form_values(next_id))

    def _person_prev_ui(person_id: str | None, person_ids: list[str] | None):
        prev_id = previous_person_id(person_ids, person_id)
        return (gr.update(value=prev_id), *_person_form_values(prev_id))

    def _person_cluster_overview_values(public_mode: bool, show_private_crops: bool):
        payload = person_cluster_overview_for_ui(
            _repository(),
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
        return (
            payload["summary"],
            payload["table"],
            payload["gallery"],
            payload["gallery_cluster_ids"],
            payload["gallery_person_ids"],
        )

    def _organizer_action_values(result: dict[str, object], public_mode: bool, show_private_crops: bool):
        overview = result["overview"]
        person_ids = result["person_ids"]
        return (
            result["message"],
            overview["summary"],
            overview["table"],
            overview["gallery"],
            overview["gallery_cluster_ids"],
            overview["gallery_person_ids"],
            result["person_table"],
            gr.update(choices=person_ids, value=person_ids[0] if person_ids else None),
            person_ids,
        )

    def _organizer_link_cluster_to_person(
        person_id: str | None,
        cluster_id: str | None,
        public_mode: bool,
        show_private_crops: bool,
    ):
        result = link_cluster_to_person_for_ui(
            _repository(),
            person_id,
            cluster_id,
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
        return _organizer_action_values(result, public_mode, show_private_crops)

    def _organizer_unlink_cluster_from_person(
        person_id: str | None,
        cluster_id: str | None,
        public_mode: bool,
        show_private_crops: bool,
    ):
        result = unlink_cluster_from_person_for_ui(
            _repository(),
            person_id,
            cluster_id,
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
        return _organizer_action_values(result, public_mode, show_private_crops)

    def _organizer_merge_cluster_into_target(
        source_cluster_id: str | None,
        target_cluster_id: str | None,
        public_mode: bool,
        show_private_crops: bool,
    ):
        result = merge_cluster_into_target_person_for_ui(
            _repository(),
            source_cluster_id,
            target_cluster_id,
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
        return _organizer_action_values(result, public_mode, show_private_crops)

    def _organizer_gallery_select(
        organizer_cluster_ids: list[str] | None,
        organizer_person_ids: list[str] | None,
        show_private_crops: bool,
        public_mode: bool,
        evt: gr.SelectData,
    ):
        index = _select_index_from_gallery(evt)
        cluster_id = None
        person_id = None
        if index is not None:
            cluster_items = [str(item_id) for item_id in (organizer_cluster_ids or []) if item_id]
            person_items = [str(item_id) for item_id in (organizer_person_ids or []) if item_id]
            if 0 <= index < len(cluster_items):
                cluster_id = cluster_items[index]
            if 0 <= index < len(person_items):
                person_id = person_items[index]
        return (
            gr.update(value=cluster_id),
            gr.update(value=person_id),
            *_person_form_values(person_id),
            *_face_cluster_detail_values(cluster_id, show_private_crops, public_mode),
        )

    def _link_face_person(person_id: str | None, cluster_id: str | None) -> str:
        return link_person_cluster_for_ui(_repository(), person_id, cluster_id)

    def _unlink_face_person(person_id: str | None, cluster_id: str | None) -> str:
        return unlink_person_cluster_for_ui(_repository(), person_id, cluster_id)

    def _label_selected_face_cluster(
        cluster_id: str | None,
        name: str | None,
        public_name: str | None,
        privacy_level: str,
    ):
        result = label_face_cluster_for_ui(_repository(), cluster_id, name, public_name, privacy_level)
        detail = result["cluster_detail"]
        linked = detail.get("linked_people") or []
        linked_text = ", ".join(person.get("display_name") or "" for person in linked) if linked else ""
        return (
            result["message"],
            result["person_table"],
            gr.update(
                choices=result["person_ids"],
                value=result["person_id"] or (result["person_ids"][0] if result["person_ids"] else None),
            ),
            result["person_ids"],
            detail.get("summary", ""),
            detail.get("member_table", []),
            detail.get("member_thumbnails", []),
            detail.get("member_thumbnail_face_ids", []),
            linked_text,
            detail.get("privacy_note", ""),
            result["message"],
        )

    def _add_face_person_alias(person_id: str | None, alias: str | None) -> str:
        return add_person_alias_for_ui(_repository(), person_id, alias)

    def _load_line_speakers_ui(limit_value: float | None):
        payload = line_speakers_for_ui(_repository(), limit=_int_or_none(limit_value) or 100)
        return f"line speakers={len(payload['rows'])}", payload["table"]

    def _link_line_speaker_ui(chat_id: str | None, speaker_name: str | None, person_id: str | None, add_alias: bool) -> str:
        return link_line_speaker_for_ui(_repository(), chat_id, speaker_name, person_id, add_alias)

    def _unlink_line_speaker_ui(chat_id: str | None, speaker_name: str | None, person_id: str | None) -> str:
        return unlink_line_speaker_for_ui(_repository(), chat_id, speaker_name, person_id)

    with gr.Blocks(**blocks_kwargs) as app:
        gr.Markdown(
            """
<div class="photos-hero">
  <div class="kicker">LOCAL PHOTO MEMORY</div>
  <h1>Personal LifeLog RAG</h1>
  <p>ローカルで写真・LINE・GPS・画像解析を見返すためのプライベートUI</p>
</div>
""",
            elem_classes=["photos-hero-markdown"],
        )

        with gr.Tab("Home / Stats"):
            db_path_box = gr.Textbox(label="DB path", value=str(resolved_db_path), interactive=False)
            with gr.Row():
                media_count_box = gr.Number(label="Registered photos", precision=0, interactive=False)
                line_count_box = gr.Number(label="Registered LINE messages", precision=0, interactive=False)
                event_count_box = gr.Number(label="Registered events", precision=0, interactive=False)
            refresh_stats = gr.Button("Refresh Stats", variant="secondary")
            refresh_stats.click(
                _stats_values,
                outputs=[db_path_box, media_count_box, line_count_box, event_count_box],
            )

        with gr.Tab("Ingest"):
            photo_path = gr.Textbox(label="Photo folder path", value="data/raw/photos")
            line_path = gr.Textbox(label="LINE folder path", value="data/raw/line")
            with gr.Row():
                ingest_photos_button = gr.Button("Ingest Photos", variant="secondary")
                ingest_line_button = gr.Button("Ingest LINE", variant="secondary")
                ingest_all_button = gr.Button("Ingest Both", variant="primary")
            photo_result = gr.Textbox(label="Photo ingest result", interactive=False)
            line_result = gr.Textbox(label="LINE ingest result", interactive=False)

            ingest_photos_button.click(_ingest_photos, inputs=photo_path, outputs=photo_result)
            ingest_line_button.click(_ingest_line, inputs=line_path, outputs=line_result)
            ingest_all_button.click(
                _ingest_all,
                inputs=[photo_path, line_path],
                outputs=[
                    photo_result,
                    line_result,
                    db_path_box,
                    media_count_box,
                    line_count_box,
                    event_count_box,
                ],
            )
            gr.Markdown("## OCR")
            with gr.Row():
                ocr_date = gr.Textbox(label="OCR date", placeholder="YYYY-MM-DD")
                ocr_limit = gr.Number(label="limit", value=20, precision=0)
                ocr_engine = gr.Dropdown(
                    label="engine",
                    choices=["tesseract_cli", "pytesseract", "noop", "fake"],
                    value="tesseract_cli",
                )
                ocr_dry_run = gr.Checkbox(label="dry-run", value=True)
            with gr.Row():
                ocr_run_button = gr.Button("Run OCR", variant="primary")
                ocr_stats_button = gr.Button("OCR Stats", variant="secondary")
            ocr_result = gr.Textbox(label="OCR result", lines=10, interactive=False)
            ocr_run_button.click(
                _run_ocr_ui,
                inputs=[ocr_date, ocr_limit, ocr_engine, ocr_dry_run],
                outputs=ocr_result,
            )
            ocr_stats_button.click(_ocr_stats_ui, inputs=ocr_date, outputs=ocr_result)

            gr.Markdown("## VLM / Image Analysis")
            with gr.Row():
                vlm_date = gr.Textbox(label="VLM date", placeholder="YYYY-MM-DD")
                vlm_limit = gr.Number(label="limit", value=10, precision=0)
                vlm_engine = gr.Dropdown(
                    label="engine",
                    choices=["noop", "fake", "ollama", "transformers", "llama_cpp"],
                    value="noop",
                )
                vlm_dry_run = gr.Checkbox(label="dry-run", value=True)
            with gr.Row():
                vlm_run_button = gr.Button("Run VLM", variant="primary")
                vlm_stats_button = gr.Button("VLM Stats", variant="secondary")
            vlm_result = gr.Textbox(label="VLM result", lines=10, interactive=False)
            vlm_run_button.click(
                _run_vlm_ui,
                inputs=[vlm_date, vlm_limit, vlm_engine, vlm_dry_run],
                outputs=vlm_result,
            )
            vlm_stats_button.click(_vlm_stats_ui, inputs=vlm_date, outputs=vlm_result)

        with gr.Tab("Ask"):
            question = gr.Textbox(label="Question", value="2024年12月24日は何していた？")
            ask_button = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", lines=14, interactive=False)
            line_evidence = gr.Dataframe(
                headers=["sent_at", "sender", "type", "text"],
                label="Evidence LINE",
                interactive=False,
            )
            photo_evidence = gr.Dataframe(
                headers=["captured_at", "file_name", "thumbnail_path", "gps"],
                label="Evidence Photos",
                interactive=False,
            )
            photo_evidence_gallery = gr.Gallery(
                label="Evidence Photos thumbnails",
                columns=6,
                height=430,
                elem_classes=["photo-grid"],
            )
            ask_button.click(
                _ask,
                inputs=question,
                outputs=[
                    answer,
                    line_evidence,
                    photo_evidence,
                    photo_evidence_gallery,
                    db_path_box,
                    media_count_box,
                    line_count_box,
                    event_count_box,
                ],
            )

        with gr.Tab("Monthly Summary"):
            with gr.Row():
                monthly_month = gr.Textbox(label="年月", value="2025-01", placeholder="YYYY-MM")
                monthly_from = gr.Textbox(label="from date", placeholder="YYYY-MM-DD")
                monthly_to = gr.Textbox(label="to date", placeholder="YYYY-MM-DD")
                monthly_mode = gr.Dropdown(label="表示モード", choices=["public", "private"], value="public")
                monthly_include_hidden = gr.Checkbox(label="include hidden (private only)", value=False)
            monthly_button = gr.Button("月次要約を表示", variant="primary")
            monthly_text = gr.Textbox(label="月次要約本文", lines=16, interactive=False)
            monthly_metrics = gr.Dataframe(headers=["metric", "value"], label="集計", interactive=False)
            monthly_titles = gr.Dataframe(headers=["title", "count"], label="title分布", interactive=False)
            monthly_days = gr.Dataframe(
                headers=["date", "events", "photos", "gps_photos", "line", "calls", "vlm", "ocr"],
                label="代表日 top5",
                interactive=False,
            )
            monthly_events = gr.Dataframe(
                headers=["date", "time", "title", "summary", "confidence", "line", "photo", "ocr", "vlm"],
                label="代表イベント",
                interactive=False,
            )
            monthly_button.click(
                _monthly_summary_ui,
                inputs=[monthly_month, monthly_from, monthly_to, monthly_mode, monthly_include_hidden],
                outputs=[monthly_text, monthly_metrics, monthly_titles, monthly_days, monthly_events],
            )

        with gr.Tab("Image Search"):
            image_query = gr.Textbox(label="Image query", value="ラーメン")
            image_limit = gr.Number(label="limit", value=20, precision=0)
            image_search_button = gr.Button("Search Images", variant="primary")
            image_search_summary = gr.Textbox(label="Result summary", lines=10, interactive=False)
            image_search_table = gr.Dataframe(
                headers=[
                    "date",
                    "media_id",
                    "file_name",
                    "captured_at",
                    "caption",
                    "OCR",
                    "matched fields",
                    "related_persons",
                    "person_evidence",
                    "person_score",
                    "thumbnail_path",
                ],
                label="Image results",
                interactive=False,
            )
            image_search_gallery = gr.Gallery(
                label="Result thumbnails",
                columns=6,
                height=430,
                elem_classes=["photo-grid"],
            )
            image_search_button.click(
                _image_search_ui,
                inputs=[image_query, image_limit],
                outputs=[image_search_summary, image_search_table, image_search_gallery],
            )

        with gr.Tab("Multimodal Search"):
            mm_query = gr.Textbox(label="Query", value="ご飯を食べた写真")
            with gr.Row():
                mm_person_filter = gr.Textbox(label="person filter", placeholder="手動確認済みperson名またはpublic_name")
                mm_place_filter = gr.Textbox(label="place filter", placeholder="手動場所ラベルまたはalias")
                mm_activity_query = gr.Textbox(label="activity query", placeholder="ご飯 / カフェ / ステージ など")
            with gr.Row():
                mm_backend = gr.Dropdown(label="backend", choices=["sql", "vlm_sql", "embedding", "hybrid"], value="hybrid")
                mm_from = gr.Textbox(label="date_from", placeholder="YYYY-MM-DD")
                mm_to = gr.Textbox(label="date_to", placeholder="YYYY-MM-DD")
                mm_limit = gr.Number(label="limit", value=10, precision=0)
                mm_include_hidden = gr.Checkbox(label="include_hidden", value=False)
                mm_include_sensitive = gr.Checkbox(label="include_sensitive_vlm", value=False)
            mm_button = gr.Button("Search", variant="primary")
            mm_summary = gr.Textbox(label="Summary", lines=12, interactive=False)
            mm_table = gr.Dataframe(
                headers=[
                    "rank",
                    "media_id",
                    "date",
                    "captured_at",
                    "score",
                    "confidence",
                    "evidence_strength",
                    "caption",
                    "matched_terms",
                    "food_cues",
                    "location_cues",
                    "related_event",
                    "related_persons",
                    "person_evidence",
                    "thumbnail_path",
                    "score_components",
                    "review_status",
                ],
                label="Multimodal results",
                interactive=False,
            )
            mm_media_ids_state = gr.State([])
            with gr.Row():
                selected_mm_media_id = gr.Dropdown(
                    label="Search Result Detail media_id",
                    choices=[],
                    value=None,
                    scale=12,
                    allow_custom_value=True,
                )
                mm_prev_media = gr.Button("↑", variant="secondary", min_width=48)
                mm_next_media = gr.Button("↓", variant="secondary", min_width=48)
            mm_thumbnail = gr.Image(
                label="thumbnail",
                type="filepath",
                interactive=False,
                elem_classes=["thumbnail-focus"],
            )
            with gr.Row():
                mm_file_name = gr.Textbox(label="file_name", interactive=False)
                mm_captured_at = gr.Textbox(label="captured_at", interactive=False)
                mm_review_status = gr.Textbox(label="review_status", interactive=False)
            mm_caption = gr.Textbox(label="VLM caption", lines=3, interactive=False)
            mm_ocr_text = gr.Textbox(label="OCR text", lines=3, interactive=False)
            mm_evidence = gr.Textbox(label="evidence", lines=5, interactive=False)
            mm_score_components = gr.Textbox(label="score_components", lines=3, interactive=False)
            with gr.Row():
                mm_accept = gr.Button("Mark accepted", variant="primary")
                mm_wrong = gr.Button("Mark wrong", variant="stop")
                mm_hide = gr.Button("Hide", variant="secondary")
                mm_not_searchable = gr.Button("Not searchable", variant="secondary")
                mm_not_event_usable = gr.Button("Not event usable", variant="secondary")
            mm_detail_status = gr.Textbox(label="review action status", interactive=False)
            mm_detail_outputs = [
                mm_thumbnail,
                mm_file_name,
                mm_captured_at,
                mm_caption,
                mm_ocr_text,
                mm_evidence,
                mm_review_status,
                mm_score_components,
            ]
            mm_button.click(
                _multimodal_search_ui,
                inputs=[
                    mm_query,
                    mm_person_filter,
                    mm_place_filter,
                    mm_activity_query,
                    mm_backend,
                    mm_from,
                    mm_to,
                    mm_limit,
                    mm_include_hidden,
                    mm_include_sensitive,
                ],
                outputs=[mm_summary, mm_table, selected_mm_media_id, mm_media_ids_state, *mm_detail_outputs],
            )
            selected_mm_media_id.change(_multimodal_detail_ui, inputs=selected_mm_media_id, outputs=mm_detail_outputs)
            mm_prev_media.click(
                lambda media_id, media_ids: _multimodal_adjacent_ui(media_id, media_ids, -1),
                inputs=[selected_mm_media_id, mm_media_ids_state],
                outputs=[selected_mm_media_id, *mm_detail_outputs],
            )
            mm_next_media.click(
                lambda media_id, media_ids: _multimodal_adjacent_ui(media_id, media_ids, 1),
                inputs=[selected_mm_media_id, mm_media_ids_state],
                outputs=[selected_mm_media_id, *mm_detail_outputs],
            )
            mm_accept.click(lambda media_id: _multimodal_quick_action(media_id, "accepted"), inputs=selected_mm_media_id, outputs=[*mm_detail_outputs, mm_detail_status])
            mm_wrong.click(lambda media_id: _multimodal_quick_action(media_id, "wrong"), inputs=selected_mm_media_id, outputs=[*mm_detail_outputs, mm_detail_status])
            mm_hide.click(lambda media_id: _multimodal_quick_action(media_id, "hidden"), inputs=selected_mm_media_id, outputs=[*mm_detail_outputs, mm_detail_status])
            mm_not_searchable.click(lambda media_id: _multimodal_quick_action(media_id, "not_searchable"), inputs=selected_mm_media_id, outputs=[*mm_detail_outputs, mm_detail_status])
            mm_not_event_usable.click(lambda media_id: _multimodal_quick_action(media_id, "not_event_usable"), inputs=selected_mm_media_id, outputs=[*mm_detail_outputs, mm_detail_status])

        with gr.Tab("Report Viewer"):
            with gr.Row():
                report_from = gr.Textbox(label="from", placeholder="YYYY-MM-DD")
                report_to = gr.Textbox(label="to", placeholder="YYYY-MM-DD")
                report_mode = gr.Dropdown(label="mode", choices=["public", "private"], value="public")
                report_examples = gr.Checkbox(label="include examples", value=False)
                report_save_json = gr.Checkbox(label="save JSON", value=True)
            with gr.Row():
                report_refresh = gr.Button("レポート一覧を更新", variant="secondary")
                report_generate = gr.Button("generate-report", variant="primary")
            initial_reports = list_reports_for_ui()
            report_choice = gr.Dropdown(
                label="reports/*.md",
                choices=initial_reports,
                value=initial_reports[0] if initial_reports else None,
            )
            report_load = gr.Button("選択レポートを読み込む", variant="secondary")
            report_path_box = gr.Textbox(label="generated / loaded report path", interactive=False)
            report_markdown = gr.Markdown(label="Markdown preview")
            report_json = gr.Textbox(label="JSON summary", lines=10, interactive=False)
            report_refresh.click(_report_choices, outputs=report_choice)
            report_load.click(_load_report_ui, inputs=report_choice, outputs=[report_markdown, report_json, report_path_box])
            report_generate.click(
                _generate_report_ui,
                inputs=[report_from, report_to, report_mode, report_examples, report_save_json],
                outputs=[report_choice, report_markdown, report_json, report_path_box],
            )

        with gr.Tab("Place Review / 場所レビュー"):
            gr.Markdown("GPSクラスタを確認し、手動の場所ラベルと公開範囲を管理します。exact GPSは既定で表示しません。")
            with gr.Row():
                place_from = gr.Textbox(label="date_from", placeholder="YYYY-MM-DD")
                place_to = gr.Textbox(label="date_to", placeholder="YYYY-MM-DD")
                place_min_points = gr.Number(label="min_points", value=3, precision=0)
                place_status = gr.Dropdown(label="status", choices=["all", "unreviewed", "accepted", "rejected", "merged"], value="unreviewed")
                place_privacy = gr.Dropdown(
                    label="cluster privacy",
                    choices=["all", "private", "public_label", "public_hidden", "exact_private", "approximate_private", "public_place_label"],
                    value="all",
                )
            with gr.Row():
                place_category = gr.Dropdown(
                    label="category",
                    choices=["all", "home", "school", "lab", "station", "cafe", "restaurant", "travel", "shop", "event_venue", "other"],
                    value="all",
                )
                place_search = gr.Textbox(label="search text")
                place_mode = gr.Dropdown(label="display mode", choices=["public", "private"], value="public")
                place_limit = gr.Number(label="limit", value=50, precision=0)
            place_refresh = gr.Button("Place clusters 読み込み", variant="primary")
            place_summary = gr.Textbox(label="summary", interactive=False)
            place_table = gr.Dataframe(
                headers=[
                    "cluster_id",
                    "points",
                    "photos",
                    "events",
                    "first_seen_at",
                    "last_seen_at",
                    "radius_m",
                    "linked_place",
                    "status",
                    "privacy_level",
                    "approximate_location",
                ],
                label="Place clusters",
                interactive=False,
            )
            selected_place_cluster = gr.Dropdown(label="cluster_id", choices=[], value=None)
            place_private_detail = gr.Checkbox(label="show exact private detail", value=False)
            place_load_detail = gr.Button("cluster detail", variant="secondary")
            place_detail_summary = gr.Textbox(label="cluster detail", lines=12, interactive=False)
            place_gallery = gr.Gallery(label="representative thumbnails", columns=6, height=300, elem_classes=["photo-grid"])
            place_photo_rows = gr.Dataframe(headers=["media_id", "captured_at", "file_name", "thumbnail_path"], label="related media", interactive=False)
            place_event_rows = gr.Dataframe(headers=["date", "time", "title", "location", "confidence"], label="related events", interactive=False)
            with gr.Row():
                place_id_box = gr.Textbox(label="place_id")
                place_name_box = gr.Textbox(label="display_name")
                place_public_name_box = gr.Textbox(label="public_name")
                place_category_box = gr.Dropdown(
                    label="place category",
                    choices=["home", "school", "lab", "station", "cafe", "restaurant", "travel", "shop", "event_venue", "other"],
                    value="other",
                )
                place_privacy_box = gr.Dropdown(label="place privacy", choices=["private", "public_label", "public_hidden"], value="private")
            with gr.Row():
                place_alias_box = gr.Textbox(label="aliases / alias")
                place_notes_box = gr.Textbox(label="notes")
            with gr.Row():
                place_create = gr.Button("Create place", variant="primary")
                place_update = gr.Button("Update place", variant="secondary")
                place_link = gr.Button("Link cluster", variant="secondary")
                place_add_alias = gr.Button("Add alias", variant="secondary")
            with gr.Row():
                place_accept = gr.Button("Mark accepted", variant="primary")
                place_reject = gr.Button("Reject cluster", variant="stop")
                place_unlink = gr.Button("Unlink cluster", variant="secondary")
                place_reassign = gr.Button("Reassign events/media", variant="secondary")
            place_action_status = gr.Textbox(label="action status", interactive=False)
            place_detail_outputs = [
                place_detail_summary,
                place_gallery,
                place_photo_rows,
                place_event_rows,
                place_id_box,
                place_name_box,
                place_public_name_box,
                place_category_box,
                place_privacy_box,
                place_alias_box,
                place_action_status,
            ]
            place_refresh.click(
                _load_place_review_queue,
                inputs=[place_from, place_to, place_min_points, place_status, place_privacy, place_category, place_search, place_mode, place_limit],
                outputs=[place_summary, place_table, selected_place_cluster, *place_detail_outputs],
            )
            place_load_detail.click(_place_detail_ui, inputs=[selected_place_cluster, place_private_detail], outputs=place_detail_outputs)
            selected_place_cluster.change(_place_detail_ui, inputs=[selected_place_cluster, place_private_detail], outputs=place_detail_outputs)
            place_create.click(
                _create_place_ui,
                inputs=[selected_place_cluster, place_name_box, place_public_name_box, place_category_box, place_privacy_box, place_alias_box, place_notes_box],
                outputs=place_action_status,
            )
            place_update.click(
                _update_place_ui,
                inputs=[place_id_box, place_name_box, place_public_name_box, place_category_box, place_privacy_box, place_notes_box],
                outputs=place_action_status,
            )
            place_link.click(_link_place_ui, inputs=[place_id_box, selected_place_cluster], outputs=place_action_status)
            place_add_alias.click(_alias_place_ui, inputs=[place_id_box, place_alias_box], outputs=place_action_status)
            place_accept.click(lambda cluster_id: _place_cluster_action_ui(cluster_id, "accepted"), inputs=selected_place_cluster, outputs=place_action_status)
            place_reject.click(lambda cluster_id: _place_cluster_action_ui(cluster_id, "rejected"), inputs=selected_place_cluster, outputs=place_action_status)
            place_unlink.click(lambda cluster_id: _place_cluster_action_ui(cluster_id, "unlink"), inputs=selected_place_cluster, outputs=place_action_status)
            place_reassign.click(_reassign_places_ui, inputs=[place_from, place_to], outputs=place_action_status)

        with gr.Tab("Face Review / 顔検出レビュー"):
            gr.Markdown("顔検出bboxをローカルで確認します。人物名付け、感情推定、関係性推定は行いません。")
            with gr.Row():
                face_from = gr.Textbox(label="date_from", placeholder="YYYY-MM-DD")
                face_to = gr.Textbox(label="date_to", placeholder="YYYY-MM-DD")
                face_status = gr.Dropdown(
                    label="status",
                    choices=["all", "success", "no_face_detected", "failed", "engine_unavailable", "skipped"],
                    value="success",
                )
                face_review_status = gr.Dropdown(
                    label="review_status",
                    choices=["all", "unreviewed", "accepted", "rejected", "bad_detection"],
                    value="unreviewed",
                )
                face_limit = gr.Number(label="limit", value=50, precision=0)
            face_refresh = gr.Button("Face review queue 読み込み", variant="primary")
            face_summary = gr.Textbox(label="summary", interactive=False)
            face_table = gr.Dataframe(
                headers=[
                    "face_id",
                    "media_id",
                    "captured_at",
                    "file_name",
                    "status",
                    "bbox",
                    "score",
                    "review_status",
                    "engine",
                ],
                label="Face detections",
                interactive=False,
            )
            face_gallery = gr.Gallery(
                label="Face detection thumbnails (click to select face_id)",
                columns=8,
                height=260,
                elem_classes=["photo-grid"],
            )
            face_ids_state = gr.State([])
            face_gallery_ids_state = gr.State([])
            with gr.Row():
                selected_face_id = gr.Dropdown(label="face_id", choices=[], value=None, scale=12, allow_custom_value=True)
                face_prev = gr.Button("↑", variant="secondary", min_width=48)
                face_next = gr.Button("↓", variant="secondary", min_width=48)
            face_show_private = gr.Checkbox(label="show private local crops / face thumbnails", value=True)
            face_load_detail = gr.Button("face detail", variant="secondary")
            face_detail_summary = gr.Textbox(label="detail", lines=10, interactive=False)
            with gr.Row():
                face_crop = gr.Image(label="face thumbnail", type="filepath", interactive=False, elem_classes=["thumbnail-focus"])
                face_original_thumb = gr.Image(label="original media thumbnail", type="filepath", interactive=False, elem_classes=["thumbnail-focus"])
            with gr.Row():
                face_file_name = gr.Textbox(label="file_name", interactive=False)
                face_crop_note = gr.Textbox(label="privacy note", interactive=False)
                face_review_box = gr.Textbox(label="review_status", interactive=False)
            with gr.Row():
                face_accept = gr.Button("Accept", variant="primary")
                face_bad = gr.Button("Bad detection", variant="stop")
                face_reject = gr.Button("Reject", variant="secondary")
                face_hide = gr.Button("Hide from face workflow", variant="secondary")
            face_action_status = gr.Textbox(label="action status", interactive=False)
            face_detail_outputs = [
                face_detail_summary,
                face_crop,
                face_original_thumb,
                face_file_name,
                face_crop_note,
                face_review_box,
                face_action_status,
            ]
            face_refresh.click(
                _load_face_review_queue,
                inputs=[face_from, face_to, face_status, face_review_status, face_limit, face_show_private],
                outputs=[face_summary, face_table, face_gallery, selected_face_id, face_ids_state, face_gallery_ids_state, *face_detail_outputs],
            )
            face_load_detail.click(_face_detail_ui, inputs=[selected_face_id, face_show_private], outputs=face_detail_outputs)
            selected_face_id.change(_face_detail_ui, inputs=[selected_face_id, face_show_private], outputs=face_detail_outputs)
            face_gallery.select(
                _face_gallery_select,
                inputs=[face_gallery_ids_state, face_show_private],
                outputs=[selected_face_id, *face_detail_outputs],
            )
            face_prev.click(_face_prev_ui, inputs=[selected_face_id, face_ids_state, face_show_private], outputs=[selected_face_id, *face_detail_outputs])
            face_next.click(_face_next_ui, inputs=[selected_face_id, face_ids_state, face_show_private], outputs=[selected_face_id, *face_detail_outputs])
            face_accept.click(
                _face_accept_and_next,
                inputs=[selected_face_id, face_ids_state, face_show_private],
                outputs=[selected_face_id, *face_detail_outputs],
            )
            face_bad.click(lambda face_id: _face_review_action(face_id, "bad_detection"), inputs=selected_face_id, outputs=face_detail_outputs)
            face_reject.click(lambda face_id: _face_review_action(face_id, "rejected"), inputs=selected_face_id, outputs=face_detail_outputs)
            face_hide.click(lambda face_id: _face_review_action(face_id, "rejected"), inputs=selected_face_id, outputs=face_detail_outputs)

            gr.Markdown("### Face clusters / 手動人物ラベル")
            gr.Markdown("人物名は手動で設定してください。アプリは顔から名前や関係性を自動推定しません。")
            with gr.Row():
                face_cluster_status = gr.Dropdown(
                    label="cluster status",
                    choices=["all", "unreviewed", "accepted", "rejected", "merged", "split"],
                    value="unreviewed",
                )
                face_cluster_limit = gr.Number(label="cluster limit", value=50, precision=0)
                face_public_mode = gr.Checkbox(label="public mode preview", value=False)
                face_cluster_refresh = gr.Button("Face clusters 読み込み", variant="primary")
            face_cluster_summary = gr.Textbox(label="cluster summary", interactive=False)
            face_cluster_table = gr.Dataframe(
                headers=[
                    "cluster_id",
                    "cluster_label",
                    "face_count",
                    "first_seen_at",
                    "last_seen_at",
                    "status",
                    "review_status",
                    "linked_person",
                    "note",
                ],
                label="Face clusters",
                interactive=False,
            )
            face_cluster_overview_gallery = gr.Gallery(
                label="Face cluster thumbnails (click to select cluster_id)",
                columns=6,
                height=320,
                elem_classes=["photo-grid"],
            )
            face_cluster_ids_state = gr.State([])
            face_cluster_gallery_ids_state = gr.State([])
            with gr.Row():
                selected_face_cluster_id = gr.Dropdown(label="cluster_id", choices=[], value=None, scale=12, allow_custom_value=True)
                face_cluster_prev = gr.Button("↑", variant="secondary", min_width=48)
                face_cluster_next = gr.Button("↓", variant="secondary", min_width=48)
            merge_target_cluster_id = gr.Dropdown(label="merge target cluster_id", choices=[], value=None, allow_custom_value=True)
            face_cluster_show_private = gr.Checkbox(label="show private cluster thumbnails", value=True)
            face_cluster_load = gr.Button("cluster detail", variant="secondary")
            face_cluster_detail_summary = gr.Textbox(label="cluster detail", lines=10, interactive=False)
            face_cluster_members = gr.Dataframe(
                headers=["face_id", "media_id", "captured_at", "file_name", "score", "review_status"],
                label="cluster members",
                interactive=False,
            )
            face_cluster_gallery = gr.Gallery(label="member thumbnails", columns=4, height=240)
            face_cluster_member_face_ids_state = gr.State([])
            with gr.Row():
                face_cluster_linked_people = gr.Textbox(label="linked person", interactive=False)
                face_cluster_privacy_note = gr.Textbox(label="cluster privacy note", interactive=False)
            gr.Markdown(
                "#### 1. 名前を付ける\n"
                "選択中の顔クラスターに表示名を入力して保存します。同じ表示名が既にある場合は、そのpersonを再利用して同一人物として扱います。"
            )
            with gr.Row():
                cluster_label_name = gr.Textbox(label="このクラスターの人物名")
                cluster_label_public_name = gr.Textbox(label="公開用名", placeholder="人物A")
                cluster_label_privacy = gr.Dropdown(
                    label="privacy_level",
                    choices=["private", "public_alias", "public_hidden"],
                    value="private",
                )
                face_cluster_label_button = gr.Button("この名前でクラスターにラベル付け", variant="primary")
            with gr.Row():
                face_cluster_accept = gr.Button("Accept cluster", variant="primary")
                face_cluster_reject = gr.Button("Reject cluster", variant="secondary")
                face_cluster_bad = gr.Button("Bad cluster", variant="stop")
            face_person_table = gr.Dataframe(
                headers=["person_id", "display_name", "public_name", "privacy_level", "linked_clusters", "aliases"],
                label="Manual persons",
                interactive=False,
            )
            person_ids_state = gr.State([])
            with gr.Row():
                selected_person_id = gr.Dropdown(label="person_id", choices=[], value=None, scale=12, allow_custom_value=True)
                face_person_prev = gr.Button("↑", variant="secondary", min_width=48)
                face_person_next = gr.Button("↓", variant="secondary", min_width=48)
            with gr.Row():
                person_name = gr.Textbox(label="display_name (manual only)")
                person_public_name = gr.Textbox(label="public_name")
                person_privacy = gr.Dropdown(
                    label="privacy_level",
                    choices=["private", "public_alias", "public_hidden"],
                    value="private",
            )
            with gr.Row():
                face_person_create = gr.Button("Create person", variant="primary")
                face_person_update = gr.Button("Update person", variant="secondary")
            with gr.Row():
                person_alias = gr.Textbox(label="alias")
                face_person_alias = gr.Button("Add alias", variant="secondary")
            with gr.Row():
                face_person_link = gr.Button("Link person to cluster", variant="primary")
                face_person_unlink = gr.Button("Unlink person from cluster", variant="secondary")
            face_person_action_status = gr.Textbox(label="person action status", interactive=False)
            gr.Markdown(
                "### Person cluster organizer / 名前ごとのクラスター整理\n"
                "ここでは、person_idに紐づいた顔クラスターを一覧できます。"
                "未ラベルのクラスターは既存personへ統合し、誤っているクラスターはpersonから外せます。"
                "クラスター同士の統合は、顔メンバーを物理的に混ぜず、同じperson_idへリンクして扱います。"
            )
            with gr.Row():
                organizer_refresh = gr.Button("Person cluster overview 読み込み", variant="secondary")
                organizer_link = gr.Button("選択clusterを選択personへ統合", variant="primary")
                organizer_merge = gr.Button("選択clusterをtarget clusterのpersonへ統合", variant="secondary")
                organizer_unlink = gr.Button("選択clusterを選択personから削除", variant="stop")
            organizer_summary = gr.Textbox(label="person cluster summary", interactive=False)
            organizer_table = gr.Dataframe(
                headers=[
                    "person_id",
                    "display_name",
                    "public_name",
                    "privacy_level",
                    "cluster_id",
                    "cluster_label",
                    "face_count",
                    "first_seen_at",
                    "last_seen_at",
                    "cluster_status",
                    "review_status",
                    "source",
                ],
                label="Person linked face clusters",
                interactive=False,
            )
            organizer_gallery = gr.Gallery(
                label="Person linked cluster thumbnails (click to select person_id and cluster_id)",
                columns=6,
                height=320,
                elem_classes=["photo-grid"],
            )
            organizer_gallery_cluster_ids_state = gr.State([])
            organizer_gallery_person_ids_state = gr.State([])
            face_cluster_detail_outputs = [
                face_cluster_detail_summary,
                face_cluster_members,
                face_cluster_gallery,
                face_cluster_member_face_ids_state,
                face_cluster_linked_people,
                face_cluster_privacy_note,
                face_person_action_status,
            ]
            organizer_action_outputs = [
                face_person_action_status,
                organizer_summary,
                organizer_table,
                organizer_gallery,
                organizer_gallery_cluster_ids_state,
                organizer_gallery_person_ids_state,
                face_person_table,
                selected_person_id,
                person_ids_state,
            ]
            face_cluster_refresh.click(
                _load_face_clusters,
                inputs=[face_cluster_status, face_cluster_limit, face_public_mode, face_cluster_show_private],
                outputs=[
                    face_cluster_summary,
                    face_cluster_table,
                    face_cluster_overview_gallery,
                    face_cluster_gallery_ids_state,
                    selected_face_cluster_id,
                    face_cluster_ids_state,
                    merge_target_cluster_id,
                    face_person_table,
                    selected_person_id,
                    person_ids_state,
                    *face_cluster_detail_outputs,
                ],
            )
            face_cluster_load.click(
                _face_cluster_detail_ui,
                inputs=[selected_face_cluster_id, face_cluster_show_private, face_public_mode],
                outputs=face_cluster_detail_outputs,
            )
            selected_face_cluster_id.change(
                _face_cluster_detail_ui,
                inputs=[selected_face_cluster_id, face_cluster_show_private, face_public_mode],
                outputs=face_cluster_detail_outputs,
            )
            face_cluster_overview_gallery.select(
                _face_cluster_gallery_select,
                inputs=[face_cluster_gallery_ids_state, face_cluster_show_private, face_public_mode],
                outputs=[selected_face_cluster_id, *face_cluster_detail_outputs],
            )
            face_cluster_gallery.select(
                _face_member_gallery_select,
                inputs=[face_cluster_member_face_ids_state, face_show_private],
                outputs=[selected_face_id, *face_detail_outputs],
            )
            face_cluster_prev.click(
                _face_cluster_prev_ui,
                inputs=[selected_face_cluster_id, face_cluster_ids_state, face_cluster_show_private, face_public_mode],
                outputs=[selected_face_cluster_id, *face_cluster_detail_outputs],
            )
            face_cluster_next.click(
                _face_cluster_next_ui,
                inputs=[selected_face_cluster_id, face_cluster_ids_state, face_cluster_show_private, face_public_mode],
                outputs=[selected_face_cluster_id, *face_cluster_detail_outputs],
            )
            face_cluster_accept.click(
                _face_cluster_accept_and_next,
                inputs=[selected_face_cluster_id, face_cluster_ids_state, face_cluster_show_private, face_public_mode],
                outputs=[selected_face_cluster_id, *face_cluster_detail_outputs],
            )
            face_cluster_reject.click(lambda cluster_id: _face_cluster_action(cluster_id, "rejected"), inputs=selected_face_cluster_id, outputs=face_cluster_detail_outputs)
            face_cluster_bad.click(lambda cluster_id: _face_cluster_action(cluster_id, "bad_cluster"), inputs=selected_face_cluster_id, outputs=face_cluster_detail_outputs)
            face_person_create.click(
                _create_face_person,
                inputs=[person_name, person_public_name, person_privacy],
                outputs=[face_person_action_status, face_person_table, selected_person_id, person_ids_state],
            )
            face_person_update.click(
                _update_face_person,
                inputs=[selected_person_id, person_name, person_public_name, person_privacy],
                outputs=[face_person_action_status, face_person_table, selected_person_id, person_ids_state],
            )
            face_cluster_label_button.click(
                _label_selected_face_cluster,
                inputs=[
                    selected_face_cluster_id,
                    cluster_label_name,
                    cluster_label_public_name,
                    cluster_label_privacy,
                ],
                outputs=[
                    face_person_action_status,
                    face_person_table,
                    selected_person_id,
                    person_ids_state,
                    *face_cluster_detail_outputs,
                ],
            )
            selected_person_id.change(
                _person_form_values,
                inputs=selected_person_id,
                outputs=[person_name, person_public_name, person_privacy],
            )
            face_person_prev.click(
                _person_prev_ui,
                inputs=[selected_person_id, person_ids_state],
                outputs=[selected_person_id, person_name, person_public_name, person_privacy],
            )
            face_person_next.click(
                _person_next_ui,
                inputs=[selected_person_id, person_ids_state],
                outputs=[selected_person_id, person_name, person_public_name, person_privacy],
            )
            organizer_refresh.click(
                _person_cluster_overview_values,
                inputs=[face_public_mode, face_cluster_show_private],
                outputs=[
                    organizer_summary,
                    organizer_table,
                    organizer_gallery,
                    organizer_gallery_cluster_ids_state,
                    organizer_gallery_person_ids_state,
                ],
            )
            organizer_gallery.select(
                _organizer_gallery_select,
                inputs=[
                    organizer_gallery_cluster_ids_state,
                    organizer_gallery_person_ids_state,
                    face_cluster_show_private,
                    face_public_mode,
                ],
                outputs=[
                    selected_face_cluster_id,
                    selected_person_id,
                    person_name,
                    person_public_name,
                    person_privacy,
                    *face_cluster_detail_outputs,
                ],
            )
            organizer_link.click(
                _organizer_link_cluster_to_person,
                inputs=[selected_person_id, selected_face_cluster_id, face_public_mode, face_cluster_show_private],
                outputs=organizer_action_outputs,
            )
            organizer_merge.click(
                _organizer_merge_cluster_into_target,
                inputs=[selected_face_cluster_id, merge_target_cluster_id, face_public_mode, face_cluster_show_private],
                outputs=organizer_action_outputs,
            )
            organizer_unlink.click(
                _organizer_unlink_cluster_from_person,
                inputs=[selected_person_id, selected_face_cluster_id, face_public_mode, face_cluster_show_private],
                outputs=organizer_action_outputs,
            )
            face_person_alias.click(_add_face_person_alias, inputs=[selected_person_id, person_alias], outputs=face_person_action_status)
            face_person_link.click(_link_face_person, inputs=[selected_person_id, selected_face_cluster_id], outputs=face_person_action_status)
            face_person_unlink.click(_unlink_face_person, inputs=[selected_person_id, selected_face_cluster_id], outputs=face_person_action_status)

            gr.Markdown("### 2. LINE speaker link / 手動LINE話者リンク")
            gr.Markdown("LINE話者と人物の対応は手動で設定してください。アプリは顔や会話内容から人物関係を自動推定しません。")
            with gr.Row():
                line_speaker_limit = gr.Number(label="LINE speaker limit", value=100, precision=0)
                line_speaker_refresh = gr.Button("LINE speakers 読み込み", variant="secondary")
            line_speaker_summary = gr.Textbox(label="LINE speaker summary", interactive=False)
            line_speaker_table = gr.Dataframe(
                headers=[
                    "chat_id",
                    "speaker_name",
                    "message_count",
                    "first_seen_at",
                    "last_seen_at",
                    "linked_person",
                    "linked_public_name",
                ],
                label="LINE speakers",
                interactive=False,
            )
            with gr.Row():
                line_chat_id = gr.Textbox(label="chat_id")
                line_speaker_name = gr.Textbox(label="speaker_name")
                line_add_alias = gr.Checkbox(label="also add speaker name as person alias", value=True)
            with gr.Row():
                line_speaker_link = gr.Button("Link LINE speaker to selected person", variant="primary")
                line_speaker_unlink = gr.Button("Unlink LINE speaker", variant="secondary")
            line_speaker_refresh.click(
                _load_line_speakers_ui,
                inputs=line_speaker_limit,
                outputs=[line_speaker_summary, line_speaker_table],
            )
            line_speaker_link.click(
                _link_line_speaker_ui,
                inputs=[line_chat_id, line_speaker_name, selected_person_id, line_add_alias],
                outputs=face_person_action_status,
            )
            line_speaker_unlink.click(
                _unlink_line_speaker_ui,
                inputs=[line_chat_id, line_speaker_name, selected_person_id],
                outputs=face_person_action_status,
            )

        with gr.Tab("VLM Review / 画像解析レビュー"):
            gr.Markdown("画像解析による推定を確認し、検索・イベント生成に使うかを管理します。")
            with gr.Row():
                vlmr_month = gr.Textbox(label="month", placeholder="YYYY-MM")
                vlmr_date = gr.Textbox(label="date", placeholder="YYYY-MM-DD")
                vlmr_from = gr.Textbox(label="date_from", placeholder="YYYY-MM-DD")
                vlmr_to = gr.Textbox(label="date_to", placeholder="YYYY-MM-DD")
                vlmr_status = gr.Dropdown(
                    label="review_status",
                    choices=["all", "unreviewed", "accepted", "rejected", "needs_fix", "wrong"],
                    value="all",
                )
            with gr.Row():
                vlmr_unreviewed = gr.Checkbox(label="unreviewed only", value=False)
                vlmr_safety = gr.Checkbox(label="safety_flagsあり", value=False)
                vlmr_food = gr.Checkbox(label="food_cuesあり", value=False)
                vlmr_performance = gr.Checkbox(label="performance/stageあり", value=False)
            with gr.Row():
                vlmr_has_ocr = gr.Checkbox(label="OCRあり", value=False)
                vlmr_has_embedding = gr.Checkbox(label="embeddingあり", value=False)
                vlmr_event_linked = gr.Checkbox(label="event linked", value=False)
                vlmr_failed = gr.Checkbox(label="failed", value=False)
                vlmr_low_conf = gr.Number(label="confidence <=", value=None)
                vlmr_limit = gr.Number(label="limit", value=100, precision=0)
            load_vlm_review_button = gr.Button("VLM Review Queue 読み込み", variant="primary")
            vlmr_summary = gr.Textbox(label="Review summary", interactive=False)
            vlmr_table = gr.Dataframe(
                headers=[
                    "media_id",
                    "captured_at",
                    "file_name",
                    "thumbnail_path",
                    "short_caption",
                    "confidence",
                    "safety_flags",
                    "scene_tags",
                    "activity_tags",
                    "food_cues",
                    "location_cues",
                    "review_status",
                    "verified",
                    "hidden",
                    "wrong",
                    "searchable",
                    "event_usable",
                    "status",
                ],
                label="VLM review queue",
                interactive=False,
            )
            selected_vlm_media_id = gr.Dropdown(label="media_id", choices=[], value=None)
            with gr.Row():
                vlmr_caption = gr.Textbox(label="caption_override", lines=3)
                vlmr_short_caption = gr.Textbox(label="short_caption_override", lines=2)
            with gr.Row():
                vlmr_scene_tags = gr.Textbox(label="scene_tags", placeholder="indoor, restaurant")
                vlmr_object_tags = gr.Textbox(label="object_tags")
                vlmr_activity_tags = gr.Textbox(label="activity_tags")
            with gr.Row():
                vlmr_food_cues = gr.Textbox(label="food_cues")
                vlmr_location_cues = gr.Textbox(label="location_cues")
            with gr.Row():
                vlmr_status_edit = gr.Dropdown(
                    label="review_status",
                    choices=["unreviewed", "accepted", "rejected", "needs_fix", "wrong"],
                    value="unreviewed",
                )
                vlmr_verified = gr.Checkbox(label="verified")
                vlmr_hidden = gr.Checkbox(label="hidden")
                vlmr_wrong = gr.Checkbox(label="wrong")
                vlmr_searchable = gr.Checkbox(label="use in search", value=True)
                vlmr_event_usable = gr.Checkbox(label="use in events", value=True)
            vlmr_note = gr.Textbox(label="review_note", lines=2)
            with gr.Row():
                save_vlm_review_button = gr.Button("Save override", variant="primary")
                accept_vlm_button = gr.Button("Accept", variant="primary")
                reject_vlm_button = gr.Button("Reject", variant="secondary")
                wrong_vlm_button = gr.Button("Mark wrong", variant="stop")
                verified_vlm_button = gr.Button("Mark verified", variant="secondary")
                hide_vlm_button = gr.Button("Hide", variant="secondary")
                search_off_vlm_button = gr.Button("Search OFF", variant="secondary")
                event_off_vlm_button = gr.Button("Events OFF", variant="secondary")
                clear_vlm_button = gr.Button("Clear override", variant="secondary")
            vlmr_status_box = gr.Textbox(label="Save status", interactive=False)
            vlmr_ocr = gr.Textbox(label="OCR preview", lines=3, interactive=False)
            vlmr_related_events = gr.Textbox(label="Related events", lines=3, interactive=False)
            gr.Markdown("## Bulk Actions")
            vlmr_bulk_ids = gr.Textbox(label="media_idを複数行で入力", lines=4)
            with gr.Row():
                vlmr_bulk_action = gr.Dropdown(
                    label="bulk action",
                    choices=["accepted", "rejected", "wrong", "hidden", "not_searchable", "not_event_usable", "tag"],
                    value="accepted",
                )
                vlmr_bulk_tag = gr.Textbox(label="bulk tag")
                vlmr_bulk_button = gr.Button("Bulk update", variant="secondary")
            vlmr_bulk_status = gr.Textbox(label="Bulk status", interactive=False)
            vlmr_eval_query = gr.Textbox(label="eval query", value="ご飯を食べた写真")
            vlmr_eval_button = gr.Button("Generate VLM eval case", variant="secondary")
            vlmr_eval_yaml = gr.Textbox(label="eval case YAML", lines=8, interactive=False)

            vlm_detail_outputs = [
                vlmr_caption,
                vlmr_short_caption,
                vlmr_scene_tags,
                vlmr_object_tags,
                vlmr_activity_tags,
                vlmr_food_cues,
                vlmr_location_cues,
                vlmr_status_edit,
                vlmr_note,
                vlmr_verified,
                vlmr_hidden,
                vlmr_wrong,
                vlmr_searchable,
                vlmr_event_usable,
                vlmr_ocr,
                vlmr_related_events,
                vlmr_status_box,
            ]
            load_vlm_review_button.click(
                _load_vlm_review_queue,
                inputs=[
                    vlmr_month,
                    vlmr_date,
                    vlmr_from,
                    vlmr_to,
                    vlmr_status,
                    vlmr_unreviewed,
                    vlmr_safety,
                    vlmr_food,
                    vlmr_performance,
                    vlmr_has_ocr,
                    vlmr_has_embedding,
                    vlmr_event_linked,
                    vlmr_failed,
                    vlmr_low_conf,
                    vlmr_limit,
                ],
                outputs=[vlmr_summary, vlmr_table, selected_vlm_media_id, *vlm_detail_outputs],
            )
            selected_vlm_media_id.change(_vlm_detail_values, inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            save_vlm_review_button.click(
                _save_vlm_review,
                inputs=[
                    selected_vlm_media_id,
                    vlmr_caption,
                    vlmr_short_caption,
                    vlmr_scene_tags,
                    vlmr_object_tags,
                    vlmr_activity_tags,
                    vlmr_food_cues,
                    vlmr_location_cues,
                    vlmr_status_edit,
                    vlmr_note,
                    vlmr_verified,
                    vlmr_hidden,
                    vlmr_wrong,
                    vlmr_searchable,
                    vlmr_event_usable,
                ],
                outputs=vlm_detail_outputs,
            )
            accept_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "accepted"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            reject_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "rejected"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            wrong_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "wrong"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            verified_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "verified"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            hide_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "hidden"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            search_off_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "search_off"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            event_off_vlm_button.click(lambda media_id: _quick_vlm_action(media_id, "event_off"), inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            clear_vlm_button.click(_clear_vlm_review, inputs=selected_vlm_media_id, outputs=vlm_detail_outputs)
            vlmr_bulk_button.click(_bulk_vlm_update, inputs=[vlmr_bulk_ids, vlmr_bulk_action, vlmr_bulk_tag], outputs=vlmr_bulk_status)
            vlmr_eval_button.click(_make_vlm_eval_case, inputs=[selected_vlm_media_id, vlmr_eval_query], outputs=vlmr_eval_yaml)

        with gr.Tab("Events / Timeline"):
            event_date = gr.Textbox(label="Date", value="2024-12-24", placeholder="YYYY-MM-DD")
            load_events_button = gr.Button("イベント読み込み", variant="primary")
            event_summary = gr.Textbox(label="Summary", interactive=False)
            events_table = gr.Dataframe(
                headers=[
                    "event_id",
                    "time range",
                    "title",
                    "summary",
                    "confidence",
                    "location_name",
                    "evidence",
                    "line",
                    "photo",
                    "verified",
                    "pinned",
                    "hidden",
                    "overridden",
                ],
                label="イベント一覧",
                interactive=False,
            )
            gr.Markdown("## Review Queue / 確認待ち")
            with gr.Row():
                rq_date = gr.Textbox(label="date", placeholder="YYYY-MM-DD")
                rq_from = gr.Textbox(label="date_from", placeholder="YYYY-MM-DD")
                rq_to = gr.Textbox(label="date_to", placeholder="YYYY-MM-DD")
                rq_confidence = gr.Number(label="confidence <=", value=None)
            with gr.Row():
                rq_title = gr.Textbox(label="title contains")
                rq_location = gr.Textbox(label="location_name contains")
                rq_modality = gr.Dropdown(
                    label="modality",
                    choices=["all", "line_only", "photo_only", "photo_and_line", "no_evidence"],
                    value="all",
                )
                rq_category = gr.Dropdown(
                    label="title category",
                    choices=[
                        "all",
                        "LINEのやりとり",
                        "通話・連絡",
                        "食事・カフェの可能性",
                        "移動・待ち合わせの可能性",
                        "位置情報付き写真の記録",
                        "写真撮影の記録",
                    ],
                    value="all",
                )
            with gr.Row():
                rq_verified = gr.Dropdown(label="verified", choices=["all", "verified only", "unverified only"], value="all")
                rq_hidden = gr.Dropdown(label="hidden", choices=["exclude hidden", "include hidden", "hidden only"], value="exclude hidden")
                rq_pinned = gr.Dropdown(label="pinned", choices=["all", "pinned only"], value="all")
                rq_evidence_min = gr.Number(label="evidence min", value=None)
                rq_evidence_max = gr.Number(label="evidence max", value=None)
            load_review_button = gr.Button("Review Queue 読み込み", variant="primary")
            review_summary = gr.Textbox(label="Review summary", interactive=False)
            review_table = gr.Dataframe(
                headers=[
                    "event_id",
                    "date",
                    "time range",
                    "title",
                    "location_name",
                    "confidence",
                    "modality",
                    "evidence",
                    "line",
                    "photo",
                    "verified",
                    "hidden",
                    "pinned",
                    "tags",
                ],
                label="Review Queue",
                interactive=False,
            )
            selected_event_id = gr.Dropdown(label="Event ID", choices=[], value=None)
            with gr.Row():
                title_input = gr.Textbox(label="title")
                location_input = gr.Textbox(label="location_name")
                location_choice = gr.Dropdown(
                    label="場所辞書候補",
                    choices=_place_choices(),
                    value=None,
                    interactive=True,
                )
            summary_input = gr.Textbox(label="summary", lines=4)
            with gr.Row():
                detail_date = gr.Textbox(label="date", interactive=False)
                detail_start = gr.Textbox(label="start_time", interactive=False)
                detail_end = gr.Textbox(label="end_time", interactive=False)
                detail_confidence = gr.Textbox(label="confidence", interactive=False)
            participants_box = gr.Textbox(label="participants_json", interactive=False)
            tags_input = gr.Textbox(label="tags", placeholder="旅行, 食事")
            with gr.Row():
                verified_checkbox = gr.Checkbox(label="このイベントを手動確認済みにする")
                pinned_checkbox = gr.Checkbox(label="このイベントを回答で優先表示する")
                hidden_checkbox = gr.Checkbox(label="このイベントを非表示にする")
            with gr.Row():
                save_event_button = gr.Button("保存", variant="primary")
                quick_verified_button = gr.Button("Mark verified", variant="secondary")
                toggle_pinned_button = gr.Button("Toggle pinned", variant="secondary")
                toggle_hidden_button = gr.Button("Toggle hidden", variant="secondary")
                clear_override_button = gr.Button("Clear overrides", variant="secondary")
                reload_event_button = gr.Button("Reload event", variant="secondary")
            save_status = gr.Textbox(label="Save status", interactive=False)
            with gr.Row():
                quick_tag_input = gr.Textbox(label="Add tag")
                quick_tag_button = gr.Button("Add tag", variant="secondary")
                copied_event_id = gr.Textbox(label="Copy event id", interactive=False)
                copy_event_id_button = gr.Button("Copy event id", variant="secondary")
            evidence_summary = gr.Textbox(label="Evidence summary", interactive=False)
            line_event_evidence = gr.Dataframe(
                headers=["sent_at", "sender", "type", "text"],
                label="LINE evidence 最大10件",
                interactive=False,
            )
            photo_event_evidence = gr.Dataframe(
                headers=["captured_at", "file_name", "gps", "location_name", "thumbnail_path"],
                label="写真 evidence 最大20件",
                interactive=False,
            )
            ocr_event_evidence = gr.Dataframe(
                headers=["media_id", "captured_at", "file_name", "status", "engine", "text"],
                label="OCR evidence",
                interactive=False,
            )
            vlm_event_evidence = gr.Dataframe(
                headers=[
                    "media_id",
                    "captured_at",
                    "file_name",
                    "status",
                    "engine",
                    "caption",
                    "scene_tags",
                    "object_tags",
                    "activity_tags",
                    "food_cues",
                    "location_cues",
                    "safety_flags",
                ],
                label="VLM evidence",
                interactive=False,
            )
            photo_gallery = gr.Gallery(
                label="写真サムネイル",
                columns=6,
                height=430,
                elem_classes=["photo-grid"],
            )
            gr.Markdown("## Bulk Actions")
            bulk_ids = gr.Textbox(label="event_idを複数行で入力", lines=4)
            with gr.Row():
                bulk_action = gr.Dropdown(label="bulk action", choices=["verified", "hidden", "unhide", "tag"], value="verified")
                bulk_tag = gr.Textbox(label="bulk tag")
                bulk_button = gr.Button("Bulk update", variant="secondary")
            bulk_status = gr.Textbox(label="Bulk status", interactive=False)
            with gr.Row():
                bulk_low_conf_threshold = gr.Number(label="low-confidence threshold", value=0.5)
                bulk_low_conf_button = gr.Button("低confidence LINE-onlyをhidden", variant="secondary")
            eval_query = gr.Textbox(label="eval question")
            eval_expected_date = gr.Textbox(label="expected date")
            eval_button = gr.Button("eval case YAMLを生成", variant="secondary")
            eval_yaml = gr.Textbox(label="eval case YAML", lines=10, interactive=False)

            event_detail_outputs = [
                title_input,
                summary_input,
                detail_date,
                detail_start,
                detail_end,
                location_input,
                detail_confidence,
                participants_box,
                tags_input,
                verified_checkbox,
                pinned_checkbox,
                hidden_checkbox,
                evidence_summary,
                line_event_evidence,
                photo_event_evidence,
                ocr_event_evidence,
                vlm_event_evidence,
                photo_gallery,
                save_status,
            ]
            load_events_button.click(
                _load_events,
                inputs=event_date,
                outputs=[event_summary, events_table, selected_event_id, *event_detail_outputs],
            )
            load_review_button.click(
                _load_review_queue,
                inputs=[
                    rq_date,
                    rq_from,
                    rq_to,
                    rq_confidence,
                    rq_title,
                    rq_location,
                    rq_modality,
                    rq_verified,
                    rq_hidden,
                    rq_pinned,
                    rq_evidence_min,
                    rq_evidence_max,
                    rq_category,
                ],
                outputs=[review_summary, review_table, selected_event_id, *event_detail_outputs],
            )
            selected_event_id.change(
                _event_detail_values,
                inputs=selected_event_id,
                outputs=event_detail_outputs,
            )
            save_event_button.click(
                _save_event,
                inputs=[
                    selected_event_id,
                    title_input,
                    summary_input,
                    location_input,
                    location_choice,
                    tags_input,
                    verified_checkbox,
                    pinned_checkbox,
                    hidden_checkbox,
                ],
                outputs=event_detail_outputs,
            )
            quick_verified_button.click(_quick_verified, inputs=selected_event_id, outputs=event_detail_outputs)
            toggle_pinned_button.click(_toggle_pinned, inputs=selected_event_id, outputs=event_detail_outputs)
            toggle_hidden_button.click(_toggle_hidden, inputs=selected_event_id, outputs=event_detail_outputs)
            clear_override_button.click(_clear_overrides, inputs=selected_event_id, outputs=event_detail_outputs)
            reload_event_button.click(_event_detail_values, inputs=selected_event_id, outputs=event_detail_outputs)
            quick_tag_button.click(_add_tag, inputs=[selected_event_id, quick_tag_input], outputs=event_detail_outputs)
            copy_event_id_button.click(_copy_event_id, inputs=selected_event_id, outputs=copied_event_id)
            bulk_button.click(_bulk_update, inputs=[bulk_ids, bulk_action, bulk_tag], outputs=bulk_status)
            bulk_low_conf_button.click(
                _bulk_hide_low_line_only,
                inputs=[rq_from, rq_to, bulk_low_conf_threshold],
                outputs=bulk_status,
            )
            eval_button.click(
                _make_eval_case,
                inputs=[selected_event_id, eval_query, eval_expected_date],
                outputs=eval_yaml,
            )

        app.load(
            _stats_values,
            outputs=[db_path_box, media_count_box, line_count_box, event_count_box],
        )

    return app


def launch(
    db_path: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    app = create_app(db_path=db_path)
    app.launch(
        server_name=ensure_localhost(host),
        server_port=port,
        share=False,
        show_error=True,
    )


def _line_export_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*.txt") if item.is_file())
    return []


def _line_row(message: dict[str, Any]) -> list[str]:
    return [
        str(message.get("sent_at") or ""),
        str(message.get("sender") or message.get("sender_name") or ""),
        str(message.get("message_type") or ""),
        redact_text(str(message.get("text") or message.get("message_text") or ""), max_chars=120),
    ]


def _photo_row(item: dict[str, Any]) -> list[str]:
    has_gps = item.get("gps_lat") is not None and item.get("gps_lon") is not None
    return [
        str(item.get("captured_at") or item.get("fallback_captured_at") or ""),
        str(item.get("file_name") or ""),
        str(item.get("thumbnail_path") or ""),
        "yes" if has_gps else "no",
    ]


def _empty_event_detail_values():
    return (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        False,
        False,
        False,
        "",
        [],
        [],
        [],
        [],
        [],
        "",
    )


def _empty_vlm_detail_values():
    return (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "unreviewed",
        "",
        False,
        False,
        False,
        True,
        True,
        "",
        "",
        "",
    )


def _related_events_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return "\n".join(
        f"{row.get('event_id')}: {row.get('start_time') or ''}-{row.get('end_time') or ''} {row.get('title') or ''}"
        for row in rows[:10]
    )


def _tags_text(value: Any) -> str:
    if not value:
        return ""
    try:
        import json

        parsed = json.loads(str(value))
    except Exception:
        return str(value)
    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed)
    return str(value)


def _blank_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _compose_person_place_query(query_text: str, person_filter: str, place_filter: str, activity_query: str) -> str:
    parts: list[str] = []
    person = _blank_or_none(person_filter)
    place = _blank_or_none(place_filter)
    activity = _blank_or_none(activity_query)
    query = _blank_or_none(query_text)
    if person and place and activity:
        parts.append(f"{person}と{place}で{activity}")
    elif person and activity:
        parts.append(f"{person}と{activity}")
    elif place and activity:
        parts.append(f"{place}で{activity}")
    elif person:
        parts.append(person)
    elif place:
        parts.append(place)
    if activity and activity not in parts:
        parts.append(activity)
    if query:
        parts.append(query)
    seen: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    return " ".join(seen) or str(query_text or "")


def _month_or_dates(month_text: str, from_text: str, to_text: str) -> tuple[str | None, str | None]:
    month = _blank_or_none(month_text)
    if month and len(month) == 7 and month[4] == "-":
        year_text, month_value = month.split("-", 1)
        first = date(int(year_text), int(month_value), 1)
        if first.month == 12:
            next_month = date(first.year + 1, 1, 1)
        else:
            next_month = date(first.year, first.month + 1, 1)
        return first.isoformat(), (next_month - timedelta(days=1)).isoformat()
    return _blank_or_none(from_text), _blank_or_none(to_text)


def _contains_any(row: dict[str, Any], needles: tuple[str, ...]) -> bool:
    haystack = " ".join(str(value or "") for value in row.values()).lower()
    return any(needle.lower() in haystack for needle in needles)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _verified_filter(value: str) -> str:
    if value == "verified only":
        return "verified"
    if value == "unverified only":
        return "unverified"
    return "all"


def _hidden_filter(value: str) -> str:
    if value == "hidden only":
        return "only"
    if value == "include hidden":
        return "include"
    return "exclude"
