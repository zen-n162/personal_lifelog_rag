"""Command line interface for personal_lifelog_rag."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import sys
from pathlib import Path
from typing import Sequence

from personal_lifelog_rag.captioning.image_analysis import analyze_images, format_analysis_report
from personal_lifelog_rag.captioning.local_vlm import get_vlm_adapter
from personal_lifelog_rag.core.config import load_event_building_config
from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.backup import DEFAULT_BACKUP_DIR, backup_sqlite_db
from personal_lifelog_rag.db.checks import format_db_check, run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository, resolve_db_path
from personal_lifelog_rag.embeddings.adapter import get_embedding_adapter
from personal_lifelog_rag.embeddings.vector_search import (
    build_embeddings,
)
from personal_lifelog_rag.evaluation.private_eval import (
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RUNS_DIR,
    compare_private_eval_reports,
    evaluate_private_questions,
    format_private_eval_comparison,
    format_private_eval_report,
    load_private_eval_questions,
    write_private_eval_report,
    write_private_eval_template,
)
from personal_lifelog_rag.evaluation.search_snapshot import (
    DEFAULT_SEARCH_SNAPSHOT_DIR,
    DEFAULT_SEARCH_SNAPSHOT_QUERIES,
    SearchSnapshotOptions,
    build_search_snapshot,
    format_search_snapshot,
    write_search_snapshot,
)
from personal_lifelog_rag.ingest.line_parser import parse_line_chat_file_with_warnings
from personal_lifelog_rag.ingest.photo_ingest import ingest_photo_directory_with_report
from personal_lifelog_rag.line.call_index import (
    build_call_index,
    call_stats,
    format_build_call_index_report,
    format_call_stats,
    format_search_calls_report,
    search_calls,
)
from personal_lifelog_rag.ocr.local_ocr import get_ocr_adapter
from personal_lifelog_rag.ocr.engines import get_ocr_engine
from personal_lifelog_rag.ocr.ocr_service import (
    OcrImagesOptions,
    format_ocr_report,
    format_ocr_show,
    format_ocr_stats,
    ocr_stats,
    run_ocr_images,
)
from personal_lifelog_rag.places.assignment import (
    assign_places_to_events,
    format_assign_places_report,
)
from personal_lifelog_rag.places.clusterer import (
    cluster_place_candidates,
    format_place_clusters,
    write_place_cluster_suggestions,
)
from personal_lifelog_rag.places.geo import privacy_safe_lat_lon
from personal_lifelog_rag.places.matcher import match_place
from personal_lifelog_rag.places.place_dictionary import (
    DEFAULT_PRIVATE_PLACES_PATH,
    PlaceConfigError,
    load_place_dictionary,
    validate_place_dictionary,
)
from personal_lifelog_rag.places.redaction import (
    format_place_display_preview,
    place_display_preview,
)
from personal_lifelog_rag.places.stats import format_place_stats, place_stats
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.date_inspector import format_date_inspection, inspect_date
from personal_lifelog_rag.retrieval.local_search import (
    LocalSearchOptions,
    format_local_search_report,
    local_text_search,
)
from personal_lifelog_rag.retrieval.query_intent import classify_query_intent
from personal_lifelog_rag.retrieval.query_router import (
    format_query_intent,
    format_routed_query_result,
    route_query,
)
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.timeline.event_builder import EventBuildConfig, build_all_events, build_events
from personal_lifelog_rag.timeline.event_reports import (
    event_stats,
    format_event_list,
    format_event_stats,
    list_events_report,
)
from personal_lifelog_rag.ui.event_review import save_event_review_override
from personal_lifelog_rag.ui.event_review_service import (
    ReviewQueueFilters,
    bulk_update_events,
    format_review_queue,
    make_eval_case_yaml,
    review_queue,
)
from personal_lifelog_rag.vlm.engines import get_vlm_engine
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import (
    VlmImagesOptions,
    format_image_search,
    format_vlm_report,
    format_vlm_show,
    format_vlm_stats,
    image_search,
    run_vlm_images,
    vlm_stats,
)


def build_parser() -> argparse.ArgumentParser:
    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument(
        "--db-path",
        type=Path,
        default=argparse.SUPPRESS,
        help="SQLite database path. Defaults to data/db/lifelog.sqlite.",
    )

    parser = argparse.ArgumentParser(
        prog="personal-lifelog-rag",
        description="Local-first personal lifelog RAG utilities.",
        parents=[db_parent],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "init-db",
        parents=[db_parent],
        help="Create or migrate the local SQLite schema.",
    )
    subparsers.add_parser(
        "stats",
        parents=[db_parent],
        help="Show row counts for lifelog tables.",
    )
    backup_db_parser = subparsers.add_parser(
        "backup-db",
        parents=[db_parent],
        help="Copy the local SQLite database to backups/ before risky operations.",
    )
    backup_db_parser.add_argument("--label", default=None)
    backup_db_parser.add_argument("--output-dir", type=Path, default=DEFAULT_BACKUP_DIR)

    ingest_line_parser = subparsers.add_parser(
        "ingest-line",
        parents=[db_parent],
        help="Parse a local LINE export txt and store messages.",
    )
    ingest_line_parser.add_argument("legacy_path", nargs="?", type=Path)
    ingest_line_parser.add_argument("--path", type=Path, default=None)
    ingest_line_parser.add_argument("--chat-name", default=None)

    ingest_photos_parser = subparsers.add_parser(
        "ingest-photos",
        parents=[db_parent],
        help="Scan a local photo/video directory and store file metadata.",
    )
    ingest_photos_parser.add_argument("legacy_path", nargs="?", type=Path)
    ingest_photos_parser.add_argument("--path", type=Path, default=None)

    ask_parser = subparsers.add_parser(
        "ask",
        parents=[db_parent],
        help="Answer a simple timeline question with local database records.",
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument("--include-hidden", action="store_true")

    classify_query_parser = subparsers.add_parser(
        "classify-query",
        parents=[db_parent],
        help="Classify a natural-language query into a local routing intent.",
    )
    classify_query_parser.add_argument("query")
    classify_query_parser.add_argument("--json", action="store_true", dest="as_json")

    qa_parser = subparsers.add_parser(
        "qa",
        parents=[db_parent],
        help="Classify and route a natural-language query to local retrieval.",
    )
    qa_parser.add_argument("query")
    qa_parser.add_argument("--limit", type=int, default=5)
    qa_parser.add_argument("--include-hidden", action="store_true")
    qa_parser.add_argument("--json", action="store_true", dest="as_json")

    ui_parser = subparsers.add_parser(
        "ui",
        parents=[db_parent],
        help="Launch the optional localhost-only Gradio UI.",
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7860)

    build_embeddings_parser = subparsers.add_parser(
        "build-embeddings",
        parents=[db_parent],
        help="Build local embeddings for LINE messages and available media text.",
    )
    build_embeddings_parser.add_argument("--backend", default=None)
    build_embeddings_parser.add_argument("--model", default=None)

    search_parser = subparsers.add_parser(
        "search",
        parents=[db_parent],
        help="Search local SQLite text records by keyword.",
    )
    search_parser.add_argument("query")
    search_parser.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    search_parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--date-from", default=None)
    search_parser.add_argument("--date-to", default=None)
    search_parser.add_argument("--mode", choices=["all", "actual", "plan", "mention"], default="all")
    search_parser.add_argument(
        "--intent",
        choices=["place_visit", "food_activity", "call_activity", "topic_mention", "generic"],
        default=None,
    )
    search_parser.add_argument("--include-hidden", action="store_true")
    search_parser.add_argument("--json", action="store_true", dest="as_json")

    build_call_index_parser = subparsers.add_parser(
        "build-call-index",
        parents=[db_parent],
        help="Extract structured LINE call events from line_messages.",
    )
    build_call_index_parser.add_argument("--from", dest="from_date", default=None)
    build_call_index_parser.add_argument("--to", dest="to_date", default=None)
    build_call_index_parser.add_argument("--force", action="store_true")
    build_call_index_parser.add_argument("--dry-run", action="store_true")

    call_stats_parser = subparsers.add_parser(
        "call-stats",
        parents=[db_parent],
        help="Show local structured LINE call statistics.",
    )
    call_stats_parser.add_argument("--from", dest="from_date", default=None)
    call_stats_parser.add_argument("--to", dest="to_date", default=None)
    call_stats_parser.add_argument("--month", default=None)
    call_stats_parser.add_argument("--json", action="store_true", dest="as_json")

    search_calls_parser = subparsers.add_parser(
        "search-calls",
        parents=[db_parent],
        help="Search structured LINE call events.",
    )
    search_calls_parser.add_argument("--completed", action="store_true")
    search_calls_parser.add_argument("--missed", action="store_true")
    search_calls_parser.add_argument("--unanswered", action="store_true")
    search_calls_parser.add_argument("--canceled", action="store_true")
    search_calls_parser.add_argument("--min-duration-sec", type=int, default=None)
    search_calls_parser.add_argument("--from", dest="from_date", default=None)
    search_calls_parser.add_argument("--to", dest="to_date", default=None)
    search_calls_parser.add_argument("--limit", type=int, default=20)
    search_calls_parser.add_argument("--json", action="store_true", dest="as_json")

    search_snapshot_parser = subparsers.add_parser(
        "search-snapshot",
        parents=[db_parent],
        help="Capture compact local search outputs for ranking evaluation.",
    )
    search_snapshot_parser.add_argument("--query", action="append", required=True)
    search_snapshot_parser.add_argument("--limit", type=int, default=5)
    search_snapshot_parser.add_argument("--date-from", default=None)
    search_snapshot_parser.add_argument("--date-to", default=None)
    search_snapshot_parser.add_argument("--save", action="store_true")
    search_snapshot_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEARCH_SNAPSHOT_DIR)
    search_snapshot_parser.add_argument("--json", action="store_true", dest="as_json")

    analyze_images_parser = subparsers.add_parser(
        "analyze-images",
        parents=[db_parent],
        help="Run optional local OCR/VLM analysis for imported images.",
    )
    analyze_images_parser.add_argument("--date", default=None)
    analyze_images_parser.add_argument("--from", dest="from_date", default=None)
    analyze_images_parser.add_argument("--to", dest="to_date", default=None)
    analyze_images_parser.add_argument("--all", action="store_true")
    analyze_images_parser.add_argument("--limit", type=int, default=100)
    analyze_images_parser.add_argument("--engine", default=None)
    analyze_images_parser.add_argument("--model", default=None)
    analyze_images_parser.add_argument("--dry-run", action="store_true")
    analyze_images_parser.add_argument("--force", action="store_true")
    analyze_images_parser.add_argument("--skip-existing", action="store_true")
    analyze_images_parser.add_argument("--only-with-ocr", action="store_true")
    analyze_images_parser.add_argument("--only-gps", action="store_true")
    analyze_images_parser.add_argument("--ocr-backend", default=None)
    analyze_images_parser.add_argument("--vlm-backend", default=None)
    analyze_images_parser.add_argument("--vlm-model", default=None)

    vlm_stats_parser = subparsers.add_parser(
        "vlm-stats",
        parents=[db_parent],
        help="Show local VLM coverage and tag counts.",
    )
    vlm_stats_parser.add_argument("--from", dest="from_date", default=None)
    vlm_stats_parser.add_argument("--to", dest="to_date", default=None)
    vlm_stats_parser.add_argument("--json", action="store_true", dest="as_json")

    vlm_show_parser = subparsers.add_parser(
        "vlm-show",
        parents=[db_parent],
        help="Show compact VLM records by media id or date.",
    )
    vlm_show_parser.add_argument("media_id", nargs="?")
    vlm_show_parser.add_argument("--date", default=None)
    vlm_show_parser.add_argument("--limit", type=int, default=10)
    vlm_show_parser.add_argument("--full", action="store_true")

    image_search_parser = subparsers.add_parser(
        "image-search",
        parents=[db_parent],
        help="Search local OCR/VLM/photo metadata for image content.",
    )
    image_search_parser.add_argument("query")
    image_search_parser.add_argument("--from", dest="from_date", default=None)
    image_search_parser.add_argument("--to", dest="to_date", default=None)
    image_search_parser.add_argument("--limit", type=int, default=20)
    image_search_parser.add_argument("--json", action="store_true", dest="as_json")

    ocr_images_parser = subparsers.add_parser(
        "ocr-images",
        parents=[db_parent],
        help="Run optional local OCR over imported image files.",
    )
    ocr_images_parser.add_argument("--date", default=None)
    ocr_images_parser.add_argument("--from", dest="from_date", default=None)
    ocr_images_parser.add_argument("--to", dest="to_date", default=None)
    ocr_images_parser.add_argument("--all", action="store_true")
    ocr_images_parser.add_argument("--limit", type=int, default=100)
    ocr_images_parser.add_argument("--engine", default=None)
    ocr_images_parser.add_argument("--languages", default="jpn+eng")
    ocr_images_parser.add_argument("--dry-run", action="store_true")
    ocr_images_parser.add_argument("--force", action="store_true")
    ocr_images_parser.add_argument("--skip-existing", action="store_true")

    ocr_stats_parser = subparsers.add_parser(
        "ocr-stats",
        parents=[db_parent],
        help="Show OCR coverage and status counts.",
    )
    ocr_stats_parser.add_argument("--from", dest="from_date", default=None)
    ocr_stats_parser.add_argument("--to", dest="to_date", default=None)
    ocr_stats_parser.add_argument("--json", action="store_true", dest="as_json")

    ocr_show_parser = subparsers.add_parser(
        "ocr-show",
        parents=[db_parent],
        help="Show compact OCR records by media id or date.",
    )
    ocr_show_parser.add_argument("media_id", nargs="?")
    ocr_show_parser.add_argument("--date", default=None)
    ocr_show_parser.add_argument("--limit", type=int, default=10)
    ocr_show_parser.add_argument("--full", action="store_true")

    build_events_parser = subparsers.add_parser(
        "build-events",
        parents=[db_parent],
        help="Build timeline event candidates from local LINE/photo evidence.",
    )
    build_events_parser.add_argument("--date", default=None, help="Single date, e.g. 2024-12-24.")
    build_events_parser.add_argument("--from", dest="from_date", default=None, help="Start date, e.g. 2024-12-01.")
    build_events_parser.add_argument("--to", dest="to_date", default=None, help="End date, e.g. 2024-12-31.")
    build_events_parser.add_argument("--all", action="store_true", help="Build events for all dates with records.")
    build_events_parser.add_argument("--dry-run", action="store_true", help="Preview target days and generated drafts without writing.")
    build_events_parser.add_argument("--skip-existing", action="store_true", help="Skip dates that already have events.")
    build_events_parser.add_argument("--force", action="store_true", help="Replace generated events for target dates before saving.")
    build_events_parser.add_argument("--limit-days", type=int, default=None, help="Maximum number of target days to process.")
    build_events_parser.add_argument("--backup", action="store_true", help="Back up the SQLite DB before writing events.")
    build_events_parser.add_argument("--check-after", action="store_true", help="Run strict DB checks after writing events.")

    rebuild_safe_parser = subparsers.add_parser(
        "rebuild-events-safe",
        parents=[db_parent],
        help="Back up DB, preview, build events, run db-check, event-stats, and save search snapshot.",
    )
    rebuild_safe_parser.add_argument("--all", action="store_true", help="Rebuild for all dates with records.")
    rebuild_safe_parser.add_argument("--from", dest="from_date", default=None)
    rebuild_safe_parser.add_argument("--to", dest="to_date", default=None)
    rebuild_safe_parser.add_argument("--limit-days", type=int, default=None)
    rebuild_safe_parser.add_argument("--skip-existing", action="store_true")
    rebuild_safe_parser.add_argument("--force", action="store_true")
    rebuild_safe_parser.add_argument("--backup-label", default="before_rebuild_events_safe")
    rebuild_safe_parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    rebuild_safe_parser.add_argument("--snapshot-query", action="append", default=None)
    rebuild_safe_parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SEARCH_SNAPSHOT_DIR)

    event_stats_parser = subparsers.add_parser(
        "event-stats",
        parents=[db_parent],
        help="Show monthly/daily/title event statistics.",
    )
    event_stats_parser.add_argument("--from", dest="from_date", default=None)
    event_stats_parser.add_argument("--to", dest="to_date", default=None)
    event_stats_parser.add_argument("--json", action="store_true", dest="as_json")

    list_events_parser = subparsers.add_parser(
        "list-events",
        parents=[db_parent],
        help="List generated or saved events in a date range.",
    )
    list_events_parser.add_argument("--date", default=None)
    list_events_parser.add_argument("--from", dest="from_date", default=None)
    list_events_parser.add_argument("--to", dest="to_date", default=None)
    list_events_parser.add_argument("--with-evidence", action="store_true")
    list_events_parser.add_argument("--include-hidden", action="store_true")
    list_events_parser.add_argument("--json", action="store_true", dest="as_json")

    update_event_parser = subparsers.add_parser(
        "update-event",
        parents=[db_parent],
        help="Save a manual event override without modifying generated evidence.",
    )
    update_event_parser.add_argument("event_id")
    update_event_parser.add_argument("--title", default=None)
    update_event_parser.add_argument("--summary", default=None)
    update_event_parser.add_argument("--location", default=None)
    update_event_parser.add_argument("--tag", action="append", default=None)
    update_event_parser.add_argument("--verified", action="store_true")
    update_event_parser.add_argument("--hidden", action="store_true")
    update_event_parser.add_argument("--pinned", action="store_true")
    update_event_parser.add_argument("--clear-overrides", action="store_true")

    review_queue_parser = subparsers.add_parser(
        "review-queue",
        parents=[db_parent],
        help="List events that need manual review.",
    )
    review_queue_parser.add_argument("--date", default=None)
    review_queue_parser.add_argument("--from", dest="from_date", default=None)
    review_queue_parser.add_argument("--to", dest="to_date", default=None)
    review_queue_parser.add_argument("--low-confidence", type=float, default=None)
    review_queue_parser.add_argument("--title-contains", default=None)
    review_queue_parser.add_argument("--location-contains", default=None)
    review_queue_parser.add_argument("--modality", choices=["all", "line_only", "photo_only", "photo_and_line", "no_evidence"], default="all")
    review_queue_parser.add_argument("--line-only", action="store_true")
    review_queue_parser.add_argument("--verified", action="store_true")
    review_queue_parser.add_argument("--unverified", action="store_true")
    review_queue_parser.add_argument("--include-hidden", action="store_true")
    review_queue_parser.add_argument("--hidden-only", action="store_true")
    review_queue_parser.add_argument("--pinned-only", action="store_true")
    review_queue_parser.add_argument("--evidence-min", type=int, default=None)
    review_queue_parser.add_argument("--evidence-max", type=int, default=None)
    review_queue_parser.add_argument("--title-category", default=None)
    review_queue_parser.add_argument("--limit", type=int, default=100)
    review_queue_parser.add_argument("--json", action="store_true", dest="as_json")

    bulk_update_parser = subparsers.add_parser(
        "bulk-update-events",
        parents=[db_parent],
        help="Apply one override flag/tag change to multiple event ids.",
    )
    bulk_update_parser.add_argument("--event-id", action="append", required=True)
    bulk_update_parser.add_argument("--verified", action="store_true")
    bulk_update_parser.add_argument("--hidden", action="store_true")
    bulk_update_parser.add_argument("--unhide", action="store_true")
    bulk_update_parser.add_argument("--pinned", action="store_true")
    bulk_update_parser.add_argument("--tag", action="append", default=None)
    bulk_update_parser.add_argument("--clear-overrides", action="store_true")
    bulk_update_parser.add_argument("--json", action="store_true", dest="as_json")

    make_eval_case_parser = subparsers.add_parser(
        "make-eval-case",
        parents=[db_parent],
        help="Generate a private-eval YAML fragment from an event or query.",
    )
    make_eval_case_parser.add_argument("--event-id", default=None)
    make_eval_case_parser.add_argument("--type", dest="case_type", default="routed_qa")
    make_eval_case_parser.add_argument("--query", default=None)
    make_eval_case_parser.add_argument("--expected-date", default=None)

    inspect_date_parser = subparsers.add_parser(
        "inspect-date",
        parents=[db_parent],
        help="Inspect one date's local DB records for answer-quality debugging.",
    )
    inspect_date_parser.add_argument("date")
    inspect_date_parser.add_argument("--limit", type=int, default=20)
    inspect_date_parser.add_argument("--no-snippets", action="store_true")
    inspect_date_parser.add_argument(
        "--places-path",
        type=Path,
        default=None,
        help="Optional local places.yaml used to label GPS summaries.",
    )

    db_check_parser = subparsers.add_parser(
        "db-check",
        parents=[db_parent],
        help="Run privacy-conscious SQLite integrity checks.",
    )
    db_check_parser.add_argument("--json", action="store_true", dest="as_json")
    db_check_parser.add_argument("--strict", action="store_true")

    private_eval_parser = subparsers.add_parser(
        "private-eval",
        aliases=["eval-private"],
        parents=[db_parent],
        help="Run local private evaluation questions without external APIs.",
    )
    private_eval_parser.add_argument("--questions", "--path", dest="questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    private_eval_parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    private_eval_parser.add_argument("--json", action="store_true", dest="as_json")
    private_eval_parser.add_argument("--limit", type=int, default=None)
    private_eval_parser.add_argument("--case-id", default=None, help="Run only the eval case with this id.")
    private_eval_parser.add_argument("--save-run", action="store_true", default=False, help="Save a compact JSON run report.")
    private_eval_parser.add_argument("--no-save-run", action="store_false", dest="save_run", help="Do not save a run report.")
    private_eval_parser.add_argument("--init-template", action="store_true")
    private_eval_parser.add_argument("--strict", action="store_true")

    eval_compare_parser = subparsers.add_parser(
        "eval-compare",
        parents=[db_parent],
        help="Compare two saved private eval JSON reports.",
    )
    eval_compare_parser.add_argument("--before", type=Path, required=True)
    eval_compare_parser.add_argument("--after", type=Path, required=True)
    eval_compare_parser.add_argument("--json", action="store_true", dest="as_json")

    places_parser = subparsers.add_parser(
        "places",
        help="Validate, list, or match local private place dictionaries.",
    )
    places_subparsers = places_parser.add_subparsers(dest="places_command", required=True)
    places_init_private_parser = places_subparsers.add_parser(
        "init-private",
        help="Create private_config/places.yaml from the dummy example without overwriting.",
    )
    places_init_private_parser.add_argument("--path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)
    places_init_private_parser.add_argument("--source", type=Path, default=Path("configs/places.example.yaml"))
    places_validate_parser = places_subparsers.add_parser("validate", help="Validate a places.yaml file.")
    places_validate_parser.add_argument("--path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)
    places_list_parser = places_subparsers.add_parser("list", help="List configured safe place labels.")
    places_list_parser.add_argument("--path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)
    places_match_parser = places_subparsers.add_parser("match", help="Match one GPS point locally.")
    places_match_parser.add_argument("--lat", type=float, required=True)
    places_match_parser.add_argument("--lon", type=float, required=True)
    places_match_parser.add_argument("--path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)
    places_redact_preview_parser = places_subparsers.add_parser(
        "redact-preview",
        help="Preview privacy-safe place display labels.",
    )
    places_redact_preview_parser.add_argument("--path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)

    cluster_places_parser = subparsers.add_parser(
        "cluster-places",
        parents=[db_parent],
        help="Cluster GPS-tagged photos into local place suggestions.",
    )
    cluster_places_parser.add_argument("--from", dest="from_date", default=None)
    cluster_places_parser.add_argument("--to", dest="to_date", default=None)
    cluster_places_parser.add_argument("--all", action="store_true")
    cluster_places_parser.add_argument("--radius-m", type=float, default=500.0)
    cluster_places_parser.add_argument("--min-points", type=int, default=5)
    cluster_places_parser.add_argument("--output", type=Path, default=None)
    cluster_places_parser.add_argument("--places-path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)

    assign_places_parser = subparsers.add_parser(
        "assign-places",
        parents=[db_parent],
        help="Assign local place dictionary names to events with GPS.",
    )
    assign_places_parser.add_argument("--date", default=None)
    assign_places_parser.add_argument("--from", dest="from_date", default=None)
    assign_places_parser.add_argument("--to", dest="to_date", default=None)
    assign_places_parser.add_argument("--all", action="store_true")
    assign_places_parser.add_argument("--path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)
    assign_places_parser.add_argument("--dry-run", action="store_true")

    place_stats_parser = subparsers.add_parser(
        "place-stats",
        parents=[db_parent],
        help="Show location_name event statistics without exposing raw GPS.",
    )
    place_stats_parser.add_argument("--from", dest="from_date", default=None)
    place_stats_parser.add_argument("--to", dest="to_date", default=None)
    place_stats_parser.add_argument("--places-path", type=Path, default=DEFAULT_PRIVATE_PLACES_PATH)
    place_stats_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def run_init_db(db_path: Path | None) -> int:
    resolved_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_path)
    repository.initialize()
    print(f"Initialized database: {resolved_path}")
    return 0


def run_stats(db_path: Path | None) -> int:
    resolved_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_path)
    repository.initialize()
    stats = repository.stats()
    print(f"Database: {resolved_path}")
    for table_name, count in stats.items():
        print(f"{table_name}: {count}")
    return 0


def run_backup_db(
    db_path: Path | None,
    *,
    label: str | None,
    output_dir: Path,
) -> int:
    resolved_path = resolve_db_path(db_path)
    try:
        result = backup_sqlite_db(resolved_path, label=label, output_dir=output_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Backed up database:")
    print(f"- source: {result.source_path}")
    print(f"- backup: {result.backup_path}")
    print(f"- size_bytes: {result.size_bytes}")
    return 0


def run_ingest_line(db_path: Path | None, path: Path, chat_name: str | None) -> int:
    resolved_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_path)
    repository.initialize()
    source_files = _line_export_files(path)
    imported = 0
    parsed = 0
    warnings = 0
    for source_file in source_files:
        result = parse_line_chat_file_with_warnings(source_file, chat_name=chat_name)
        parsed += len(result.messages)
        warnings += len(result.warnings)
        imported += repository.add_line_messages(result.messages)
    skipped = parsed - imported
    print(
        "Imported LINE messages: "
        f"{imported} new, {skipped} duplicate, {warnings} warning(s), {len(source_files)} file(s)"
    )
    return 0


def run_ingest_photos(db_path: Path | None, path: Path) -> int:
    resolved_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_path)
    repository.initialize()
    report = ingest_photo_directory_with_report(path, repository)
    print(
        "Imported media files: "
        f"{report.imported} new, {report.duplicates} duplicate, "
        f"{report.skipped} skipped, {report.scanned} file(s)"
    )
    return 0


def run_ask(db_path: Path | None, question: str, *, include_hidden: bool = False) -> int:
    resolved_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_path)
    date_range = parse_date_query(question)
    result = search_timeline(repository, question, date_range=date_range, include_hidden=include_hidden)
    print(build_answer(question, result))
    return 0


def run_classify_query(query: str, *, as_json: bool) -> int:
    result = classify_query_intent(query)
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_query_intent(result))
    return 0


def run_qa(
    db_path: Path | None,
    query: str,
    *,
    limit: int,
    include_hidden: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    result = route_query(repository, query, limit=limit, include_hidden=include_hidden)
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_routed_query_result(result))
    return 0


def run_ui(db_path: Path | None, host: str, port: int) -> int:
    try:
        from personal_lifelog_rag.app.gradio_app import launch

        launch(db_path=resolve_db_path(db_path), host=host, port=port)
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def run_build_embeddings(
    db_path: Path | None,
    *,
    backend: str | None,
    model_name: str | None,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    adapter = get_embedding_adapter(backend=backend, model_name=model_name)
    report = build_embeddings(repository, adapter)
    print(
        "Built embeddings: "
        f"{report.embedded} row(s), "
        f"line_messages={report.line_messages_seen}, "
        f"media_items={report.media_items_seen}, "
        f"model={report.model_name}"
    )
    return 0


def run_search(
    db_path: Path | None,
    query: str,
    *,
    limit: int,
    date_from: str | None,
    date_to: str | None,
    mode: str,
    intent: str | None,
    include_hidden: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = local_text_search(
        repository,
        LocalSearchOptions(
            query=query,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            mode=mode,  # type: ignore[arg-type]
            intent=intent,  # type: ignore[arg-type]
            include_hidden=include_hidden,
        ),
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_local_search_report(report))
    return 0


def run_build_call_index(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    force: bool,
    dry_run: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = build_call_index(
        repository,
        start_date=from_date,
        end_date=to_date or from_date,
        force=force,
        dry_run=dry_run,
    )
    print(format_build_call_index_report(report))
    return 0


def run_call_stats(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    month: str | None,
    as_json: bool,
) -> int:
    if month and (from_date or to_date):
        raise ValueError("call-stats accepts either --month or --from/--to, not both")
    start_date = from_date
    end_date = to_date or from_date
    if month:
        start_date, end_date = _month_range(month)
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = call_stats(repository, start_date=start_date, end_date=end_date)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_call_stats(report))
    return 0


def run_search_calls(
    db_path: Path | None,
    *,
    completed: bool,
    missed: bool,
    unanswered: bool,
    canceled: bool,
    min_duration_sec: int | None,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    as_json: bool,
) -> int:
    statuses = _call_status_filters(
        completed=completed,
        missed=missed,
        unanswered=unanswered,
        canceled=canceled,
    )
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = search_calls(
        repository,
        statuses=statuses,
        min_duration_sec=min_duration_sec,
        start_date=from_date,
        end_date=to_date or from_date,
        limit=limit,
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_search_calls_report(report))
    return 0


def run_search_snapshot(
    db_path: Path | None,
    *,
    queries: list[str],
    limit: int,
    date_from: str | None,
    date_to: str | None,
    save: bool,
    output_dir: Path,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    snapshot = build_search_snapshot(
        repository,
        SearchSnapshotOptions(
            queries=queries,
            limit=limit,
            date_from=date_from,
            date_to=date_to,
        ),
    )
    output_path = write_search_snapshot(snapshot, output_dir=output_dir) if save else None
    if as_json:
        payload = dict(snapshot)
        if output_path is not None:
            payload["output_path"] = str(output_path)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_search_snapshot(snapshot, output_path=output_path))
    return 0


def run_analyze_images(
    db_path: Path | None,
    *,
    date_value: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    all_dates: bool = False,
    limit: int,
    engine_name: str | None = None,
    model_name: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    skip_existing: bool = False,
    only_with_ocr: bool = False,
    only_gps: bool = False,
    ocr_backend: str | None,
    vlm_backend: str | None,
    vlm_model: str | None,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    use_vlm_table = any(
        [
            date_value,
            from_date,
            all_dates,
            engine_name,
            model_name,
            dry_run,
            force,
            skip_existing,
            only_with_ocr,
            only_gps,
        ]
    )
    if use_vlm_table:
        start_date, end_date = _resolve_range_selection(
            date_value=date_value,
            from_date=from_date,
            to_date=to_date,
            all_dates=all_dates,
            command_name="analyze-images",
            allow_all_without_dates=True,
        )

        def progress(message: str) -> None:
            print(message, file=sys.stderr)

        report = run_vlm_images(
            repository,
            VlmImagesOptions(
                start_date=start_date,
                end_date=end_date,
                all_dates=all_dates,
                limit=limit,
                engine_name=engine_name,
                model_name=model_name,
                dry_run=dry_run,
                force=force,
                skip_existing=skip_existing,
                only_with_ocr=only_with_ocr,
                only_gps=only_gps,
            ),
            engine=get_vlm_engine(engine_name, model_name=model_name),
            progress_callback=progress,
        )
        print(format_vlm_report(report))
        return 0

    report = analyze_images(
        repository,
        limit=limit,
        ocr_adapter=get_ocr_adapter(backend=ocr_backend),
        vlm_adapter=get_vlm_adapter(backend=vlm_backend, model_name=vlm_model),
    )
    print(format_analysis_report(report))
    return 0


def run_vlm_stats_cli(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = vlm_stats(repository, start_date=from_date, end_date=to_date or from_date)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_vlm_stats(report))
    return 0


def run_vlm_show_cli(
    db_path: Path | None,
    *,
    media_id: str | None,
    date_value: str | None,
    limit: int,
    full: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if media_id:
        row = repository.get_media_vlm(media_id)
        rows = [row] if row else []
    else:
        rows = repository.list_media_vlm(
            start_date=date_value,
            end_date=date_value,
            limit=limit,
        )
    print(format_vlm_show(rows, full=full))
    return 0


def run_image_search_cli(
    db_path: Path | None,
    *,
    query: str,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = image_search(
        repository,
        ImageSearchOptions(query=query, date_from=from_date, date_to=to_date, limit=limit),
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_image_search(report))
    return 0


def run_ocr_images_cli(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    all_dates: bool,
    limit: int,
    engine_name: str | None,
    languages: str,
    dry_run: bool,
    force: bool,
    skip_existing: bool,
) -> int:
    start_date, end_date = _resolve_range_selection(
        date_value=date_value,
        from_date=from_date,
        to_date=to_date,
        all_dates=all_dates,
        command_name="ocr-images",
        allow_all_without_dates=True,
    )
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    report = run_ocr_images(
        repository,
        OcrImagesOptions(
            start_date=start_date,
            end_date=end_date,
            all_dates=all_dates,
            limit=limit,
            engine_name=engine_name,
            languages=_parse_languages(languages),
            dry_run=dry_run,
            force=force,
            skip_existing=skip_existing,
        ),
        engine=get_ocr_engine(engine_name),
        progress_callback=progress,
    )
    print(format_ocr_report(report))
    return 0


def run_ocr_stats_cli(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = ocr_stats(repository, start_date=from_date, end_date=to_date or from_date)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_ocr_stats(report))
    return 0


def run_ocr_show_cli(
    db_path: Path | None,
    *,
    media_id: str | None,
    date_value: str | None,
    limit: int,
    full: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if media_id:
        row = repository.get_media_ocr(media_id)
        rows = [row] if row else []
    else:
        rows = repository.list_media_ocr(
            start_date=date_value,
            end_date=date_value,
            limit=limit,
        )
    print(format_ocr_show(rows, full=full))
    return 0


def run_build_events(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    all_dates: bool,
    dry_run: bool,
    skip_existing: bool,
    force: bool,
    limit_days: int | None,
    backup: bool = False,
    check_after: bool = False,
) -> int:
    selected_modes = sum(1 for value in (date_value, from_date, all_dates) if value)
    if selected_modes != 1:
        raise ValueError("build-events requires exactly one of --date, --from, or --all")

    resolved_db_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_db_path)
    repository.initialize()
    if backup and not dry_run:
        backup_result = backup_sqlite_db(resolved_db_path, label="before_build_events", output_dir=DEFAULT_BACKUP_DIR)
        print(f"Backup: {backup_result.backup_path} ({backup_result.size_bytes} bytes)")
    before_inputs = _input_table_counts(repository)
    event_config = _event_build_config()
    effective_force = True if dry_run else (force or not skip_existing)

    def progress(message: str) -> None:
        if not dry_run:
            print(message, file=sys.stderr)

    if all_dates:
        report = build_all_events(
            repository,
            config=event_config,
            dry_run=dry_run,
            skip_existing=skip_existing,
            force=effective_force,
            limit_days=limit_days,
            progress_callback=progress,
        )
    elif date_value:
        start_date = date_value
        end_date = date_value
        report = build_events(
            repository,
            start_date=start_date,
            end_date=end_date,
            config=event_config,
            dry_run=dry_run,
            skip_existing=skip_existing,
            force=effective_force,
            limit_days=limit_days,
            progress_callback=progress,
        )
    elif from_date:
        start_date = from_date
        end_date = to_date or from_date
        report = build_events(
            repository,
            start_date=start_date,
            end_date=end_date,
            config=event_config,
            dry_run=dry_run,
            skip_existing=skip_existing,
            force=effective_force,
            limit_days=limit_days,
            progress_callback=progress,
        )
    after_inputs = _input_table_counts(repository)
    if after_inputs != before_inputs:
        raise RuntimeError(
            "build-events safety check failed: media_items or line_messages changed "
            f"before={before_inputs} after={after_inputs}"
        )
    print(_format_build_events_report(report, before_inputs=before_inputs, after_inputs=after_inputs))
    if check_after and not dry_run:
        check_report = run_db_check(resolved_db_path)
        print("")
        print(format_db_check(check_report))
        if not check_report["strict"]["ok"]:
            return 1
    return 0


def run_rebuild_events_safe(
    db_path: Path | None,
    *,
    all_dates: bool,
    from_date: str | None,
    to_date: str | None,
    limit_days: int | None,
    skip_existing: bool,
    force: bool,
    backup_label: str,
    backup_dir: Path,
    snapshot_queries: list[str] | None,
    snapshot_dir: Path,
) -> int:
    if not all_dates and from_date is None:
        raise ValueError("rebuild-events-safe requires --all or --from")
    resolved_db_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_db_path)
    repository.initialize()

    backup_result = backup_sqlite_db(resolved_db_path, label=backup_label, output_dir=backup_dir)
    print(f"Backup: {backup_result.backup_path} ({backup_result.size_bytes} bytes)")
    print("")
    print("Dry-run preview:")
    preview_exit = run_build_events(
        db_path,
        date_value=None,
        from_date=from_date,
        to_date=to_date,
        all_dates=all_dates,
        dry_run=True,
        skip_existing=skip_existing,
        force=force,
        limit_days=limit_days,
    )
    if preview_exit:
        return preview_exit
    print("")
    print("Building events:")
    build_exit = run_build_events(
        db_path,
        date_value=None,
        from_date=from_date,
        to_date=to_date,
        all_dates=all_dates,
        dry_run=False,
        skip_existing=skip_existing,
        force=force,
        limit_days=limit_days,
        check_after=True,
    )
    if build_exit:
        return build_exit

    print("")
    print("Event stats:")
    stats_report = event_stats(repository)
    print(format_event_stats(stats_report))
    print("")
    queries = snapshot_queries or DEFAULT_SEARCH_SNAPSHOT_QUERIES
    snapshot = build_search_snapshot(
        repository,
        SearchSnapshotOptions(queries=queries, limit=5),
    )
    snapshot_path = write_search_snapshot(snapshot, output_dir=snapshot_dir)
    print(format_search_snapshot(snapshot, output_path=snapshot_path))
    return 0


def run_event_stats(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = event_stats(repository, start_date=from_date, end_date=to_date)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_event_stats(report))
    return 0


def run_list_events(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    with_evidence: bool,
    include_hidden: bool,
    as_json: bool,
) -> int:
    if date_value and (from_date or to_date):
        raise ValueError("list-events accepts either --date or --from/--to, not both")
    start_date = date_value or from_date
    end_date = date_value or to_date or from_date
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    rows = list_events_report(
        repository,
        start_date=start_date,
        end_date=end_date,
        with_evidence=with_evidence,
        include_hidden=include_hidden,
    )
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_event_list(rows, with_evidence=with_evidence))
    return 0


def run_update_event(
    db_path: Path | None,
    *,
    event_id: str,
    title: str | None,
    summary: str | None,
    location: str | None,
    tags: list[str] | None,
    verified: bool,
    hidden: bool,
    pinned: bool,
    clear_overrides: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if repository.get_event(event_id, include_hidden=True) is None:
        print(f"Event not found: {event_id}", file=sys.stderr)
        return 1
    if clear_overrides:
        deleted = repository.delete_event_override(event_id)
        print(f"Cleared event override: {event_id} ({deleted} row(s))")
        return 0
    if not any([title is not None, summary is not None, location is not None, tags is not None, verified, hidden, pinned]):
        print("No update options were provided.", file=sys.stderr)
        return 1
    override = save_event_review_override(
        repository,
        event_id,
        title=title,
        summary=summary,
        location_name=location,
        tags=tags,
        is_verified=verified if verified else None,
        is_hidden=hidden if hidden else None,
        is_pinned=pinned if pinned else None,
    )
    event = repository.get_event(event_id, include_hidden=True) or {}
    print("Updated event override:")
    print(f"- event_id: {event_id}")
    print(f"- title: {event.get('title') or ''}")
    print(f"- summary: {redact_text(event.get('summary'), max_chars=120)}")
    print(f"- location_name: {event.get('location_name') or ''}")
    print(f"- tags_json: {override.get('tags_json') or ''}")
    print(f"- is_verified: {int(override.get('is_verified') or 0)}")
    print(f"- is_hidden: {int(override.get('is_hidden') or 0)}")
    print(f"- is_pinned: {int(override.get('is_pinned') or 0)}")
    return 0


def run_review_queue(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    low_confidence: float | None,
    title_contains: str | None,
    location_contains: str | None,
    modality: str,
    line_only: bool,
    verified: bool,
    unverified: bool,
    include_hidden: bool,
    hidden_only: bool,
    pinned_only: bool,
    evidence_min: int | None,
    evidence_max: int | None,
    title_category: str | None,
    limit: int,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    verified_filter = "all"
    if verified and unverified:
        raise ValueError("review-queue accepts only one of --verified or --unverified")
    if verified:
        verified_filter = "verified"
    if unverified:
        verified_filter = "unverified"
    hidden_filter = "only" if hidden_only else ("include" if include_hidden else "exclude")
    report = review_queue(
        repository,
        ReviewQueueFilters(
            date=date_value,
            date_from=from_date,
            date_to=to_date,
            confidence_lte=low_confidence,
            title_contains=title_contains,
            location_contains=location_contains,
            modality="line_only" if line_only else modality,
            verified=verified_filter,
            hidden=hidden_filter,
            pinned="pinned" if pinned_only else "all",
            evidence_count_min=evidence_min,
            evidence_count_max=evidence_max,
            title_category=title_category,
            limit=limit,
        ),
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_review_queue(report))
    return 0


def run_bulk_update_events(
    db_path: Path | None,
    *,
    event_ids: list[str],
    verified: bool,
    hidden: bool,
    unhide: bool,
    pinned: bool,
    tags: list[str] | None,
    clear_overrides: bool,
    as_json: bool,
) -> int:
    if hidden and unhide:
        raise ValueError("bulk-update-events accepts only one of --hidden or --unhide")
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = bulk_update_events(
        repository,
        event_ids,
        verified=True if verified else None,
        hidden=(False if unhide else (True if hidden else None)),
        pinned=True if pinned else None,
        add_tags=tags,
        clear_overrides=clear_overrides,
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print("Bulk event update")
        print(f"- updated: {report['updated_count']}")
        print(f"- missing: {report['missing_count']}")
        for event_id in report["updated"][:20]:
            print(f"- {event_id}")
    return 0


def run_make_eval_case(
    db_path: Path | None,
    *,
    event_id: str | None,
    case_type: str,
    query: str | None,
    expected_date: str | None,
) -> int:
    if not event_id and not query:
        raise ValueError("make-eval-case requires --event-id or --query")
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if event_id and repository.get_event(event_id, include_hidden=True) is None:
        print(f"Event not found: {event_id}", file=sys.stderr)
        return 1
    print(
        make_eval_case_yaml(
            repository,
            event_id=event_id,
            case_type=case_type,
            query=query,
            expected_date=expected_date,
        )
    )
    return 0


def run_inspect_date(
    db_path: Path | None,
    *,
    target_date: str,
    limit: int,
    no_snippets: bool,
    places_path: Path | None = None,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    place_dictionary = _load_optional_places(places_path)
    inspection = inspect_date(
        repository,
        target_date,
        limit=limit,
        include_snippets=not no_snippets,
        place_dictionary=place_dictionary,
    )
    print(format_date_inspection(inspection))
    return 0


def run_places_validate(path: Path) -> int:
    validation = validate_place_dictionary(path)
    if validation.valid:
        print(f"Places config valid: {path} ({len(validation.places)} place(s))")
        return 0
    print(f"Places config invalid: {path}", file=sys.stderr)
    for error in validation.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


def run_places_init_private(path: Path, *, source: Path) -> int:
    target = path.expanduser()
    source_path = source.expanduser()
    if target.exists():
        print(f"Private places config already exists, not overwritten: {target}")
        print("Edit it manually when you want to add real local-only place labels.")
        return 0
    if not source_path.exists():
        print(f"Places example not found: {source_path}", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created private places config: {target}")
    print("This file contains dummy coordinates only. Edit it manually before assign-places.")
    print("Do not commit private_config/ or place_suggestions.yaml.")
    return 0


def run_places_list(path: Path) -> int:
    places = load_place_dictionary(path)
    lines = [f"Places: {len(places)}"]
    for place in places:
        center = privacy_safe_lat_lon(
            place.lat,
            place.lon,
            show_exact_location=place.show_exact_location,
            privacy_level=place.privacy_level,
        )
        lines.append(
            f"- {place.id}: {place.display_name}, "
            f"category={place.category}, "
            f"radius_m={place.radius_m:g}, "
            f"privacy_level={place.privacy_level}, "
            f"show_exact_location={place.show_exact_location}, "
            f"center={center}"
        )
    print("\n".join(lines))
    return 0


def run_places_match(path: Path, *, lat: float, lon: float) -> int:
    places = load_place_dictionary(path)
    match = match_place(lat, lon, places)
    if match is None:
        print("No registered place matched.")
        return 0
    print("Matched place:")
    print(f"- place_id: {match.place_id}")
    print(f"- display_name: {match.display_name}")
    print(f"- distance_m: {match.distance_m:.1f}")
    print(f"- privacy_level: {match.privacy_level}")
    print(f"- show_exact_location: {match.show_exact_location}")
    if not match.show_exact_location:
        print("- exact GPS display: disabled")
    return 0


def run_places_redact_preview(path: Path) -> int:
    places = load_place_dictionary(path)
    print(format_place_display_preview(place_display_preview(places)))
    return 0


def run_cluster_places(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    all_dates: bool,
    radius_m: float,
    min_points: int,
    output: Path | None,
    places_path: Path | None,
) -> int:
    start_date, end_date = _resolve_range_selection(
        date_value=None,
        from_date=from_date,
        to_date=to_date,
        all_dates=all_dates,
        command_name="cluster-places",
        allow_all_without_dates=True,
    )
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    media_items = repository.list_media_items(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
    )
    places = _load_optional_places(places_path)
    clusters = cluster_place_candidates(
        media_items,
        radius_m=radius_m,
        min_points=min_points,
        places=places,
    )
    print(format_place_clusters(clusters))
    if output is not None:
        written = write_place_cluster_suggestions(output, clusters)
        print(f"\nSaved place suggestions: {written}")
    return 0


def run_assign_places(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    all_dates: bool,
    path: Path,
    dry_run: bool,
) -> int:
    start_date, end_date = _resolve_range_selection(
        date_value=date_value,
        from_date=from_date,
        to_date=to_date,
        all_dates=all_dates,
        command_name="assign-places",
        allow_all_without_dates=True,
    )
    places = load_place_dictionary(path)
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = assign_places_to_events(
        repository,
        places=places,
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
    )
    print(format_assign_places_report(report))
    return 0


def run_place_stats(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    places_path: Path | None,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    places = _load_optional_places(places_path)
    report = place_stats(
        repository,
        start_date=from_date,
        end_date=to_date or from_date,
        places=places,
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_place_stats(report))
    return 0


def run_db_check_cli(
    db_path: Path | None,
    *,
    as_json: bool,
    strict: bool,
) -> int:
    report = run_db_check(resolve_db_path(db_path))
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_db_check(report))
    if strict and not report["strict"]["ok"]:
        return 1
    return 0


def run_private_eval(
    db_path: Path | None,
    *,
    questions_path: Path,
    output_dir: Path,
    as_json: bool,
    limit: int | None,
    case_id: str | None,
    save_run: bool,
    init_template: bool,
    strict: bool,
) -> int:
    if init_template:
        template_path = write_private_eval_template(questions_path)
        print(f"Created private eval template: {template_path}")
        return 0

    if not questions_path.expanduser().exists():
        print(
            f"Private eval questions not found: {questions_path}. "
            "Run private-eval --init-template first.",
            file=sys.stderr,
        )
        return 1

    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    questions = load_private_eval_questions(questions_path)
    if case_id is not None:
        questions = [question for question in questions if question.id == case_id]
        if not questions:
            print(f"Private eval case not found: {case_id}", file=sys.stderr)
            return 1
    if limit is not None:
        questions = questions[: max(limit, 0)]
    report = evaluate_private_questions(repository, questions)
    output_path = write_private_eval_report(report, output_dir) if save_run else None
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_private_eval_report(report, output_path=output_path))
    if strict and report["summary"]["failed"]:
        return 1
    return 0


def run_eval_compare(
    *,
    before: Path,
    after: Path,
    as_json: bool,
) -> int:
    report = compare_private_eval_reports(before, after)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_private_eval_comparison(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = getattr(args, "db_path", None)

    if args.command == "init-db":
        return run_init_db(db_path)
    if args.command == "stats":
        return run_stats(db_path)
    if args.command == "backup-db":
        return run_backup_db(db_path, label=args.label, output_dir=args.output_dir)
    if args.command == "ingest-line":
        path = args.path or args.legacy_path
        if path is None:
            parser.error("ingest-line requires --path or a path argument")
        return run_ingest_line(db_path, path, args.chat_name)
    if args.command == "ingest-photos":
        path = args.path or args.legacy_path
        if path is None:
            parser.error("ingest-photos requires --path or a path argument")
        return run_ingest_photos(db_path, path)
    if args.command == "ask":
        return run_ask(db_path, args.question, include_hidden=args.include_hidden)
    if args.command == "classify-query":
        return run_classify_query(args.query, as_json=args.as_json)
    if args.command == "qa":
        return run_qa(
            db_path,
            args.query,
            limit=args.limit,
            include_hidden=args.include_hidden,
            as_json=args.as_json,
        )
    if args.command == "ui":
        return run_ui(db_path, args.host, args.port)
    if args.command == "build-embeddings":
        return run_build_embeddings(db_path, backend=args.backend, model_name=args.model)
    if args.command == "search":
        return run_search(
            db_path,
            args.query,
            limit=args.limit,
            date_from=args.date_from,
            date_to=args.date_to,
            mode=args.mode,
            intent=args.intent,
            include_hidden=args.include_hidden,
            as_json=args.as_json,
        )
    if args.command == "build-call-index":
        return run_build_call_index(
            db_path,
            from_date=args.from_date,
            to_date=args.to_date,
            force=args.force,
            dry_run=args.dry_run,
        )
    if args.command == "call-stats":
        try:
            return run_call_stats(
                db_path,
                from_date=args.from_date,
                to_date=args.to_date,
                month=args.month,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "search-calls":
        return run_search_calls(
            db_path,
            completed=args.completed,
            missed=args.missed,
            unanswered=args.unanswered,
            canceled=args.canceled,
            min_duration_sec=args.min_duration_sec,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            as_json=args.as_json,
        )
    if args.command == "search-snapshot":
        return run_search_snapshot(
            db_path,
            queries=args.query,
            limit=args.limit,
            date_from=args.date_from,
            date_to=args.date_to,
            save=args.save,
            output_dir=args.output_dir,
            as_json=args.as_json,
        )
    if args.command == "analyze-images":
        try:
            return run_analyze_images(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                all_dates=args.all,
                limit=args.limit,
                engine_name=args.engine,
                model_name=args.model or args.vlm_model,
                dry_run=args.dry_run,
                force=args.force,
                skip_existing=args.skip_existing,
                only_with_ocr=args.only_with_ocr,
                only_gps=args.only_gps,
                ocr_backend=args.ocr_backend,
                vlm_backend=args.vlm_backend,
                vlm_model=args.vlm_model,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "vlm-stats":
        return run_vlm_stats_cli(
            db_path,
            from_date=args.from_date,
            to_date=args.to_date,
            as_json=args.as_json,
        )
    if args.command == "vlm-show":
        return run_vlm_show_cli(
            db_path,
            media_id=args.media_id,
            date_value=args.date,
            limit=args.limit,
            full=args.full,
        )
    if args.command == "image-search":
        return run_image_search_cli(
            db_path,
            query=args.query,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            as_json=args.as_json,
        )
    if args.command == "ocr-images":
        try:
            return run_ocr_images_cli(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                all_dates=args.all,
                limit=args.limit,
                engine_name=args.engine,
                languages=args.languages,
                dry_run=args.dry_run,
                force=args.force,
                skip_existing=args.skip_existing,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "ocr-stats":
        return run_ocr_stats_cli(
            db_path,
            from_date=args.from_date,
            to_date=args.to_date,
            as_json=args.as_json,
        )
    if args.command == "ocr-show":
        return run_ocr_show_cli(
            db_path,
            media_id=args.media_id,
            date_value=args.date,
            limit=args.limit,
            full=args.full,
        )
    if args.command == "build-events":
        try:
            return run_build_events(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                all_dates=args.all,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                force=args.force,
                limit_days=args.limit_days,
                backup=args.backup,
                check_after=args.check_after,
            )
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.command == "rebuild-events-safe":
        try:
            return run_rebuild_events_safe(
                db_path,
                all_dates=args.all,
                from_date=args.from_date,
                to_date=args.to_date,
                limit_days=args.limit_days,
                skip_existing=args.skip_existing,
                force=args.force,
                backup_label=args.backup_label,
                backup_dir=args.backup_dir,
                snapshot_queries=args.snapshot_query,
                snapshot_dir=args.snapshot_dir,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.command == "event-stats":
        return run_event_stats(
            db_path,
            from_date=args.from_date,
            to_date=args.to_date,
            as_json=args.as_json,
        )
    if args.command == "list-events":
        try:
            return run_list_events(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                with_evidence=args.with_evidence,
                include_hidden=args.include_hidden,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "update-event":
        return run_update_event(
            db_path,
            event_id=args.event_id,
            title=args.title,
            summary=args.summary,
            location=args.location,
            tags=args.tag,
            verified=args.verified,
            hidden=args.hidden,
            pinned=args.pinned,
            clear_overrides=args.clear_overrides,
        )
    if args.command == "review-queue":
        try:
            return run_review_queue(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                low_confidence=args.low_confidence,
                title_contains=args.title_contains,
                location_contains=args.location_contains,
                modality=args.modality,
                line_only=args.line_only,
                verified=args.verified,
                unverified=args.unverified,
                include_hidden=args.include_hidden,
                hidden_only=args.hidden_only,
                pinned_only=args.pinned_only,
                evidence_min=args.evidence_min,
                evidence_max=args.evidence_max,
                title_category=args.title_category,
                limit=args.limit,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "bulk-update-events":
        try:
            return run_bulk_update_events(
                db_path,
                event_ids=args.event_id,
                verified=args.verified,
                hidden=args.hidden,
                unhide=args.unhide,
                pinned=args.pinned,
                tags=args.tag,
                clear_overrides=args.clear_overrides,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "make-eval-case":
        try:
            return run_make_eval_case(
                db_path,
                event_id=args.event_id,
                case_type=args.case_type,
                query=args.query,
                expected_date=args.expected_date,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "inspect-date":
        try:
            return run_inspect_date(
                db_path,
                target_date=args.date,
                limit=args.limit,
                no_snippets=args.no_snippets,
                places_path=args.places_path,
            )
        except (ValueError, PlaceConfigError) as exc:
            parser.error(str(exc))
    if args.command == "places":
        try:
            if args.places_command == "init-private":
                return run_places_init_private(args.path, source=args.source)
            if args.places_command == "validate":
                return run_places_validate(args.path)
            if args.places_command == "list":
                return run_places_list(args.path)
            if args.places_command == "match":
                return run_places_match(args.path, lat=args.lat, lon=args.lon)
            if args.places_command == "redact-preview":
                return run_places_redact_preview(args.path)
        except PlaceConfigError as exc:
            parser.error(str(exc))
    if args.command == "cluster-places":
        try:
            return run_cluster_places(
                db_path,
                from_date=args.from_date,
                to_date=args.to_date,
                all_dates=args.all,
                radius_m=args.radius_m,
                min_points=args.min_points,
                output=args.output,
                places_path=args.places_path,
            )
        except (ValueError, PlaceConfigError) as exc:
            parser.error(str(exc))
    if args.command == "assign-places":
        try:
            return run_assign_places(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                all_dates=args.all,
                path=args.path,
                dry_run=args.dry_run,
            )
        except (ValueError, PlaceConfigError) as exc:
            parser.error(str(exc))
    if args.command == "place-stats":
        try:
            return run_place_stats(
                db_path,
                from_date=args.from_date,
                to_date=args.to_date,
                places_path=args.places_path,
                as_json=args.as_json,
            )
        except PlaceConfigError as exc:
            parser.error(str(exc))
    if args.command == "db-check":
        return run_db_check_cli(
            db_path,
            as_json=args.as_json,
            strict=args.strict,
        )
    if args.command in {"private-eval", "eval-private"}:
        return run_private_eval(
            db_path,
            questions_path=args.questions,
            output_dir=args.output_dir,
            as_json=args.as_json,
            limit=args.limit,
            case_id=args.case_id,
            save_run=args.save_run,
            init_template=args.init_template,
            strict=args.strict,
        )
    if args.command == "eval-compare":
        return run_eval_compare(
            before=args.before,
            after=args.after,
            as_json=args.as_json,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


def _line_export_files(path: Path) -> list[Path]:
    expanded = path.expanduser()
    if expanded.is_file():
        return [expanded]
    if expanded.is_dir():
        return sorted(item for item in expanded.rglob("*.txt") if item.is_file())
    raise FileNotFoundError(f"LINE export path does not exist: {expanded}")


def _load_optional_places(path: Path | None = None):
    resolved = Path(path or DEFAULT_PRIVATE_PLACES_PATH).expanduser()
    if not resolved.exists():
        return []
    return load_place_dictionary(resolved, required=False)


def _call_status_filters(
    *,
    completed: bool,
    missed: bool,
    unanswered: bool,
    canceled: bool,
) -> list[str] | None:
    statuses: list[str] = []
    if completed:
        statuses.append("completed")
    if missed:
        statuses.append("missed")
    if unanswered:
        statuses.append("unanswered")
    if canceled:
        statuses.append("canceled")
    return statuses or None


def _parse_languages(value: str | None) -> list[str]:
    if not value:
        return ["jpn", "eng"]
    return [part.strip() for part in value.replace(",", "+").split("+") if part.strip()]


def _month_range(month: str) -> tuple[str, str]:
    year_text, month_text = month.split("-", 1)
    first_day = date(int(year_text), int(month_text), 1)
    if first_day.month == 12:
        next_month = date(first_day.year + 1, 1, 1)
    else:
        next_month = date(first_day.year, first_day.month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return first_day.isoformat(), last_day.isoformat()


def _resolve_range_selection(
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    all_dates: bool,
    command_name: str,
    allow_all_without_dates: bool,
) -> tuple[str | None, str | None]:
    selected = sum(1 for value in (date_value, from_date, all_dates) if value)
    if selected != 1:
        modes = "--date, --from, or --all" if date_value is not None or command_name == "assign-places" else "--from or --all"
        raise ValueError(f"{command_name} requires exactly one of {modes}")
    if all_dates:
        if not allow_all_without_dates:
            raise ValueError(f"{command_name} does not support --all")
        return None, None
    if date_value is not None:
        return date_value, date_value
    if from_date is not None:
        return from_date, to_date or from_date
    raise ValueError(f"{command_name} requires a date range")


def _event_build_config() -> EventBuildConfig:
    file_config = EventBuildConfig.from_mapping(load_event_building_config())
    return EventBuildConfig.from_env(base=file_config)


def _input_table_counts(repository: LifelogRepository) -> dict[str, int]:
    stats = repository.stats()
    return {
        "media_items": stats.get("media_items", 0),
        "line_messages": stats.get("line_messages", 0),
    }


def _format_build_events_report(
    report,
    *,
    before_inputs: dict[str, int] | None = None,
    after_inputs: dict[str, int] | None = None,
) -> str:
    verb = "Dry-run events" if report.dry_run else "Built events"
    events_value = report.events_planned if report.dry_run else report.events_created
    evidence_value = report.evidence_planned if report.dry_run else report.evidence_saved
    lines = [
        f"{verb}: "
        f"{events_value} event(s), "
        f"{evidence_value} evidence row(s), "
        f"{report.events_deleted} old event(s) replaced, "
        f"{report.days_scanned} day(s), "
        f"{report.days_skipped} skipped, "
        f"range={report.start_date}..{report.end_date}",
    ]
    if report.day_reports:
        lines.append("Day summary:")
        for day in report.day_reports[:30]:
            titles = day.get("titles") or {}
            title_text = ", ".join(f"{title}={count}" for title, count in titles.items()) if titles else "none"
            lines.append(
                f"- {day['date']}: {day['action']}, "
                f"existing={day['existing_events']}, "
                f"events={day['events']}, "
                f"evidence={day['evidence']}, "
                f"titles={title_text}"
            )
        if len(report.day_reports) > 30:
            lines.append(f"- ... {len(report.day_reports) - 30} more day(s)")
    if before_inputs is not None and after_inputs is not None:
        stable = before_inputs == after_inputs
        lines.extend(
            [
                "Input table safety:",
                f"- media_items: {before_inputs['media_items']} -> {after_inputs['media_items']}",
                f"- line_messages: {before_inputs['line_messages']} -> {after_inputs['line_messages']}",
                f"- unchanged: {stable}",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
