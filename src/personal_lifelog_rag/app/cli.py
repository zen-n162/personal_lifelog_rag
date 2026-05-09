"""Command line interface for personal_lifelog_rag."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from personal_lifelog_rag.benchmark.benchmark_report import (
    DEFAULT_BENCHMARK_OUTPUT_DIR,
    assemble_report,
    build_multimodal_benchmark_report,
    format_benchmark_summary,
    format_model_info,
    model_info,
    write_benchmark_report,
)
from personal_lifelog_rag.benchmark.embedding_benchmark import (
    benchmark_image_embedding,
    engine_from_spec as embedding_engine_from_spec,
)
from personal_lifelog_rag.benchmark.schemas import (
    ModelRuntimeConfig,
    load_benchmark_cases,
    load_model_runtime_config,
)
from personal_lifelog_rag.benchmark.vlm_benchmark import (
    benchmark_vlm,
    engine_from_spec as vlm_engine_from_spec,
)
from personal_lifelog_rag.captioning.image_analysis import analyze_images, format_analysis_report
from personal_lifelog_rag.captioning.local_vlm import get_vlm_adapter
from personal_lifelog_rag.core.config import load_event_building_config
from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.backup import DEFAULT_BACKUP_DIR, backup_sqlite_db
from personal_lifelog_rag.db.checks import format_db_check, run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository, resolve_db_path
from personal_lifelog_rag.embeddings.adapter import get_embedding_adapter
from personal_lifelog_rag.embeddings.embedding_service import (
    build_image_embeddings,
    build_text_embeddings,
    embedding_stats,
    format_embedding_build_report,
    format_embedding_stats,
)
from personal_lifelog_rag.embeddings.engines import (
    get_cached_multimodal_embedding_engine,
    get_multimodal_embedding_engine,
)
from personal_lifelog_rag.embeddings.multimodal_search import (
    format_multimodal_search,
    multimodal_search,
)
from personal_lifelog_rag.embeddings.schemas import (
    BuildMediaEmbeddingsOptions,
    MultimodalSearchOptions,
)
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
from personal_lifelog_rag.evaluation.private_eval_templates import (
    format_private_eval_template_summary,
    write_private_eval_template_for_date,
)
from personal_lifelog_rag.evaluation.search_snapshot import (
    DEFAULT_SEARCH_SNAPSHOT_DIR,
    DEFAULT_SEARCH_SNAPSHOT_QUERIES,
    SearchSnapshotOptions,
    build_search_snapshot,
    format_search_snapshot,
    write_search_snapshot,
)
from personal_lifelog_rag.env_check import format_env_check, run_env_check
from personal_lifelog_rag.fake_analysis_cleanup import (
    cleanup_fake_analysis,
    format_cleanup_fake_analysis,
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
from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository
from personal_lifelog_rag.jobs.job_service import (
    dumps_json,
    format_analysis_cleanup,
    format_analysis_plan,
    format_analysis_run_report,
    format_analysis_status,
)
from personal_lifelog_rag.jobs.planners import plan_analysis
from personal_lifelog_rag.jobs.runners import (
    DEFAULT_ANALYSIS_JOB_OUTPUT_DIR,
    resume_analysis_job,
    retry_failed_analysis,
    run_analysis_job,
)
from personal_lifelog_rag.jobs.schemas import AnalysisPlanOptions, AnalysisRunOptions
from personal_lifelog_rag.jobs.storage import (
    format_db_maintenance,
    format_storage_stats,
    run_db_maintenance,
    storage_stats,
)
from personal_lifelog_rag.model_diagnostics import format_model_diagnostics, run_model_diagnostics
from personal_lifelog_rag.ocr.local_ocr import get_ocr_adapter
from personal_lifelog_rag.ocr.engines import get_ocr_engine
from personal_lifelog_rag.ocr.config import load_ocr_runtime_config
from personal_lifelog_rag.ocr.diagnostics import format_ocr_diagnostics, run_ocr_diagnostics
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
from personal_lifelog_rag.reporting.report_builder import DEFAULT_REPORTS_DIR, build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions
from personal_lifelog_rag.rollout.monthly_rollout import (
    DEFAULT_CONFIG_PATH as DEFAULT_MONTH_ROLLOUT_CONFIG_PATH,
    format_month_batch_plan,
    format_month_plan,
    format_month_run_plan,
    format_month_status,
    month_batch_plan,
    month_plan,
    month_run_plan,
    month_status,
    parse_month,
)
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.timeline.event_builder import EventBuildConfig, build_all_events, build_events
from personal_lifelog_rag.timeline.event_reports import (
    event_stats,
    format_event_list,
    format_event_stats,
    list_events_report,
)
from personal_lifelog_rag.timeline.event_rebuild_analysis import (
    DEFAULT_EVENT_REBUILD_OUTPUT_DIR,
    EventRebuildOptions,
    diff_event_snapshots,
    format_event_diff,
    format_event_rebuild_report,
    load_snapshot_or_report,
    rebuild_events_with_analysis,
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
from personal_lifelog_rag.vlm.prompts import get_vlm_prompt_template
from personal_lifelog_rag.vlm.safety import safety_check_text
from personal_lifelog_rag.vlm.pilot import VlmPilotOptions, run_vlm_pilot
from personal_lifelog_rag.vlm.pilot_report import (
    DEFAULT_VLM_PILOT_OUTPUT_DIR,
    format_vlm_pilot_report,
)
from personal_lifelog_rag.vlm.review_service import (
    VlmOverrideUpdate,
    VlmReviewFilters,
    bulk_update_vlm_overrides,
    clear_vlm_override,
    format_vlm_review_queue,
    generate_vlm_eval_case,
    list_vlm_review_items,
    parse_tag_text,
    save_vlm_override,
)
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.status_cleanup import (
    VALID_CLEANUP_STATUSES,
    cleanup_vlm_status,
    format_cleanup_vlm_status,
)
from personal_lifelog_rag.vlm.vlm_service import (
    VlmImagesOptions,
    format_recover_failed_vlm_report,
    format_image_search,
    format_vlm_report,
    format_vlm_show,
    format_vlm_stats,
    image_search,
    recover_failed_vlm_json_rows,
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

    month_plan_parser = subparsers.add_parser(
        "month-plan",
        parents=[db_parent],
        help="Plan a safe month-by-month VLM/embedding/rebuild rollout.",
    )
    month_plan_parser.add_argument("--month", required=True, help="Target month in YYYY-MM format.")
    month_plan_parser.add_argument("--limit", type=int, default=300)
    month_plan_parser.add_argument("--config", type=Path, default=DEFAULT_MONTH_ROLLOUT_CONFIG_PATH)
    month_plan_parser.add_argument("--json", action="store_true", dest="as_json")

    month_run_parser = subparsers.add_parser(
        "month-run",
        parents=[db_parent],
        help="Run or dry-run a safe monthly rollout pipeline.",
    )
    month_run_parser.add_argument("--month", required=True, help="Target month in YYYY-MM format.")
    month_run_parser.add_argument("--limit", type=int, default=300)
    month_run_parser.add_argument("--vlm-limit", type=int, default=None)
    month_run_parser.add_argument("--embedding-limit", type=int, default=None)
    month_run_parser.add_argument("--config", type=Path, default=DEFAULT_MONTH_ROLLOUT_CONFIG_PATH)
    month_run_parser.add_argument("--dry-run", action="store_true")
    month_run_parser.add_argument("--skip-vlm", action="store_true")
    month_run_parser.add_argument("--skip-embedding", action="store_true")
    month_run_parser.add_argument("--skip-rebuild", action="store_true")
    month_run_parser.add_argument("--skip-eval", action="store_true")
    month_run_parser.add_argument("--skip-report", action="store_true")
    month_run_parser.add_argument("--save-report", action="store_true")
    month_run_parser.add_argument("--yes", action="store_true")
    month_run_parser.add_argument("--json", action="store_true", dest="as_json")

    month_status_parser = subparsers.add_parser(
        "month-status",
        parents=[db_parent],
        help="Show month rollout progress and artifact presence.",
    )
    month_status_parser.add_argument("--month", required=True, help="Target month in YYYY-MM format.")
    month_status_parser.add_argument("--json", action="store_true", dest="as_json")

    month_batch_parser = subparsers.add_parser(
        "month-batch",
        parents=[db_parent],
        help="Dry-run planning for several monthly rollouts.",
    )
    month_batch_parser.add_argument("--from-month", required=True)
    month_batch_parser.add_argument("--to-month", required=True)
    month_batch_parser.add_argument("--limit", type=int, default=300)
    month_batch_parser.add_argument("--config", type=Path, default=DEFAULT_MONTH_ROLLOUT_CONFIG_PATH)
    month_batch_parser.add_argument("--dry-run", action="store_true")
    month_batch_parser.add_argument("--json", action="store_true", dest="as_json")

    analysis_plan_parser = subparsers.add_parser(
        "analysis-plan",
        parents=[db_parent],
        help="Plan local OCR/VLM/embedding/event analysis work before running it.",
    )
    _add_analysis_target_args(analysis_plan_parser)
    _add_analysis_filter_args(analysis_plan_parser)
    analysis_plan_parser.add_argument("--json", action="store_true", dest="as_json")

    analysis_run_parser = subparsers.add_parser(
        "analysis-run",
        parents=[db_parent],
        help="Run a local analysis job with DB-backed progress metadata.",
    )
    _add_analysis_target_args(analysis_run_parser)
    _add_analysis_filter_args(analysis_run_parser)
    analysis_run_parser.add_argument("--dry-run", action="store_true")
    analysis_run_parser.add_argument("--job-id", default=None)
    analysis_run_parser.add_argument("--save-report", action="store_true")
    analysis_run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_JOB_OUTPUT_DIR)
    analysis_run_parser.add_argument("--allow-fake-write", action="store_true")
    analysis_run_parser.add_argument("--json", action="store_true", dest="as_json")

    analysis_status_parser = subparsers.add_parser(
        "analysis-status",
        parents=[db_parent],
        help="Show recent analysis jobs or one job with item status.",
    )
    analysis_status_parser.add_argument("--job-id", default=None)
    analysis_status_parser.add_argument("--recent", type=int, default=10)
    analysis_status_parser.add_argument("--json", action="store_true", dest="as_json")

    analysis_resume_parser = subparsers.add_parser(
        "analysis-resume",
        parents=[db_parent],
        help="Resume a prior analysis job by planning remaining failed/unavailable items.",
    )
    analysis_resume_parser.add_argument("--job-id", required=True)
    analysis_resume_parser.add_argument("--failed-only", action="store_true")
    analysis_resume_parser.add_argument("--engine-unavailable-only", action="store_true")
    analysis_resume_parser.add_argument("--limit", type=int, default=None)
    analysis_resume_parser.add_argument("--dry-run", action="store_true")
    analysis_resume_parser.add_argument("--save-report", action="store_true")
    analysis_resume_parser.add_argument("--json", action="store_true", dest="as_json")

    analysis_retry_parser = subparsers.add_parser(
        "analysis-retry-failed",
        parents=[db_parent],
        help="Retry failed analysis rows either from a previous job or a fresh scope.",
    )
    analysis_retry_parser.add_argument("--job-id", default=None)
    _add_analysis_target_args(analysis_retry_parser, required_type=False)
    _add_analysis_filter_args(analysis_retry_parser)
    analysis_retry_parser.add_argument("--dry-run", action="store_true")
    analysis_retry_parser.add_argument("--save-report", action="store_true")
    analysis_retry_parser.add_argument("--json", action="store_true", dest="as_json")

    analysis_cleanup_parser = subparsers.add_parser(
        "analysis-cleanup",
        parents=[db_parent],
        help="Safely clean analysis job metadata; real deletion requires --yes.",
    )
    analysis_cleanup_parser.add_argument("--dry-run", action="store_true")
    analysis_cleanup_parser.add_argument("--failed", action="store_true")
    analysis_cleanup_parser.add_argument("--engine-unavailable", action="store_true")
    analysis_cleanup_parser.add_argument("--old-runs", type=int, default=None)
    analysis_cleanup_parser.add_argument("--yes", action="store_true")
    analysis_cleanup_parser.add_argument("--json", action="store_true", dest="as_json")

    storage_stats_parser = subparsers.add_parser(
        "storage-stats",
        parents=[db_parent],
        help="Show local DB/artifact storage usage without exposing private content.",
    )
    storage_stats_parser.add_argument("--json", action="store_true", dest="as_json")

    db_maintenance_parser = subparsers.add_parser(
        "db-maintenance",
        parents=[db_parent],
        help="Run safe SQLite backup/vacuum maintenance.",
    )
    db_maintenance_parser.add_argument("--backup", action="store_true")
    db_maintenance_parser.add_argument("--vacuum", action="store_true")
    db_maintenance_parser.add_argument("--yes", action="store_true")
    db_maintenance_parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    db_maintenance_parser.add_argument("--json", action="store_true", dest="as_json")

    model_diagnostics_parser = subparsers.add_parser(
        "model-diagnostics",
        help="Diagnose configured local Qwen VLM/embedding runtimes without loading model weights.",
    )
    model_diagnostics_parser.add_argument("--config", type=Path, default=None)
    model_diagnostics_parser.add_argument("--json", action="store_true", dest="as_json")

    cleanup_fake_parser = subparsers.add_parser(
        "cleanup-fake-analysis",
        parents=[db_parent],
        help="Remove test-only fake VLM/embedding rows; real deletion requires --yes.",
    )
    cleanup_fake_parser.add_argument("--dry-run", action="store_true")
    cleanup_fake_parser.add_argument("--yes", action="store_true")
    cleanup_fake_parser.add_argument("--include-engine-unavailable", action="store_true")
    cleanup_fake_parser.add_argument("--date", default=None)
    cleanup_fake_parser.add_argument("--from", dest="from_date", default=None)
    cleanup_fake_parser.add_argument("--to", dest="to_date", default=None)
    cleanup_fake_parser.add_argument("--json", action="store_true", dest="as_json")

    cleanup_vlm_status_parser = subparsers.add_parser(
        "cleanup-vlm-status",
        parents=[db_parent],
        help="Remove selected non-fake VLM status rows and related VLM event evidence; real deletion requires --yes.",
    )
    cleanup_vlm_status_parser.add_argument("--date", default=None)
    cleanup_vlm_status_parser.add_argument("--from", dest="from_date", default=None)
    cleanup_vlm_status_parser.add_argument("--to", dest="to_date", default=None)
    cleanup_vlm_status_parser.add_argument("--engine", default=None)
    cleanup_vlm_status_parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        choices=sorted(VALID_CLEANUP_STATUSES),
        default=None,
    )
    cleanup_vlm_status_parser.add_argument("--dry-run", action="store_true")
    cleanup_vlm_status_parser.add_argument("--yes", action="store_true")
    cleanup_vlm_status_parser.add_argument("--json", action="store_true", dest="as_json")

    env_check_parser = subparsers.add_parser(
        "env-check",
        help="Check active Python, conda, package, and local model-path health.",
    )
    env_check_parser.add_argument("--config", type=Path, default=None)
    env_check_parser.add_argument("--json", action="store_true", dest="as_json")

    generate_report_parser = subparsers.add_parser(
        "generate-report",
        parents=[db_parent],
        help="Generate a privacy-preserving Markdown evaluation report for research/portfolio use.",
    )
    generate_report_parser.add_argument("--from", dest="from_date", default=None)
    generate_report_parser.add_argument("--to", dest="to_date", default=None)
    mode_group = generate_report_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--public", action="store_true", dest="public_mode")
    mode_group.add_argument("--private", action="store_true", dest="private_mode")
    generate_report_parser.add_argument("--eval-path", type=Path, default=None)
    generate_report_parser.add_argument("--eval-run", type=Path, default=None)
    generate_report_parser.add_argument("--output", type=Path, default=None)
    examples_group = generate_report_parser.add_mutually_exclusive_group()
    examples_group.add_argument("--include-examples", action="store_true", dest="include_examples")
    examples_group.add_argument("--no-examples", action="store_true", dest="no_examples")
    generate_report_parser.add_argument("--save-json", action="store_true")

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

    batch_qa_parser = subparsers.add_parser(
        "batch-qa",
        parents=[db_parent],
        help="Run multiple natural-language QA queries in one local process.",
    )
    batch_qa_parser.add_argument("--query", action="append", required=True, help="Question to run. Repeat for multiple queries.")
    batch_qa_parser.add_argument("--limit", type=int, default=5)
    batch_qa_parser.add_argument("--include-hidden", action="store_true")
    batch_qa_parser.add_argument("--config", type=Path, default=None, help="Optional private model runtime config.")
    batch_qa_parser.add_argument("--output-json", type=Path, default=None)
    batch_qa_parser.add_argument("--output-md", type=Path, default=None)
    batch_qa_parser.add_argument("--save-run", action="store_true")

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
    analyze_images_parser.add_argument("--config", type=Path, default=None)
    analyze_images_parser.add_argument("--dry-run", action="store_true")
    analyze_images_parser.add_argument("--force", action="store_true")
    analyze_images_parser.add_argument("--skip-existing", action="store_true")
    analyze_images_parser.add_argument("--failed-only", action="store_true")
    analyze_images_parser.add_argument("--only-with-ocr", action="store_true")
    analyze_images_parser.add_argument("--only-gps", action="store_true")
    analyze_images_parser.add_argument("--ocr-backend", default=None)
    analyze_images_parser.add_argument("--vlm-backend", default=None)
    analyze_images_parser.add_argument("--vlm-model", default=None)
    analyze_images_parser.add_argument("--prompt-template", default=None)
    analyze_images_parser.add_argument("--allow-fake-write", action="store_true")

    retry_vlm_failed_parser = subparsers.add_parser(
        "retry-vlm-failed",
        parents=[db_parent],
        help="Retry local VLM rows whose previous status is failed.",
    )
    retry_vlm_failed_parser.add_argument("--date", default=None)
    retry_vlm_failed_parser.add_argument("--from", dest="from_date", default=None)
    retry_vlm_failed_parser.add_argument("--to", dest="to_date", default=None)
    retry_vlm_failed_parser.add_argument("--limit", type=int, default=20)
    retry_vlm_failed_parser.add_argument("--engine", default=None)
    retry_vlm_failed_parser.add_argument("--model", default=None)
    retry_vlm_failed_parser.add_argument("--config", type=Path, default=None)
    retry_vlm_failed_parser.add_argument("--prompt-template", default=None)
    retry_vlm_failed_parser.add_argument("--dry-run", action="store_true")
    retry_vlm_failed_parser.add_argument("--rerun-model", action="store_true")
    retry_vlm_failed_parser.add_argument("--allow-fake-write", action="store_true")

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
    vlm_show_parser.add_argument("--show-errors", action="store_true")

    vlm_review_parser = subparsers.add_parser(
        "vlm-review-queue",
        parents=[db_parent],
        help="List local VLM records that need human review.",
    )
    vlm_review_parser.add_argument("--date", default=None)
    vlm_review_parser.add_argument("--from", dest="from_date", default=None)
    vlm_review_parser.add_argument("--to", dest="to_date", default=None)
    vlm_review_parser.add_argument("--status", default=None)
    vlm_review_parser.add_argument("--unreviewed", action="store_true")
    vlm_review_parser.add_argument("--safety-flags", action="store_true")
    vlm_review_parser.add_argument("--people-present", action="store_true")
    vlm_review_parser.add_argument("--low-confidence", type=float, default=None)
    vlm_review_parser.add_argument("--food-cues", action="store_true")
    vlm_review_parser.add_argument("--location-cues", action="store_true")
    vlm_review_parser.add_argument("--ocr", action="store_true", dest="has_ocr")
    vlm_review_parser.add_argument("--embedding", action="store_true", dest="has_embedding")
    vlm_review_parser.add_argument("--hidden", action="store_true")
    vlm_review_parser.add_argument("--wrong", action="store_true")
    vlm_review_parser.add_argument("--limit", type=int, default=50)
    vlm_review_parser.add_argument("--json", action="store_true", dest="as_json")

    update_vlm_parser = subparsers.add_parser(
        "update-vlm-result",
        parents=[db_parent],
        help="Save a local human review override for one VLM result.",
    )
    update_vlm_parser.add_argument("media_id")
    update_vlm_parser.add_argument("--caption", default=None)
    update_vlm_parser.add_argument("--short-caption", default=None)
    update_vlm_parser.add_argument("--tag", action="append", default=[])
    update_vlm_parser.add_argument("--scene-tag", action="append", default=[])
    update_vlm_parser.add_argument("--object-tag", action="append", default=[])
    update_vlm_parser.add_argument("--activity-tag", action="append", default=[])
    update_vlm_parser.add_argument("--location-cue", action="append", default=[])
    update_vlm_parser.add_argument("--accepted", action="store_true")
    update_vlm_parser.add_argument("--rejected", action="store_true")
    update_vlm_parser.add_argument("--wrong", action="store_true")
    update_vlm_parser.add_argument("--needs-fix", action="store_true")
    update_vlm_parser.add_argument("--verified", action="store_true")
    update_vlm_parser.add_argument("--hidden", action="store_true")
    update_vlm_parser.add_argument("--not-searchable", action="store_true")
    update_vlm_parser.add_argument("--not-event-usable", action="store_true")
    update_vlm_parser.add_argument("--note", default=None)
    update_vlm_parser.add_argument("--clear-override", action="store_true")
    update_vlm_parser.add_argument("--json", action="store_true", dest="as_json")

    bulk_vlm_parser = subparsers.add_parser(
        "bulk-update-vlm-results",
        parents=[db_parent],
        help="Apply one VLM review action to multiple media ids.",
    )
    bulk_vlm_parser.add_argument("--media-id", action="append", default=[])
    bulk_vlm_parser.add_argument("--from-file", type=Path, default=None)
    bulk_vlm_parser.add_argument("--accepted", action="store_true")
    bulk_vlm_parser.add_argument("--rejected", action="store_true")
    bulk_vlm_parser.add_argument("--wrong", action="store_true")
    bulk_vlm_parser.add_argument("--verified", action="store_true")
    bulk_vlm_parser.add_argument("--hidden", action="store_true")
    bulk_vlm_parser.add_argument("--not-searchable", action="store_true")
    bulk_vlm_parser.add_argument("--not-event-usable", action="store_true")
    bulk_vlm_parser.add_argument("--tag", action="append", default=[])
    bulk_vlm_parser.add_argument("--json", action="store_true", dest="as_json")

    make_vlm_eval_parser = subparsers.add_parser(
        "make-vlm-eval-case",
        parents=[db_parent],
        help="Print a private eval YAML snippet for a reviewed VLM result.",
    )
    make_vlm_eval_parser.add_argument("--media-id", default=None)
    make_vlm_eval_parser.add_argument("--query", default=None)
    make_vlm_eval_parser.add_argument("--expected-media-id", default=None)

    image_search_parser = subparsers.add_parser(
        "image-search",
        parents=[db_parent],
        help="Search local OCR/VLM/photo metadata for image content.",
    )
    image_search_parser.add_argument("query")
    image_search_parser.add_argument("--from", dest="from_date", default=None)
    image_search_parser.add_argument("--to", dest="to_date", default=None)
    image_search_parser.add_argument("--limit", type=int, default=20)
    image_search_parser.add_argument("--backend", choices=["sql", "vlm_sql", "embedding", "hybrid"], default="sql")
    image_search_parser.add_argument("--include-hidden", action="store_true")
    image_search_parser.add_argument("--json", action="store_true", dest="as_json")

    build_image_embeddings_parser = subparsers.add_parser(
        "build-image-embeddings",
        parents=[db_parent],
        help="Build local image embeddings for imported media without external APIs.",
    )
    build_image_embeddings_parser.add_argument("--date", default=None)
    build_image_embeddings_parser.add_argument("--from", dest="from_date", default=None)
    build_image_embeddings_parser.add_argument("--to", dest="to_date", default=None)
    build_image_embeddings_parser.add_argument("--limit", type=int, default=100)
    build_image_embeddings_parser.add_argument("--engine", default=None)
    build_image_embeddings_parser.add_argument("--model", default=None)
    build_image_embeddings_parser.add_argument("--model-path", default=None)
    build_image_embeddings_parser.add_argument("--config", type=Path, default=None)
    build_image_embeddings_parser.add_argument("--dry-run", action="store_true")
    build_image_embeddings_parser.add_argument("--force", action="store_true")
    build_image_embeddings_parser.add_argument("--skip-existing", action="store_true")
    build_image_embeddings_parser.add_argument("--allow-fake-write", action="store_true")

    build_text_embeddings_parser = subparsers.add_parser(
        "build-text-embeddings",
        parents=[db_parent],
        help="Build local media text embeddings from OCR/VLM caption metadata.",
    )
    build_text_embeddings_parser.add_argument("--date", default=None)
    build_text_embeddings_parser.add_argument("--from", dest="from_date", default=None)
    build_text_embeddings_parser.add_argument("--to", dest="to_date", default=None)
    build_text_embeddings_parser.add_argument("--limit", type=int, default=100)
    build_text_embeddings_parser.add_argument("--type", choices=["caption", "ocr", "combined_text"], default="combined_text")
    build_text_embeddings_parser.add_argument("--engine", default=None)
    build_text_embeddings_parser.add_argument("--model", default=None)
    build_text_embeddings_parser.add_argument("--model-path", default=None)
    build_text_embeddings_parser.add_argument("--config", type=Path, default=None)
    build_text_embeddings_parser.add_argument("--dry-run", action="store_true")
    build_text_embeddings_parser.add_argument("--force", action="store_true")
    build_text_embeddings_parser.add_argument("--skip-existing", action="store_true")
    build_text_embeddings_parser.add_argument("--allow-fake-write", action="store_true")

    embedding_stats_parser = subparsers.add_parser(
        "embedding-stats",
        parents=[db_parent],
        help="Show local media embedding coverage and status counts.",
    )
    embedding_stats_parser.add_argument("--from", dest="from_date", default=None)
    embedding_stats_parser.add_argument("--to", dest="to_date", default=None)
    embedding_stats_parser.add_argument("--json", action="store_true", dest="as_json")

    multimodal_search_parser = subparsers.add_parser(
        "multimodal-search",
        parents=[db_parent],
        help="Search images with local embeddings plus OCR/VLM/LINE/event reranking.",
    )
    multimodal_search_parser.add_argument("query")
    multimodal_search_parser.add_argument("--from", dest="from_date", default=None)
    multimodal_search_parser.add_argument("--to", dest="to_date", default=None)
    multimodal_search_parser.add_argument("--limit", type=int, default=10)
    multimodal_search_parser.add_argument("--backend", choices=["sql", "vlm_sql", "embedding", "hybrid"], default="hybrid")
    multimodal_search_parser.add_argument("--engine", default=None)
    multimodal_search_parser.add_argument("--model", default=None)
    multimodal_search_parser.add_argument("--model-path", default=None)
    multimodal_search_parser.add_argument("--config", type=Path, default=None)
    multimodal_search_parser.add_argument("--include-hidden", action="store_true")
    multimodal_search_parser.add_argument("--json", action="store_true", dest="as_json")

    vlm_prompt_parser = subparsers.add_parser(
        "vlm-prompt",
        help="Print a local VLM prompt template for dry-run review.",
    )
    vlm_prompt_parser.add_argument("--template", required=True)
    vlm_prompt_parser.add_argument("--json", action="store_true", dest="as_json")

    vlm_safety_parser = subparsers.add_parser(
        "vlm-safety-check",
        help="Run local VLM safety filtering on a short text snippet.",
    )
    vlm_safety_parser.add_argument("--text", required=True)
    vlm_safety_parser.add_argument("--json", action="store_true", dest="as_json")

    vlm_pilot_parser = subparsers.add_parser(
        "vlm-pilot",
        parents=[db_parent],
        help="Run a small local-only VLM analysis pilot with backup, checks, and smoke tests.",
    )
    vlm_pilot_parser.add_argument("--date", required=True)
    vlm_pilot_parser.add_argument("--limit", type=int, default=20)
    vlm_pilot_parser.add_argument("--engine", default=None)
    vlm_pilot_parser.add_argument("--model", default=None)
    vlm_pilot_parser.add_argument("--config", type=Path, default=None)
    vlm_pilot_parser.add_argument("--prompt-template", default="lifelog_structured_tags_v1")
    vlm_pilot_parser.add_argument("--dry-run", action="store_true")
    vlm_pilot_parser.add_argument("--save-report", action="store_true")
    vlm_pilot_parser.add_argument("--force", action="store_true")
    vlm_pilot_parser.add_argument("--skip-existing", action="store_true")
    vlm_pilot_parser.add_argument("--include-hidden", action="store_true")
    vlm_pilot_parser.add_argument(
        "--strategy",
        choices=["time_spread", "event_evidence", "ocr_first", "gps_first"],
        default="time_spread",
    )
    vlm_pilot_parser.add_argument("--output-dir", type=Path, default=DEFAULT_VLM_PILOT_OUTPUT_DIR)
    vlm_pilot_parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    vlm_pilot_parser.add_argument("--json", action="store_true", dest="as_json")

    vlm_model_info_parser = subparsers.add_parser(
        "vlm-model-info",
        help="Show configured local VLM and multimodal embedding model settings.",
    )
    vlm_model_info_parser.add_argument("--config", type=Path, default=None)
    vlm_model_info_parser.add_argument("--json", action="store_true", dest="as_json")

    benchmark_vlm_parser = subparsers.add_parser(
        "benchmark-vlm",
        help="Benchmark local image caption/tag extraction engines without external APIs.",
    )
    benchmark_vlm_parser.add_argument("--cases", type=Path, required=True)
    benchmark_vlm_parser.add_argument("--config", type=Path, default=None)
    benchmark_vlm_parser.add_argument("--engine", default=None)
    benchmark_vlm_parser.add_argument("--limit", type=int, default=None)
    benchmark_vlm_parser.add_argument("--save", action="store_true")
    benchmark_vlm_parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT_DIR)
    benchmark_vlm_parser.add_argument("--json", action="store_true", dest="as_json")

    benchmark_embedding_parser = subparsers.add_parser(
        "benchmark-image-embedding",
        help="Benchmark local text/image embedding retrieval engines.",
    )
    benchmark_embedding_parser.add_argument("--cases", type=Path, required=True)
    benchmark_embedding_parser.add_argument("--config", type=Path, default=None)
    benchmark_embedding_parser.add_argument("--engine", default=None)
    benchmark_embedding_parser.add_argument("--limit", type=int, default=None)
    benchmark_embedding_parser.add_argument("--save", action="store_true")
    benchmark_embedding_parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT_DIR)
    benchmark_embedding_parser.add_argument("--json", action="store_true", dest="as_json")

    benchmark_qwen_parser = subparsers.add_parser(
        "benchmark-qwen-multimodal",
        help="Run paired Qwen3-VL and Qwen3-VL-Embedding benchmark flows.",
    )
    benchmark_qwen_parser.add_argument("--cases", type=Path, required=True)
    benchmark_qwen_parser.add_argument("--config", type=Path, default=None)
    benchmark_qwen_parser.add_argument("--engine", default=None, help="Use one engine override for both VLM and embedding, e.g. fake.")
    benchmark_qwen_parser.add_argument("--vlm-engine", default=None)
    benchmark_qwen_parser.add_argument("--embedding-engine", default=None)
    benchmark_qwen_parser.add_argument("--limit", type=int, default=None)
    benchmark_qwen_parser.add_argument("--save", action="store_true")
    benchmark_qwen_parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT_DIR)
    benchmark_qwen_parser.add_argument("--json", action="store_true", dest="as_json")

    ocr_diag_parser = subparsers.add_parser(
        "ocr-diagnostics",
        help="Inspect local OCR engine availability without using cloud OCR.",
    )
    ocr_diag_parser.add_argument("--config", type=Path, default=None)
    ocr_diag_parser.add_argument("--json", action="store_true", dest="as_json")

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
    ocr_images_parser.add_argument("--config", type=Path, default=None)
    ocr_images_parser.add_argument("--languages", default=None)
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
    ocr_show_parser.add_argument("--show-errors", action="store_true")

    ocr_search_parser = subparsers.add_parser(
        "ocr-search",
        parents=[db_parent],
        help="Search local OCR text and redacted previews.",
    )
    ocr_search_parser.add_argument("query")
    ocr_search_parser.add_argument("--from", dest="from_date", default=None)
    ocr_search_parser.add_argument("--to", dest="to_date", default=None)
    ocr_search_parser.add_argument("--limit", type=int, default=20)
    ocr_search_parser.add_argument("--include-non-success", action="store_true")
    ocr_search_parser.add_argument("--json", action="store_true", dest="as_json")

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

    rebuild_analysis_parser = subparsers.add_parser(
        "rebuild-events-with-analysis",
        parents=[db_parent],
        help="Rebuild events after OCR/VLM analysis and save before/after quality diffs.",
    )
    rebuild_analysis_parser.add_argument("--date", default=None)
    rebuild_analysis_parser.add_argument("--from", dest="from_date", default=None)
    rebuild_analysis_parser.add_argument("--to", dest="to_date", default=None)
    rebuild_analysis_parser.add_argument("--dry-run", action="store_true")
    rebuild_analysis_parser.add_argument("--save-report", action="store_true")
    rebuild_analysis_parser.add_argument("--force", action="store_true")
    rebuild_analysis_parser.add_argument("--eval-path", type=Path, default=None)
    rebuild_analysis_parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVENT_REBUILD_OUTPUT_DIR)
    rebuild_analysis_parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    rebuild_analysis_parser.add_argument("--json", action="store_true", dest="as_json")

    event_diff_parser = subparsers.add_parser(
        "event-diff",
        help="Compare two event rebuild snapshots or reports.",
    )
    event_diff_parser.add_argument("--date", default=None)
    event_diff_parser.add_argument("--before", type=Path, required=True)
    event_diff_parser.add_argument("--after", type=Path, required=True)
    event_diff_parser.add_argument("--json", action="store_true", dest="as_json")

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

    make_private_eval_template_parser = subparsers.add_parser(
        "make-private-eval-template",
        parents=[db_parent],
        help="Generate a local private eval YAML template from aggregate DB state for one baseline date.",
    )
    make_private_eval_template_parser.add_argument("--date", required=True)
    make_private_eval_template_parser.add_argument("--output", type=Path, required=True)
    make_private_eval_template_parser.add_argument("--json", action="store_true", dest="as_json")

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


def _add_analysis_target_args(parser: argparse.ArgumentParser, *, required_type: bool = True) -> None:
    parser.add_argument(
        "--type",
        choices=["ocr", "vlm", "image_embedding", "text_embedding", "event_rebuild"],
        required=required_type,
        dest="job_type",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--from", dest="from_date", default=None)
    parser.add_argument("--to", dest="to_date", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)


def _add_analysis_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--analysis-version", default=None)
    parser.add_argument("--embedding-type", choices=["caption", "ocr", "combined_text"], default="combined_text")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--engine-unavailable-only", action="store_true")
    parser.add_argument("--version-changed-only", action="store_true")


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


def run_model_diagnostics_cli(*, config_path: Path | None, as_json: bool) -> int:
    report = run_model_diagnostics(config_path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_model_diagnostics(report))
    return 0


def run_cleanup_fake_analysis_cli(
    db_path: Path | None,
    *,
    dry_run: bool,
    yes: bool,
    include_engine_unavailable: bool,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    as_json: bool,
) -> int:
    report = cleanup_fake_analysis(
        resolve_db_path(db_path),
        dry_run=dry_run or not yes,
        yes=yes,
        include_engine_unavailable=include_engine_unavailable,
        date=date_value,
        from_date=from_date,
        to_date=to_date,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_cleanup_fake_analysis(report))
    return 0 if not report.get("error") else 1


def run_cleanup_vlm_status_cli(
    db_path: Path | None,
    *,
    statuses: list[str] | None,
    engine: str | None,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    dry_run: bool,
    yes: bool,
    as_json: bool,
) -> int:
    report = cleanup_vlm_status(
        resolve_db_path(db_path),
        statuses=statuses,
        engine=engine,
        date=date_value,
        from_date=from_date,
        to_date=to_date,
        dry_run=dry_run or not yes,
        yes=yes,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_cleanup_vlm_status(report))
    return 0 if not report.get("error") else 1


def run_analysis_plan_cli(
    db_path: Path | None,
    *,
    args,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    options = _analysis_plan_options_from_args(args, command_name="analysis-plan")
    plan = plan_analysis(repository, options).to_dict()
    print(dumps_json(plan) if args.as_json else format_analysis_plan(plan))
    return 0


def run_analysis_run_cli(
    db_path: Path | None,
    *,
    args,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    options = _analysis_run_options_from_args(args, command_name="analysis-run")
    if _is_fake_name(options.engine_name, options.model_name, options.model_path) and not args.allow_fake_write and not options.dry_run:
        print(
            "analysis-run: fake engine/model is test-only; forcing dry-run. "
            "Use --allow-fake-write only for isolated test databases.",
            file=sys.stderr,
        )
        scope = options.to_scope()
        scope["dry_run"] = True
        options = AnalysisRunOptions(**scope)

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    report = run_analysis_job(repository, options, output_dir=args.output_dir, progress_callback=progress)
    print(dumps_json(report) if args.as_json else format_analysis_run_report(report))
    return 0


def run_analysis_status_cli(
    db_path: Path | None,
    *,
    job_id: str | None,
    recent: int,
    as_json: bool,
) -> int:
    repository = AnalysisJobRepository(resolve_db_path(db_path))
    repository.initialize()
    if job_id:
        job = repository.get_job(job_id)
        if not job:
            print(f"analysis job not found: {job_id}", file=sys.stderr)
            return 1
        payload = {"job": job, "items": repository.list_items(job_id, limit=100)}
    else:
        payload = {"jobs": repository.list_jobs(recent=recent)}
    print(dumps_json(payload) if as_json else format_analysis_status(payload))
    return 0


def run_analysis_resume_cli(
    db_path: Path | None,
    *,
    args,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    try:
        report = resume_analysis_job(
            repository,
            args.job_id,
            failed_only=args.failed_only,
            engine_unavailable_only=args.engine_unavailable_only,
            limit=args.limit,
            dry_run=args.dry_run,
            save_report=args.save_report,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(dumps_json(report) if args.as_json else format_analysis_run_report(report))
    return 0


def run_analysis_retry_failed_cli(
    db_path: Path | None,
    *,
    args,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if args.job_id:
        try:
            report = resume_analysis_job(
                repository,
                args.job_id,
                failed_only=True,
                limit=args.limit,
                dry_run=args.dry_run,
                save_report=args.save_report,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        if not args.job_type:
            print("analysis-retry-failed requires --job-id or --type", file=sys.stderr)
            return 1
        options = _analysis_run_options_from_args(args, command_name="analysis-retry-failed")
        report = retry_failed_analysis(repository, options)
    print(dumps_json(report) if args.as_json else format_analysis_run_report(report))
    return 0


def run_analysis_cleanup_cli(
    db_path: Path | None,
    *,
    failed: bool,
    engine_unavailable: bool,
    old_runs: int | None,
    dry_run: bool,
    yes: bool,
    as_json: bool,
) -> int:
    repository = AnalysisJobRepository(resolve_db_path(db_path))
    repository.initialize()
    report = repository.cleanup(
        failed=failed,
        engine_unavailable=engine_unavailable,
        old_runs_days=old_runs,
        dry_run=True if not yes else dry_run,
        yes=yes,
    )
    print(dumps_json(report) if as_json else format_analysis_cleanup(report))
    return 0


def run_storage_stats_cli(db_path: Path | None, *, as_json: bool) -> int:
    report = storage_stats(resolve_db_path(db_path))
    print(dumps_json(report) if as_json else format_storage_stats(report))
    return 0


def run_db_maintenance_cli(
    db_path: Path | None,
    *,
    backup: bool,
    vacuum: bool,
    yes: bool,
    backup_dir: Path,
    as_json: bool,
) -> int:
    report = run_db_maintenance(
        resolve_db_path(db_path),
        backup=backup,
        vacuum=vacuum,
        yes=yes,
        backup_dir=backup_dir,
    )
    print(dumps_json(report) if as_json else format_db_maintenance(report))
    return 0


def run_generate_report_cli(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    public_mode: bool,
    private_mode: bool,
    eval_path: Path | None,
    eval_run: Path | None,
    output: Path | None,
    include_examples: bool,
    no_examples: bool,
    save_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    mode = "private" if private_mode else "public"
    examples = bool(include_examples) and not no_examples
    report = build_report(
        repository,
        ReportOptions(
            start_date=from_date,
            end_date=to_date,
            mode=mode,  # type: ignore[arg-type]
            eval_path=eval_path,
            eval_run=eval_run,
            include_examples=examples,
            save_json=save_json,
        ),
    )
    result = write_report(report, output_path=output, save_json=save_json, reports_dir=DEFAULT_REPORTS_DIR)
    print("Generated report:")
    print(f"- markdown: {result.markdown_path}")
    if result.json_path:
        print(f"- json: {result.json_path}")
    print(f"- mode: {mode}")
    print(f"- examples: {examples}")
    return 0


def run_month_plan_cli(
    db_path: Path | None,
    *,
    month: str,
    limit: int,
    config_path: Path | None,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    plan = month_plan(repository, month=month, limit=limit, config_path=config_path)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_month_plan(plan))
    return 0


def run_month_status_cli(
    db_path: Path | None,
    *,
    month: str,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    status = month_status(repository, month=month)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_month_status(status))
    return 0


def run_month_batch_cli(
    db_path: Path | None,
    *,
    from_month: str,
    to_month: str,
    limit: int,
    config_path: Path | None,
    dry_run: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    plan = month_batch_plan(repository, from_month=from_month, to_month=to_month, limit=limit, config_path=config_path)
    if not dry_run:
        plan["warning"] = "month-batch is planning-only in this version; run month-run one month at a time."
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_month_batch_plan(plan))
    return 0


def run_month_run_cli(
    db_path: Path | None,
    *,
    month: str,
    limit: int,
    vlm_limit: int | None,
    embedding_limit: int | None,
    config_path: Path | None,
    dry_run: bool,
    skip_vlm: bool,
    skip_embedding: bool,
    skip_rebuild: bool,
    skip_eval: bool,
    skip_report: bool,
    save_report: bool,
    yes: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    plan = month_run_plan(
        repository,
        month=month,
        limit=limit,
        vlm_limit=vlm_limit,
        embedding_limit=embedding_limit,
        config_path=config_path,
        save_report=save_report,
        skip_vlm=skip_vlm,
        skip_embedding=skip_embedding,
        skip_rebuild=skip_rebuild,
        skip_eval=skip_eval,
        skip_report=skip_report,
    )
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) if as_json else format_month_run_plan(plan, dry_run=True))
        return 0
    if not yes:
        if as_json:
            payload = dict(plan)
            payload["error"] = "month-run requires --yes for real execution"
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            print(format_month_run_plan(plan, dry_run=False))
            print("")
            print("Refusing to run without --yes. Re-run with --dry-run first, then add --yes when ready.")
        return 2

    rng = parse_month(month)
    resolved_vlm_limit = int(plan["limits"]["vlm_limit"])
    resolved_embedding_limit = int(plan["limits"]["embedding_limit"])
    step_results: list[dict[str, object]] = []

    def run_step(name: str, callback) -> int:
        print("")
        print(f"== {name} ==")
        started = time.perf_counter()
        try:
            code = int(callback())
            elapsed = time.perf_counter() - started
            step_results.append({"name": name, "status": "success" if code == 0 else "failed", "exit_code": code, "elapsed_sec": round(elapsed, 3)})
            if code:
                print(f"Step failed: {name}. Next recovery command: python -m personal_lifelog_rag.app.cli month-status --month {rng.month}")
            return code
        except Exception as exc:  # pragma: no cover - defensive safety path
            elapsed = time.perf_counter() - started
            step_results.append({"name": name, "status": "failed", "exit_code": 1, "elapsed_sec": round(elapsed, 3), "error": f"{type(exc).__name__}: {exc}"})
            print(f"Step failed: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"Next recovery command: python -m personal_lifelog_rag.app.cli month-status --month {rng.month}", file=sys.stderr)
            return 1

    code = run_step(
        "backup-db",
        lambda: run_backup_db(db_path, label=f"before_month_rollout_{rng.month.replace('-', '_')}", output_dir=DEFAULT_BACKUP_DIR),
    )
    if code:
        return code
    if not skip_vlm:
        code = run_step(
            "analyze-images",
            lambda: run_analyze_images(
                db_path,
                date_value=None,
                from_date=rng.start_date,
                to_date=rng.end_date,
                all_dates=False,
                limit=resolved_vlm_limit,
                engine_name="qwen3_vl_transformers",
                model_name=None,
                config_path=config_path,
                dry_run=False,
                force=False,
                skip_existing=True,
                only_with_ocr=False,
                only_gps=False,
                ocr_backend=None,
                vlm_backend=None,
                vlm_model=None,
                prompt_template="lifelog_structured_tags_v1",
                allow_fake_write=False,
                failed_only=False,
            ),
        )
        if code:
            return code
    if not skip_embedding:
        code = run_step(
            "build-image-embeddings",
            lambda: run_build_image_embeddings_cli(
                db_path,
                date_value=None,
                from_date=rng.start_date,
                to_date=rng.end_date,
                limit=resolved_embedding_limit,
                engine_name="qwen3_vl_embedding",
                model_name=None,
                model_path=None,
                config_path=config_path,
                dry_run=False,
                force=False,
                skip_existing=True,
                allow_fake_write=False,
            ),
        )
        if code:
            return code
        code = run_step(
            "build-text-embeddings",
            lambda: run_build_text_embeddings_cli(
                db_path,
                date_value=None,
                from_date=rng.start_date,
                to_date=rng.end_date,
                limit=resolved_embedding_limit,
                embedding_type="combined_text",
                engine_name="qwen3_vl_embedding",
                model_name=None,
                model_path=None,
                config_path=config_path,
                dry_run=False,
                force=False,
                skip_existing=True,
                allow_fake_write=False,
            ),
        )
        if code:
            return code
    if not skip_rebuild:
        code = run_step(
            "rebuild-events-with-analysis",
            lambda: run_rebuild_events_with_analysis_cli(
                db_path,
                date_value=None,
                from_date=rng.start_date,
                to_date=rng.end_date,
                dry_run=False,
                save_report=save_report,
                force=True,
                eval_path=None,
                output_dir=DEFAULT_EVENT_REBUILD_OUTPUT_DIR,
                backup_dir=DEFAULT_BACKUP_DIR,
                as_json=False,
            ),
        )
        if code:
            return code
    code = run_step("db-check", lambda: run_db_check_cli(db_path, as_json=False, strict=True))
    if code:
        return code
    eval_path = Path("private_eval") / f"questions_{rng.month.replace('-', '')}_month.yaml"
    if not skip_eval and eval_path.exists():
        code = run_step(
            "eval-private",
            lambda: run_private_eval(
                db_path,
                questions_path=eval_path,
                output_dir=DEFAULT_RUNS_DIR,
                as_json=False,
                limit=None,
                case_id=None,
                save_run=True,
                init_template=False,
                strict=False,
            ),
        )
        if code:
            return code
    elif not skip_eval:
        print("")
        print(f"== eval-private ==")
        print(f"Skipped: {eval_path} not found")
        step_results.append({"name": "eval-private", "status": "skipped", "reason": f"{eval_path} not found"})
    if not skip_report:
        code = run_step(
            "generate-report",
            lambda: run_generate_report_cli(
                db_path,
                from_date=rng.start_date,
                to_date=rng.end_date,
                public_mode=True,
                private_mode=False,
                eval_path=None,
                eval_run=None,
                output=None,
                include_examples=False,
                no_examples=True,
                save_json=save_report,
            ),
        )
        if code:
            return code

    summary = {"month": rng.month, "status": "completed", "steps": step_results}
    print("")
    print("Month run completed")
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        for step in step_results:
            print(f"- {step['name']}: {step['status']}")
    return 0


def _analysis_plan_options_from_args(args, *, command_name: str) -> AnalysisPlanOptions:
    start_date, end_date = _resolve_range_selection(
        date_value=args.date,
        from_date=args.from_date,
        to_date=args.to_date,
        all_dates=args.all,
        command_name=command_name,
        allow_all_without_dates=True,
    )
    return AnalysisPlanOptions(
        job_type=args.job_type,
        start_date=start_date,
        end_date=end_date,
        all_dates=args.all,
        limit=args.limit,
        engine_name=args.engine,
        model_name=args.model,
        model_path=args.model_path,
        prompt_version=args.prompt_version,
        analysis_version=args.analysis_version,
        embedding_type=args.embedding_type,
        force=args.force,
        skip_existing=args.skip_existing,
        failed_only=args.failed_only,
        engine_unavailable_only=args.engine_unavailable_only,
        version_changed_only=args.version_changed_only,
    )


def _analysis_run_options_from_args(args, *, command_name: str) -> AnalysisRunOptions:
    plan = _analysis_plan_options_from_args(args, command_name=command_name)
    return AnalysisRunOptions(
        **plan.to_scope(),
        dry_run=getattr(args, "dry_run", False),
        job_id=getattr(args, "job_id", None),
        save_report=getattr(args, "save_report", False),
    )


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
    default_model_config = Path("private_config/model_runtime.yaml")
    multimodal_config = {}
    if default_model_config.exists():
        config = load_model_runtime_config(default_model_config).multimodal_embedding
        multimodal_config = config.to_dict()
    result = route_query(
        repository,
        query,
        limit=limit,
        include_hidden=include_hidden,
        multimodal_config=multimodal_config,
    )
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_routed_query_result(result))
    return 0


def run_batch_qa(
    db_path: Path | None,
    queries: Sequence[str],
    *,
    limit: int,
    include_hidden: bool,
    config_path: Path | None,
    output_json: Path | None,
    output_md: Path | None,
    save_run: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    multimodal_config = _load_multimodal_runtime_config(config_path)
    multimodal_engine = _cached_batch_multimodal_engine(multimodal_config)
    started = datetime.now()
    rows: list[dict[str, object]] = []
    for query in queries:
        query_started = time.perf_counter()
        try:
            result = route_query(
                repository,
                query,
                limit=limit,
                include_hidden=include_hidden,
                multimodal_config=multimodal_config,
                multimodal_engine=multimodal_engine,
            )
            elapsed = time.perf_counter() - query_started
            answer = redact_text(result.answer, max_chars=2000)
            rows.append(
                {
                    "query": query,
                    "intent": result.intent,
                    "intent_confidence": result.intent_confidence,
                    "routing": result.routing,
                    "success": bool(result.answer.strip()),
                    "elapsed_sec": round(elapsed, 3),
                    "answer": answer,
                    "answer_summary": redact_text(result.answer, max_chars=500),
                    "result_count": len(result.results),
                    "results": result.results[:5],
                    "intent_reasons": result.intent_reasons,
                    "error_message": None,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            elapsed = time.perf_counter() - query_started
            rows.append(
                {
                    "query": query,
                    "intent": None,
                    "routing": None,
                    "success": False,
                    "elapsed_sec": round(elapsed, 3),
                    "answer": "",
                    "answer_summary": "",
                    "result_count": 0,
                    "error_message": f"{exc.__class__.__name__}: {exc}",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
    report = {
        "created_at": started.isoformat(timespec="seconds"),
        "queries": rows,
        "summary": {
            "total": len(rows),
            "success": sum(1 for row in rows if row.get("success")),
            "failed": sum(1 for row in rows if not row.get("success")),
            "elapsed_sec": round(sum(float(row.get("elapsed_sec") or 0.0) for row in rows), 3),
        },
        "model_cache": {
            "enabled": multimodal_engine is not None,
            "engine": getattr(multimodal_engine, "name", None) if multimodal_engine is not None else None,
            "model_name": getattr(multimodal_engine, "model_name", None) if multimodal_engine is not None else None,
            "strategy": "same-process cached embedding engine" if multimodal_engine is not None else "no configured embedding engine",
        },
    }
    json_path, md_path = _resolve_batch_qa_output_paths(
        output_json=output_json,
        output_md=output_md,
        save_run=save_run,
        created_at=started,
    )
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_format_batch_qa_markdown(report), encoding="utf-8")
    print(_format_batch_qa_text(report, json_path=json_path, md_path=md_path))
    return 0


def _load_multimodal_runtime_config(config_path: Path | None) -> dict[str, object]:
    path = config_path
    if path is None:
        default_model_config = Path("private_config/model_runtime.yaml")
        path = default_model_config if default_model_config.exists() else None
    if path is None or not path.exists():
        return {}
    return load_model_runtime_config(path).multimodal_embedding.to_dict()


def _cached_batch_multimodal_engine(config: dict[str, object]):
    if not any(config.get(key) for key in ("engine", "model_name", "model_path")):
        return None
    local_files_only = config.get("local_files_only")
    return get_cached_multimodal_embedding_engine(
        str(config.get("engine") or "noop"),
        model_name=str(config.get("model_name")) if config.get("model_name") else None,
        model_path=str(config.get("model_path")) if config.get("model_path") else None,
        device=str(config.get("device") or "auto"),
        dtype=str(config.get("dtype")) if config.get("dtype") is not None else None,
        local_files_only=True if local_files_only is None else bool(local_files_only),
        embedding_dim=int(config["embedding_dim"]) if config.get("embedding_dim") is not None else None,
        batch_size=int(config["batch_size"]) if config.get("batch_size") is not None else None,
    )


def _resolve_batch_qa_output_paths(
    *,
    output_json: Path | None,
    output_md: Path | None,
    save_run: bool,
    created_at: datetime,
) -> tuple[Path | None, Path | None]:
    if save_run:
        output_dir = Path("eval_outputs/batch_qa")
        stem = "batch_qa_" + created_at.strftime("%Y%m%d_%H%M%S")
        output_json = output_json or output_dir / f"{stem}.json"
        output_md = output_md or output_dir / f"{stem}.md"
    return output_json, output_md


def _format_batch_qa_text(report: dict[str, object], *, json_path: Path | None, md_path: Path | None) -> str:
    summary = report["summary"]  # type: ignore[index]
    model_cache = report.get("model_cache") or {}
    lines = [
        "Batch QA",
        "",
        f"queries: {summary['total']}",  # type: ignore[index]
        f"success: {summary['success']}",  # type: ignore[index]
        f"failed: {summary['failed']}",  # type: ignore[index]
        f"model_cache: {model_cache.get('strategy') if isinstance(model_cache, dict) else ''}",
        "",
    ]
    for index, row in enumerate(report["queries"], start=1):  # type: ignore[index]
        lines.extend(
            [
                f"{index}. {row['query']}",
                f"   intent: {row.get('intent')} routing: {row.get('routing')} elapsed={row.get('elapsed_sec')}s",
                f"   success: {row.get('success')} results={row.get('result_count')}",
                f"   answer: {row.get('answer_summary')}",
            ]
        )
        if row.get("error"):
            lines.append(f"   error: {row['error']}")
        elif row.get("error_message"):
            lines.append(f"   error: {row['error_message']}")
    if json_path is not None:
        lines.append(f"json: {json_path}")
    if md_path is not None:
        lines.append(f"markdown: {md_path}")
    return "\n".join(lines)


def _format_batch_qa_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]  # type: ignore[index]
    model_cache = report.get("model_cache") or {}
    lines = [
        "# Batch QA Run",
        "",
        f"- created_at: {report['created_at']}",
        f"- queries: {summary['total']}",  # type: ignore[index]
        f"- success: {summary['success']}",  # type: ignore[index]
        f"- failed: {summary['failed']}",  # type: ignore[index]
        f"- model_cache: {model_cache.get('strategy') if isinstance(model_cache, dict) else ''}",
        "",
        "## Results",
        "",
    ]
    for index, row in enumerate(report["queries"], start=1):  # type: ignore[index]
        lines.extend(
            [
                f"### {index}. {row['query']}",
                "",
                f"- intent: {row.get('intent')}",
                f"- routing: {row.get('routing')}",
                f"- success: {row.get('success')}",
                f"- elapsed_sec: {row.get('elapsed_sec')}",
                f"- result_count: {row.get('result_count')}",
                "",
                str(row.get("answer_summary") or ""),
                "",
            ]
        )
        if row.get("error") or row.get("error_message"):
            lines.extend([f"- error: {row.get('error') or row.get('error_message')}", ""])
    return "\n".join(lines)


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
    backend: str | None,
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
    if backend in {"embedding", "hybrid"}:
        report = multimodal_search(
            repository,
            MultimodalSearchOptions(
                query=query,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
                backend=backend,  # type: ignore[arg-type]
                include_hidden=include_hidden,
            ),
        )
        if as_json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            print(format_multimodal_search(report))
        return 0
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
    config_path: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    skip_existing: bool = False,
    only_with_ocr: bool = False,
    only_gps: bool = False,
    ocr_backend: str | None,
    vlm_backend: str | None,
    vlm_model: str | None,
    prompt_template: str | None = None,
    allow_fake_write: bool = False,
    failed_only: bool = False,
) -> int:
    config = load_model_runtime_config(config_path).vlm
    resolved_engine = engine_name or config.engine
    resolved_model_name = model_name or config.model_name
    resolved_model_path = None if model_name else config.model_path
    resolved_prompt_template = prompt_template or config.prompt_version
    guarded_dry_run = _guard_fake_write(
        command_name="analyze-images",
        engine_name=resolved_engine,
        model_name=resolved_model_name,
        dry_run=dry_run,
        allow_fake_write=allow_fake_write,
    )
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    use_vlm_table = any(
        [
            date_value,
            from_date,
            all_dates,
            resolved_engine,
            resolved_model_name,
            resolved_model_path,
            config_path,
            guarded_dry_run,
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
                engine_name=resolved_engine,
                model_name=resolved_model_path or resolved_model_name,
                dry_run=guarded_dry_run,
                force=force,
                skip_existing=skip_existing,
                only_with_ocr=only_with_ocr,
                only_gps=only_gps,
                prompt_template=resolved_prompt_template,
                failed_only=failed_only,
            ),
            engine=get_vlm_engine(
                resolved_engine,
                model_name=resolved_model_name,
                model_path=resolved_model_path,
                device=config.device,
                dtype=config.dtype,
                local_files_only=config.local_files_only,
                max_image_size=config.max_image_size,
                max_new_tokens=config.max_new_tokens,
            ),
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


def run_retry_vlm_failed_cli(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    engine_name: str | None,
    model_name: str | None,
    config_path: Path | None,
    prompt_template: str | None,
    dry_run: bool,
    allow_fake_write: bool,
    rerun_model: bool = False,
) -> int:
    start_date, end_date = _resolve_range_selection(
        date_value=date_value,
        from_date=from_date,
        to_date=to_date,
        all_dates=False,
        command_name="retry-vlm-failed",
        allow_all_without_dates=False,
    )
    config = load_model_runtime_config(config_path).vlm
    resolved_engine = engine_name or config.engine
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if dry_run:
        rows = repository.list_media_vlm(start_date=start_date, end_date=end_date, statuses=["failed"], limit=max(limit, 0))
        if resolved_engine:
            rows = [row for row in rows if str(row.get("vlm_engine") or "") == resolved_engine]
        print(
            format_recover_failed_vlm_report(
                {
                    "selected_failed_rows": len(rows),
                    "recovered": 0,
                    "unrecovered": len(rows),
                    "remaining_failed_hint": len(rows),
                    "rows": [
                        {"media_id": row.get("media_id"), "status": "pending", "reason": "dry-run"}
                        for row in rows[:20]
                    ],
                }
            )
        )
        return 0
    repair_report = recover_failed_vlm_json_rows(
        repository,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        engine=resolved_engine,
    )
    print(format_recover_failed_vlm_report(repair_report))
    if not rerun_model:
        if repair_report.get("unrecovered"):
            print("")
            print("Use --rerun-model to run Qwen3-VL again for rows that cannot be repaired from stored output.")
        return 0
    return run_analyze_images(
        db_path,
        date_value=None,
        from_date=start_date,
        to_date=end_date,
        all_dates=False,
        limit=limit,
        engine_name=engine_name,
        model_name=model_name,
        config_path=config_path,
        dry_run=dry_run,
        force=True,
        skip_existing=False,
        only_with_ocr=False,
        only_gps=False,
        ocr_backend=None,
        vlm_backend=None,
        vlm_model=None,
        prompt_template=prompt_template,
        allow_fake_write=allow_fake_write,
        failed_only=True,
    )


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
    show_errors: bool,
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
    print(format_vlm_show(rows, full=full, show_errors=show_errors))
    return 0


def run_env_check_cli(*, config_path: Path | None, as_json: bool) -> int:
    report = run_env_check(config_path)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_env_check(report))
    return 0


def run_vlm_review_queue_cli(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    status: str | None,
    unreviewed: bool,
    safety_flags: bool,
    people_present: bool,
    low_confidence: float | None,
    food_cues: bool,
    location_cues: bool,
    has_ocr: bool,
    has_embedding: bool,
    hidden: bool,
    wrong: bool,
    limit: int,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    rows = list_vlm_review_items(
        repository,
        VlmReviewFilters(
            date=date_value,
            date_from=from_date,
            date_to=to_date,
            review_status=status,
            unreviewed=unreviewed,
            safety_flags=safety_flags,
            people_present=people_present,
            low_confidence=low_confidence,
            food_cues=food_cues,
            location_cues=location_cues,
            has_ocr=has_ocr,
            has_embedding=has_embedding,
            hidden=True if hidden else None,
            wrong=True if wrong else None,
            limit=limit,
        ),
    )
    if as_json:
        print(json.dumps({"results": rows, "total": len(rows)}, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_vlm_review_queue(rows))
    return 0


def run_update_vlm_result_cli(
    db_path: Path | None,
    *,
    media_id: str,
    caption: str | None,
    short_caption: str | None,
    tags: list[str],
    scene_tags: list[str],
    object_tags: list[str],
    activity_tags: list[str],
    location_cues: list[str],
    accepted: bool,
    rejected: bool,
    wrong: bool,
    needs_fix: bool,
    verified: bool,
    hidden: bool,
    not_searchable: bool,
    not_event_usable: bool,
    note: str | None,
    clear_override: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if clear_override:
        deleted = clear_vlm_override(repository, media_id)
        payload = {"media_id": media_id, "deleted": deleted}
    else:
        review_status = _vlm_review_status_from_flags(accepted=accepted, rejected=rejected, wrong=wrong, needs_fix=needs_fix)
        payload = save_vlm_override(
            repository,
            VlmOverrideUpdate(
                media_id=media_id,
                caption_override=caption,
                short_caption_override=short_caption,
                scene_tags_override=scene_tags or None,
                object_tags_override=object_tags or None,
                activity_tags_override=activity_tags or None,
                food_cues_override=tags or None,
                location_cues_override=location_cues or None,
                is_verified=True if verified else None,
                is_hidden=True if hidden else None,
                is_wrong=True if wrong else None,
                is_searchable=False if not_searchable or rejected or wrong else None,
                is_event_usable=False if not_event_usable or rejected or wrong else None,
                review_status=review_status,
                review_note=note,
            ),
        )
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"VLM override saved: {media_id}" if not clear_override else f"VLM override cleared: {media_id} ({payload['deleted']})")
    return 0


def run_bulk_update_vlm_results_cli(
    db_path: Path | None,
    *,
    media_ids: list[str],
    from_file: Path | None,
    accepted: bool,
    rejected: bool,
    wrong: bool,
    verified: bool,
    hidden: bool,
    not_searchable: bool,
    not_event_usable: bool,
    tags: list[str],
    as_json: bool,
) -> int:
    ids = list(media_ids)
    if from_file is not None and from_file.expanduser().exists():
        ids.extend(from_file.expanduser().read_text(encoding="utf-8").splitlines())
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    review_status = _vlm_review_status_from_flags(accepted=accepted, rejected=rejected, wrong=wrong, needs_fix=False)
    payload = bulk_update_vlm_overrides(
        repository,
        ids,
        review_status=review_status,
        is_verified=True if verified else None,
        is_hidden=True if hidden else None,
        is_wrong=True if wrong else None,
        is_searchable=False if not_searchable or rejected or wrong else None,
        is_event_usable=False if not_event_usable or rejected or wrong else None,
        add_tags=tags or None,
    )
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"VLM overrides updated: {payload['updated']}")
    return 0


def run_make_vlm_eval_case_cli(*, media_id: str | None, query: str | None, expected_media_id: str | None) -> int:
    print(generate_vlm_eval_case(media_id=media_id, query=query, expected_media_id=expected_media_id))
    return 0


def _vlm_review_status_from_flags(
    *,
    accepted: bool,
    rejected: bool,
    wrong: bool,
    needs_fix: bool,
) -> str | None:
    if wrong:
        return "wrong"
    if rejected:
        return "rejected"
    if needs_fix:
        return "needs_fix"
    if accepted:
        return "accepted"
    return None


def run_image_search_cli(
    db_path: Path | None,
    *,
    query: str,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    backend: str,
    include_hidden: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    if backend != "sql":
        report = multimodal_search(
            repository,
            MultimodalSearchOptions(
                query=query,
                date_from=from_date,
                date_to=to_date,
                limit=limit,
                backend=backend,  # type: ignore[arg-type]
                include_hidden=include_hidden,
            ),
        )
        if as_json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            print(format_multimodal_search(report))
        return 0
    report = image_search(
        repository,
        ImageSearchOptions(query=query, date_from=from_date, date_to=to_date, limit=limit, include_hidden=include_hidden),
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_image_search(report))
    return 0


def run_build_image_embeddings_cli(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    engine_name: str | None,
    model_name: str | None,
    model_path: str | None,
    config_path: Path | None,
    dry_run: bool,
    force: bool,
    skip_existing: bool,
    allow_fake_write: bool = False,
) -> int:
    start_date, end_date = _resolve_range_selection(
        date_value=date_value,
        from_date=from_date,
        to_date=to_date,
        all_dates=False,
        command_name="build-image-embeddings",
        allow_all_without_dates=False,
    )
    config = load_model_runtime_config(config_path).multimodal_embedding
    resolved_engine = engine_name or config.engine
    resolved_model = model_name or config.model_name
    resolved_path = model_path or config.model_path
    guarded_dry_run = _guard_fake_write(
        command_name="build-image-embeddings",
        engine_name=resolved_engine,
        model_name=resolved_model or resolved_path,
        dry_run=dry_run,
        allow_fake_write=allow_fake_write,
    )
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    report = build_image_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            embedding_type="image",
            engine_name=resolved_engine,
            model_name=resolved_model,
            model_path=resolved_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
            dry_run=guarded_dry_run,
            force=force,
            skip_existing=skip_existing,
        ),
        engine=get_multimodal_embedding_engine(
            resolved_engine,
            model_name=resolved_model,
            model_path=resolved_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
        ),
        progress_callback=progress,
    )
    print(format_embedding_build_report(report))
    return 0


def run_build_text_embeddings_cli(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    embedding_type: str,
    engine_name: str | None,
    model_name: str | None,
    model_path: str | None,
    config_path: Path | None,
    dry_run: bool,
    force: bool,
    skip_existing: bool,
    allow_fake_write: bool = False,
) -> int:
    start_date, end_date = _resolve_range_selection(
        date_value=date_value,
        from_date=from_date,
        to_date=to_date,
        all_dates=False,
        command_name="build-text-embeddings",
        allow_all_without_dates=False,
    )
    config = load_model_runtime_config(config_path).multimodal_embedding
    resolved_engine = engine_name or config.engine
    resolved_model = model_name or config.model_name
    resolved_path = model_path or config.model_path
    guarded_dry_run = _guard_fake_write(
        command_name="build-text-embeddings",
        engine_name=resolved_engine,
        model_name=resolved_model or resolved_path,
        dry_run=dry_run,
        allow_fake_write=allow_fake_write,
    )
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    report = build_text_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            embedding_type=embedding_type,  # type: ignore[arg-type]
            engine_name=resolved_engine,
            model_name=resolved_model,
            model_path=resolved_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
            dry_run=guarded_dry_run,
            force=force,
            skip_existing=skip_existing,
        ),
        engine=get_multimodal_embedding_engine(
            resolved_engine,
            model_name=resolved_model,
            model_path=resolved_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
        ),
        progress_callback=progress,
    )
    print(format_embedding_build_report(report))
    return 0


def run_embedding_stats_cli(
    db_path: Path | None,
    *,
    from_date: str | None,
    to_date: str | None,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = embedding_stats(repository, start_date=from_date, end_date=to_date or from_date)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_embedding_stats(report))
    return 0


def _is_fake_name(*values: str | None) -> bool:
    return any("fake" in str(value or "").lower() for value in values if value is not None)


def _guard_fake_write(
    *,
    command_name: str,
    engine_name: str | None,
    model_name: str | None,
    dry_run: bool,
    allow_fake_write: bool,
) -> bool:
    if dry_run or allow_fake_write or not _is_fake_name(engine_name, model_name):
        return dry_run
    print(
        f"{command_name}: fake engine/model is test-only; forcing dry-run. "
        "Use --allow-fake-write only for isolated test databases.",
        file=sys.stderr,
    )
    return True


def run_multimodal_search_cli(
    db_path: Path | None,
    *,
    query: str,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    backend: str,
    engine_name: str | None,
    model_name: str | None,
    model_path: str | None,
    config_path: Path | None,
    include_hidden: bool,
    as_json: bool,
) -> int:
    resolved_config_path = config_path
    if resolved_config_path is None:
        default_private_config = Path("private_config/model_runtime.yaml")
        resolved_config_path = default_private_config if default_private_config.exists() else None
    config = load_model_runtime_config(resolved_config_path).multimodal_embedding
    resolved_engine = engine_name or config.engine
    resolved_model = model_name or config.model_name
    resolved_path = model_path or config.model_path
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    report = multimodal_search(
        repository,
        MultimodalSearchOptions(
            query=query,
            date_from=from_date,
            date_to=to_date,
            limit=limit,
            backend=backend,  # type: ignore[arg-type]
            engine_name=resolved_engine,
            model_name=resolved_model,
            model_path=resolved_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
            include_hidden=include_hidden,
        ),
        engine=get_multimodal_embedding_engine(
            resolved_engine,
            model_name=resolved_model,
            model_path=resolved_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
        ),
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_multimodal_search(report))
    return 0


def run_vlm_prompt(*, template: str, as_json: bool) -> int:
    try:
        prompt_template = get_vlm_prompt_template(template)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(prompt_template.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(prompt_template.prompt)
    return 0


def run_vlm_safety_check(*, text: str, as_json: bool) -> int:
    report = safety_check_text(text)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print("Safety check:")
        print(f"- violations: {', '.join(report['violations']) or 'none'}")
        print(f"- sanitized: {report['sanitized']}")
        print(f"- safety_flags: {', '.join(report['safety_flags']) or 'none'}")
        print(f"- evidence_strength: {report['evidence_strength']}")
    return 0


def run_vlm_pilot_cli(
    db_path: Path | None,
    *,
    date_value: str,
    limit: int,
    engine_name: str | None,
    model_name: str | None,
    config_path: Path | None,
    prompt_template: str | None,
    dry_run: bool,
    save_report: bool,
    force: bool,
    skip_existing: bool,
    include_hidden: bool,
    strategy: str,
    output_dir: Path,
    backup_dir: Path,
    as_json: bool,
) -> int:
    config = load_model_runtime_config(config_path)
    resolved_engine_name = engine_name or config.vlm.engine
    resolved_model_name = model_name or config.vlm.model_name
    resolved_model_path = None if model_name else config.vlm.model_path
    resolved_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_path)
    repository.initialize()

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    report = run_vlm_pilot(
        repository,
        resolved_path,
        VlmPilotOptions(
            date=date_value,
            limit=limit,
            engine_name=resolved_engine_name,
            model_name=resolved_model_path or resolved_model_name,
            prompt_template=prompt_template or config.vlm.prompt_version,
            dry_run=dry_run,
            save_report=save_report,
            force=force,
            skip_existing=skip_existing,
            include_hidden=include_hidden,
            strategy=strategy,  # type: ignore[arg-type]
            output_dir=output_dir,
            backup_dir=backup_dir,
        ),
        engine=get_vlm_engine(
            resolved_engine_name,
            model_name=resolved_model_name,
            model_path=resolved_model_path,
            device=config.vlm.device,
            dtype=config.vlm.dtype,
            local_files_only=config.vlm.local_files_only,
            max_image_size=config.vlm.max_image_size,
            max_new_tokens=config.vlm.max_new_tokens,
        ),
        progress_callback=progress,
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_vlm_pilot_report(report))
    return 0 if report.get("db_safety", {}).get("strict_ok") else 1


def run_vlm_model_info(*, config_path: Path | None, as_json: bool) -> int:
    config = load_model_runtime_config(config_path)
    report = model_info(config)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_model_info(report))
    return 0


def run_benchmark_vlm_cli(
    *,
    cases_path: Path,
    config_path: Path | None,
    engine_name: str | None,
    limit: int | None,
    save: bool,
    output_dir: Path,
    as_json: bool,
) -> int:
    config = load_model_runtime_config(config_path)
    cases = load_benchmark_cases(cases_path, limit=limit)
    engine = vlm_engine_from_spec(config.vlm, override_engine=engine_name)
    vlm_report = benchmark_vlm(cases, engine=engine)
    report = assemble_report(cases=cases, config=config, vlm_report=vlm_report, embedding_report=None)
    output_paths = write_benchmark_report(report, output_dir=output_dir) if save else None
    if as_json:
        payload = dict(report)
        if output_paths:
            payload["output_paths"] = output_paths
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_benchmark_summary(report, output_paths=output_paths))
    return 0


def run_benchmark_image_embedding_cli(
    *,
    cases_path: Path,
    config_path: Path | None,
    engine_name: str | None,
    limit: int | None,
    save: bool,
    output_dir: Path,
    as_json: bool,
) -> int:
    config = load_model_runtime_config(config_path)
    cases = load_benchmark_cases(cases_path, limit=limit)
    engine = embedding_engine_from_spec(config.multimodal_embedding, override_engine=engine_name)
    embedding_report = benchmark_image_embedding(cases, engine=engine)
    report = assemble_report(cases=cases, config=config, vlm_report=None, embedding_report=embedding_report)
    output_paths = write_benchmark_report(report, output_dir=output_dir) if save else None
    if as_json:
        payload = dict(report)
        if output_paths:
            payload["output_paths"] = output_paths
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_benchmark_summary(report, output_paths=output_paths))
    return 0


def run_benchmark_qwen_multimodal_cli(
    *,
    cases_path: Path,
    config_path: Path | None,
    engine_name: str | None,
    vlm_engine_name: str | None,
    embedding_engine_name: str | None,
    limit: int | None,
    save: bool,
    output_dir: Path,
    as_json: bool,
) -> int:
    config = load_model_runtime_config(config_path)
    if engine_name == "fake":
        config = ModelRuntimeConfig(
            vlm=config.vlm,
            multimodal_embedding=config.multimodal_embedding,
        )
    cases = load_benchmark_cases(cases_path, limit=limit)
    report = build_multimodal_benchmark_report(
        cases,
        config,
        engine_override=engine_name,
        vlm_engine_override=vlm_engine_name,
        embedding_engine_override=embedding_engine_name,
    )
    output_paths = write_benchmark_report(report, output_dir=output_dir) if save else None
    if as_json:
        payload = dict(report)
        if output_paths:
            payload["output_paths"] = output_paths
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_benchmark_summary(report, output_paths=output_paths))
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
    config_path: Path | None,
    languages: str | None,
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
    ocr_config = load_ocr_runtime_config(config_path)
    resolved_engine_name = engine_name or ocr_config.engine
    resolved_languages = languages or ocr_config.languages

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    report = run_ocr_images(
        repository,
        OcrImagesOptions(
            start_date=start_date,
            end_date=end_date,
            all_dates=all_dates,
            limit=limit,
            engine_name=resolved_engine_name,
            languages=_parse_languages(resolved_languages),
            dry_run=dry_run,
            force=force,
            skip_existing=skip_existing,
        ),
        engine=get_ocr_engine(resolved_engine_name, config=ocr_config),
        progress_callback=progress,
    )
    print(format_ocr_report(report))
    return 0


def run_ocr_diagnostics_cli(*, config_path: Path | None, as_json: bool) -> int:
    report = run_ocr_diagnostics(config_path)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_ocr_diagnostics(report))
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
    show_errors: bool,
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
    print(format_ocr_show(rows, full=full, show_errors=show_errors))
    return 0


def run_ocr_search_cli(
    db_path: Path | None,
    *,
    query: str,
    from_date: str | None,
    to_date: str | None,
    limit: int,
    include_non_success: bool,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    statuses = None if include_non_success else ["success"]
    rows = repository.list_media_ocr(
        start_date=from_date,
        end_date=to_date or from_date,
        statuses=statuses,
        keyword=query,
        limit=limit,
    )
    payload = {
        "query": query,
        "results": [
            {
                "media_id": row.get("media_id"),
                "file_name": row.get("file_name"),
                "captured_at": row.get("captured_at") or row.get("fallback_captured_at"),
                "status": row.get("status"),
                "engine": row.get("ocr_engine"),
                "text": redact_text(row.get("ocr_text_redacted") or row.get("ocr_text"), max_chars=180),
            }
            for row in rows
        ],
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    lines = [f"OCR Search: {query}", f"results: {len(rows)}"]
    for index, row in enumerate(payload["results"], start=1):
        lines.extend(
            [
                "",
                f"{index}. {row['captured_at'] or ''} media_id={row['media_id']}",
                f"   file: {row['file_name'] or ''}",
                f"   status: {row['status']} engine: {row['engine']}",
                f"   text: {row['text'] or ''}",
            ]
        )
    print("\n".join(lines))
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


def run_rebuild_events_with_analysis_cli(
    db_path: Path | None,
    *,
    date_value: str | None,
    from_date: str | None,
    to_date: str | None,
    dry_run: bool,
    save_report: bool,
    force: bool,
    eval_path: Path | None,
    output_dir: Path,
    backup_dir: Path,
    as_json: bool,
) -> int:
    if bool(date_value) == bool(from_date):
        raise ValueError("rebuild-events-with-analysis requires exactly one of --date or --from")
    resolved_db_path = resolve_db_path(db_path)
    repository = LifelogRepository(resolved_db_path)
    repository.initialize()

    def progress(message: str) -> None:
        if not dry_run:
            print(message, file=sys.stderr)

    report = rebuild_events_with_analysis(
        repository,
        resolved_db_path,
        EventRebuildOptions(
            date=date_value,
            start_date=from_date,
            end_date=to_date,
            dry_run=dry_run,
            save_report=save_report,
            force=force,
            eval_path=eval_path,
            output_dir=output_dir,
            backup_dir=backup_dir,
        ),
        config=_event_build_config(),
        progress_callback=progress,
    )
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_event_rebuild_report(report))
    return 0 if report.get("db_safety", {}).get("strict_ok") else 1


def run_event_diff_cli(
    *,
    before_path: Path,
    after_path: Path,
    as_json: bool,
) -> int:
    before = load_snapshot_or_report(before_path, slot="before_snapshot")
    after = load_snapshot_or_report(after_path, slot="after_snapshot")
    diff = diff_event_snapshots(before, after)
    if as_json:
        print(json.dumps(diff, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_event_diff(diff))
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


def run_make_private_eval_template(
    db_path: Path | None,
    *,
    date_value: str,
    output_path: Path,
    as_json: bool,
) -> int:
    repository = LifelogRepository(resolve_db_path(db_path))
    repository.initialize()
    summary = write_private_eval_template_for_date(repository, date=date_value, output_path=output_path)
    if as_json:
        payload = dict(summary.__dict__)
        payload["path"] = str(summary.path)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(format_private_eval_template_summary(summary))
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
    if args.command == "month-plan":
        try:
            return run_month_plan_cli(
                db_path,
                month=args.month,
                limit=args.limit,
                config_path=args.config,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "month-run":
        try:
            return run_month_run_cli(
                db_path,
                month=args.month,
                limit=args.limit,
                vlm_limit=args.vlm_limit,
                embedding_limit=args.embedding_limit,
                config_path=args.config,
                dry_run=args.dry_run,
                skip_vlm=args.skip_vlm,
                skip_embedding=args.skip_embedding,
                skip_rebuild=args.skip_rebuild,
                skip_eval=args.skip_eval,
                skip_report=args.skip_report,
                save_report=args.save_report,
                yes=args.yes,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "month-status":
        try:
            return run_month_status_cli(db_path, month=args.month, as_json=args.as_json)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "month-batch":
        try:
            return run_month_batch_cli(
                db_path,
                from_month=args.from_month,
                to_month=args.to_month,
                limit=args.limit,
                config_path=args.config,
                dry_run=args.dry_run,
                as_json=args.as_json,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "analysis-plan":
        return run_analysis_plan_cli(db_path, args=args)
    if args.command == "analysis-run":
        return run_analysis_run_cli(db_path, args=args)
    if args.command == "analysis-status":
        return run_analysis_status_cli(
            db_path,
            job_id=args.job_id,
            recent=args.recent,
            as_json=args.as_json,
        )
    if args.command == "analysis-resume":
        return run_analysis_resume_cli(db_path, args=args)
    if args.command == "analysis-retry-failed":
        return run_analysis_retry_failed_cli(db_path, args=args)
    if args.command == "analysis-cleanup":
        return run_analysis_cleanup_cli(
            db_path,
            failed=args.failed,
            engine_unavailable=args.engine_unavailable,
            old_runs=args.old_runs,
            dry_run=args.dry_run,
            yes=args.yes,
            as_json=args.as_json,
        )
    if args.command == "storage-stats":
        return run_storage_stats_cli(db_path, as_json=args.as_json)
    if args.command == "db-maintenance":
        return run_db_maintenance_cli(
            db_path,
            backup=args.backup,
            vacuum=args.vacuum,
            yes=args.yes,
            backup_dir=args.backup_dir,
            as_json=args.as_json,
        )
    if args.command == "model-diagnostics":
        return run_model_diagnostics_cli(config_path=args.config, as_json=args.as_json)
    if args.command == "cleanup-fake-analysis":
        return run_cleanup_fake_analysis_cli(
            db_path,
            dry_run=args.dry_run,
            yes=args.yes,
            include_engine_unavailable=args.include_engine_unavailable,
            date_value=args.date,
            from_date=args.from_date,
            to_date=args.to_date,
            as_json=args.as_json,
        )
    if args.command == "cleanup-vlm-status":
        return run_cleanup_vlm_status_cli(
            db_path,
            statuses=args.statuses,
            engine=args.engine,
            date_value=args.date,
            from_date=args.from_date,
            to_date=args.to_date,
            dry_run=args.dry_run,
            yes=args.yes,
            as_json=args.as_json,
        )
    if args.command == "env-check":
        return run_env_check_cli(config_path=args.config, as_json=args.as_json)
    if args.command == "generate-report":
        return run_generate_report_cli(
            db_path,
            from_date=args.from_date,
            to_date=args.to_date,
            public_mode=args.public_mode,
            private_mode=args.private_mode,
            eval_path=args.eval_path,
            eval_run=args.eval_run,
            output=args.output,
            include_examples=args.include_examples,
            no_examples=args.no_examples,
            save_json=args.save_json,
        )
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
    if args.command == "batch-qa":
        return run_batch_qa(
            db_path,
            args.query,
            limit=args.limit,
            include_hidden=args.include_hidden,
            config_path=args.config,
            output_json=args.output_json,
            output_md=args.output_md,
            save_run=args.save_run,
        )
    if args.command == "ui":
        return run_ui(db_path, args.host, args.port)
    if args.command == "build-embeddings":
        return run_build_embeddings(db_path, backend=args.backend, model_name=args.model)
    if args.command == "search":
        return run_search(
            db_path,
            args.query,
            backend=args.backend,
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
                config_path=args.config,
                dry_run=args.dry_run,
                force=args.force,
                skip_existing=args.skip_existing,
                only_with_ocr=args.only_with_ocr,
                only_gps=args.only_gps,
                ocr_backend=args.ocr_backend,
                vlm_backend=args.vlm_backend,
                vlm_model=args.vlm_model,
                prompt_template=args.prompt_template,
                allow_fake_write=args.allow_fake_write,
                failed_only=args.failed_only,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "retry-vlm-failed":
        try:
            return run_retry_vlm_failed_cli(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                limit=args.limit,
                engine_name=args.engine,
                model_name=args.model,
                config_path=args.config,
                prompt_template=args.prompt_template,
                dry_run=args.dry_run,
                rerun_model=args.rerun_model,
                allow_fake_write=args.allow_fake_write,
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
            show_errors=args.show_errors,
        )
    if args.command == "vlm-review-queue":
        return run_vlm_review_queue_cli(
            db_path,
            date_value=args.date,
            from_date=args.from_date,
            to_date=args.to_date,
            status=args.status,
            unreviewed=args.unreviewed,
            safety_flags=args.safety_flags,
            people_present=args.people_present,
            low_confidence=args.low_confidence,
            food_cues=args.food_cues,
            location_cues=args.location_cues,
            has_ocr=args.has_ocr,
            has_embedding=args.has_embedding,
            hidden=args.hidden,
            wrong=args.wrong,
            limit=args.limit,
            as_json=args.as_json,
        )
    if args.command == "update-vlm-result":
        return run_update_vlm_result_cli(
            db_path,
            media_id=args.media_id,
            caption=args.caption,
            short_caption=args.short_caption,
            tags=args.tag,
            scene_tags=args.scene_tag,
            object_tags=args.object_tag,
            activity_tags=args.activity_tag,
            location_cues=args.location_cue,
            accepted=args.accepted,
            rejected=args.rejected,
            wrong=args.wrong,
            needs_fix=args.needs_fix,
            verified=args.verified,
            hidden=args.hidden,
            not_searchable=args.not_searchable,
            not_event_usable=args.not_event_usable,
            note=args.note,
            clear_override=args.clear_override,
            as_json=args.as_json,
        )
    if args.command == "bulk-update-vlm-results":
        return run_bulk_update_vlm_results_cli(
            db_path,
            media_ids=args.media_id,
            from_file=args.from_file,
            accepted=args.accepted,
            rejected=args.rejected,
            wrong=args.wrong,
            verified=args.verified,
            hidden=args.hidden,
            not_searchable=args.not_searchable,
            not_event_usable=args.not_event_usable,
            tags=args.tag,
            as_json=args.as_json,
        )
    if args.command == "make-vlm-eval-case":
        return run_make_vlm_eval_case_cli(
            media_id=args.media_id,
            query=args.query,
            expected_media_id=args.expected_media_id,
        )
    if args.command == "image-search":
        return run_image_search_cli(
            db_path,
            query=args.query,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            backend=args.backend,
            include_hidden=args.include_hidden,
            as_json=args.as_json,
        )
    if args.command == "build-image-embeddings":
        return run_build_image_embeddings_cli(
            db_path,
            date_value=args.date,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            engine_name=args.engine,
            model_name=args.model,
            model_path=args.model_path,
            config_path=args.config,
            dry_run=args.dry_run,
            force=args.force,
            skip_existing=args.skip_existing,
            allow_fake_write=args.allow_fake_write,
        )
    if args.command == "build-text-embeddings":
        return run_build_text_embeddings_cli(
            db_path,
            date_value=args.date,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            embedding_type=args.type,
            engine_name=args.engine,
            model_name=args.model,
            model_path=args.model_path,
            config_path=args.config,
            dry_run=args.dry_run,
            force=args.force,
            skip_existing=args.skip_existing,
            allow_fake_write=args.allow_fake_write,
        )
    if args.command == "embedding-stats":
        return run_embedding_stats_cli(
            db_path,
            from_date=args.from_date,
            to_date=args.to_date,
            as_json=args.as_json,
        )
    if args.command == "multimodal-search":
        return run_multimodal_search_cli(
            db_path,
            query=args.query,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            backend=args.backend,
            engine_name=args.engine,
            model_name=args.model,
            model_path=args.model_path,
            config_path=args.config,
            include_hidden=args.include_hidden,
            as_json=args.as_json,
        )
    if args.command == "vlm-prompt":
        return run_vlm_prompt(template=args.template, as_json=args.as_json)
    if args.command == "vlm-safety-check":
        return run_vlm_safety_check(text=args.text, as_json=args.as_json)
    if args.command == "vlm-pilot":
        return run_vlm_pilot_cli(
            db_path,
            date_value=args.date,
            limit=args.limit,
            engine_name=args.engine,
            model_name=args.model,
            config_path=args.config,
            prompt_template=args.prompt_template,
            dry_run=args.dry_run,
            save_report=args.save_report,
            force=args.force,
            skip_existing=args.skip_existing,
            include_hidden=args.include_hidden,
            strategy=args.strategy,
            output_dir=args.output_dir,
            backup_dir=args.backup_dir,
            as_json=args.as_json,
        )
    if args.command == "vlm-model-info":
        return run_vlm_model_info(
            config_path=args.config,
            as_json=args.as_json,
        )
    if args.command == "benchmark-vlm":
        return run_benchmark_vlm_cli(
            cases_path=args.cases,
            config_path=args.config,
            engine_name=args.engine,
            limit=args.limit,
            save=args.save,
            output_dir=args.output_dir,
            as_json=args.as_json,
        )
    if args.command == "benchmark-image-embedding":
        return run_benchmark_image_embedding_cli(
            cases_path=args.cases,
            config_path=args.config,
            engine_name=args.engine,
            limit=args.limit,
            save=args.save,
            output_dir=args.output_dir,
            as_json=args.as_json,
        )
    if args.command == "benchmark-qwen-multimodal":
        return run_benchmark_qwen_multimodal_cli(
            cases_path=args.cases,
            config_path=args.config,
            engine_name=args.engine,
            vlm_engine_name=args.vlm_engine,
            embedding_engine_name=args.embedding_engine,
            limit=args.limit,
            save=args.save,
            output_dir=args.output_dir,
            as_json=args.as_json,
        )
    if args.command == "ocr-diagnostics":
        return run_ocr_diagnostics_cli(
            config_path=args.config,
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
                config_path=args.config,
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
            show_errors=args.show_errors,
        )
    if args.command == "ocr-search":
        return run_ocr_search_cli(
            db_path,
            query=args.query,
            from_date=args.from_date,
            to_date=args.to_date,
            limit=args.limit,
            include_non_success=args.include_non_success,
            as_json=args.as_json,
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
    if args.command == "rebuild-events-with-analysis":
        try:
            return run_rebuild_events_with_analysis_cli(
                db_path,
                date_value=args.date,
                from_date=args.from_date,
                to_date=args.to_date,
                dry_run=args.dry_run,
                save_report=args.save_report,
                force=args.force,
                eval_path=args.eval_path,
                output_dir=args.output_dir,
                backup_dir=args.backup_dir,
                as_json=args.as_json,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.command == "event-diff":
        return run_event_diff_cli(
            before_path=args.before,
            after_path=args.after,
            as_json=args.as_json,
        )
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
    if args.command == "make-private-eval-template":
        return run_make_private_eval_template(
            db_path,
            date_value=args.date,
            output_path=args.output,
            as_json=args.as_json,
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
