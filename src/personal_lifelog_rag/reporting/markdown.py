"""Markdown rendering for research/portfolio reports."""

from __future__ import annotations

from typing import Any


def render_markdown_report(data: dict[str, Any]) -> str:
    lines: list[str] = ["# Personal LifeLog RAG Evaluation Report", ""]
    lines.extend(_overview(data))
    lines.extend(_architecture())
    lines.extend(_privacy(data))
    lines.extend(_dataset(data))
    lines.extend(_events(data))
    lines.extend(_search_eval(data))
    lines.extend(_multimodal(data))
    lines.extend(_examples(data))
    lines.extend(_error_analysis())
    lines.extend(_strengths())
    lines.extend(_limitations())
    lines.extend(_roadmap())
    return "\n".join(lines).rstrip() + "\n"


def _overview(data: dict[str, Any]) -> list[str]:
    options = data.get("options") or {}
    return [
        "## 1. Overview",
        "",
        "- Local-first personal lifelog RAG application.",
        "- Integrates photos, EXIF/GPS metadata, LINE history, OCR, VLM tags, events, and local multimodal search.",
        "- External APIs and cloud model calls are not required for this report.",
        f"- Report mode: `{options.get('mode', 'public')}`.",
        f"- Date range: `{options.get('start_date') or 'all'}..{options.get('end_date') or options.get('start_date') or 'all'}`.",
        "",
    ]


def _architecture() -> list[str]:
    return [
        "## 2. System Architecture",
        "",
        "- Data ingestion: local photo metadata and local LINE exports.",
        "- Storage: SQLite tables for media, messages, events, evidence, OCR, VLM, embeddings, and analysis jobs.",
        "- OCR: optional local OCR with redacted preview output.",
        "- VLM: optional local image-caption/tag extraction with safety filtering.",
        "- Multimodal embeddings: local image/text vectors for candidate retrieval.",
        "- Event generation: evidence-backed daily timeline candidates.",
        "- Search/QA: query intent routing, keyword search, image search, multimodal search, and date QA.",
        "- Human-in-the-loop: event overrides, VLM review overrides, hidden/pinned/verified controls.",
        "- Evaluation: private eval, search snapshots, analysis job reports, db-check.",
        "",
    ]


def _privacy(data: dict[str, Any]) -> list[str]:
    redaction = data.get("redaction") or {}
    options = data.get("options") or {}
    ignored_dirs = (
        "Private local data, config, model, backup, evaluation, and report folders are kept out of Git."
        if options.get("mode", "public") == "public"
        else "`data/`, `private_config/`, `private_eval/`, `eval_outputs/`, `backups/`, `models/`, and `reports/` are kept out of Git."
    )
    return [
        "## 3. Privacy and Safety Design",
        "",
        "- Local-only processing; no cloud OCR/VLM/embedding API is required.",
        f"- {ignored_dirs}",
        "- Exact GPS coordinates are hidden in reports.",
        "- File paths and media IDs are hidden or shortened.",
        "- LINE text is omitted or shortened; full message bodies are not included.",
        "- Sensitive location labels can be redacted.",
        "- VLM safety filters avoid identity, relationship, emotion, and sensitive-attribute inference.",
        "- Face recognition and person identification are outside the app scope.",
        f"- Report redaction mode: `{redaction.get('mode')}`.",
        "",
    ]


def _dataset(data: dict[str, Any]) -> list[str]:
    summary = data.get("db_summary") or {}
    db_check = data.get("db_check_summary") or {}
    return [
        "## 4. Dataset Summary",
        "",
        *_metric_lines(
            {
                "media_items": summary.get("media_items"),
                "GPS photos": summary.get("gps_photos"),
                "LINE messages": summary.get("line_messages"),
                "events": summary.get("events"),
                "event_evidence": summary.get("event_evidence"),
                "OCR analyzed": summary.get("ocr_analyzed"),
                "VLM analyzed": summary.get("vlm_analyzed"),
                "embeddings": summary.get("embeddings"),
                "call events": summary.get("call_events"),
                "db-check strict ok": db_check.get("strict_ok"),
            }
        ),
        "",
    ]


def _events(data: dict[str, Any]) -> list[str]:
    report = data.get("event_stats") or {}
    confidence = report.get("confidence") or {}
    lines = [
        "## 5. Event Generation Summary",
        "",
        *_metric_lines(
            {
                "total events": report.get("total_events"),
                "total evidence": report.get("total_event_evidence"),
                "confidence min": confidence.get("min"),
                "confidence max": confidence.get("max"),
                "confidence avg": confidence.get("avg"),
            }
        ),
        "",
        "### Monthly Event Counts",
        *_count_lines(report.get("monthly_event_counts")),
        "",
        "### Title Distribution",
        *_count_lines(report.get("title_counts"), limit=12),
        "",
        "### Confidence Buckets",
        *_count_lines(report.get("confidence_buckets")),
        "",
        "### Modality Distribution",
        *_count_lines(report.get("modality_counts")),
        "",
    ]
    return lines


def _search_eval(data: dict[str, Any]) -> list[str]:
    eval_report = data.get("private_eval")
    lines = ["## 6. Search / QA Evaluation", ""]
    if not eval_report:
        lines.extend(["- Private eval was not attached to this report.", ""])
        return lines
    summary = eval_report.get("summary") or {}
    lines.extend(
        [
            *_metric_lines(
                {
                    "cases": summary.get("cases", summary.get("total")),
                    "passed": summary.get("passed"),
                    "failed": summary.get("failed"),
                    "skipped": summary.get("skipped"),
                }
            ),
            "",
            "### By Type",
        ]
    )
    for case_type, stats in sorted((eval_report.get("by_type") or {}).items()):
        lines.append(f"- {case_type}: {stats.get('passed')}/{stats.get('total')} passed, skipped={stats.get('skipped')}")
    ranking = eval_report.get("ranking_metrics") or {}
    safety = eval_report.get("safety_metrics") or {}
    lines.extend(
        [
            "",
            "### Ranking",
            *_metric_lines(
                {
                    "top1 accuracy": ranking.get("top1_accuracy"),
                    "expected date recall@5": ranking.get("expected_date_recall_at_5"),
                }
            ),
            "",
            "### Safety",
            *_metric_lines(
                {
                    "forbidden phrase violations": safety.get("forbidden_phrase_violations"),
                    "overclaim violations": safety.get("overclaim_violations"),
                }
            ),
            "",
        ]
    )
    return lines


def _multimodal(data: dict[str, Any]) -> list[str]:
    ocr = data.get("ocr_stats") or {}
    vlm = data.get("vlm_stats") or {}
    embeddings = data.get("embedding_stats") or {}
    calls = data.get("call_stats") or {}
    places = data.get("place_stats") or {}
    jobs = data.get("analysis_jobs") or {}
    return [
        "## 7. OCR / VLM / Embedding Evaluation",
        "",
        "### OCR",
        *_count_lines(ocr.get("status_counts")),
        "",
        "### VLM",
        *_count_lines(vlm.get("status_counts")),
        "",
        "### Embeddings",
        *_metric_lines({"total": embeddings.get("total")}),
        *_count_lines(embeddings.get("by_type")),
        "",
        "### Calls / Places / Jobs",
        *_metric_lines(
            {
                "call events": calls.get("total"),
                "places assigned events": len((places.get("location_counts") or {})),
                "recent analysis jobs": jobs.get("recent_count"),
            }
        ),
        "",
    ]


def _examples(data: dict[str, Any]) -> list[str]:
    examples = data.get("examples") or []
    lines = ["## 8. Example Queries", ""]
    if not examples:
        lines.extend(["- Examples were not included in this run.", ""])
        return lines
    for example in examples:
        lines.extend(
            [
                f"- Query: `{example.get('query')}`",
                f"  - Result: {example.get('result_summary')}",
                f"  - Evidence types: {', '.join(example.get('evidence_types') or [])}",
            ]
        )
    lines.append("")
    return lines


def _error_analysis() -> list[str]:
    return [
        "## 9. Error Analysis",
        "",
        "- Overclaim risk: activity claims should remain evidence-backed and conservative.",
        "- VLM-only evidence is weak and should not decide final activity claims alone.",
        "- OCR may produce false positives, especially on signs, receipts, screenshots, or low-quality images.",
        "- Place ambiguity remains when GPS exists but the local place dictionary is incomplete.",
        "- LINE mention vs actual action remains a ranking challenge.",
        "- Low-confidence events should be reviewed or hidden through the UI.",
        "",
    ]


def _strengths() -> list[str]:
    return [
        "## 10. Strengths",
        "",
        "- Local-first and privacy-aware.",
        "- Multimodal evidence: LINE, photo metadata, GPS, OCR, VLM, embeddings, calls, places.",
        "- Evidence-based event generation and QA.",
        "- Human-in-the-loop review for events and VLM outputs.",
        "- Repeatable private eval and analysis job workflows.",
        "",
    ]


def _limitations() -> list[str]:
    return [
        "## 11. Limitations",
        "",
        "- VLM captions and tags can be wrong.",
        "- OCR can misread text and should be treated as supporting evidence.",
        "- LINE-only evidence often cannot prove whether a plan happened.",
        "- Embedding search is candidate retrieval, not a final explanation.",
        "- Local place dictionaries require manual maintenance.",
        "",
    ]


def _roadmap() -> list[str]:
    return [
        "## 12. Roadmap",
        "",
        "- Improve multimodal reranking with stronger evidence calibration.",
        "- Add event split/merge UI.",
        "- Add active-learning review loops for weak/uncertain events.",
        "- Evaluate FAISS, Chroma, or Qdrant for larger local vector indexes.",
        "- Add batch scheduling and richer job dashboards.",
        "- Add PDF/HTML report export with public/private presets.",
        "",
    ]


def _metric_lines(values: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value if value is not None else 'n/a'}" for key, value in values.items()]


def _count_lines(values: dict[str, Any] | None, *, limit: int = 20) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in list(values.items())[:limit]]
