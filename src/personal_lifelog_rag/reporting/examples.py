"""Anonymous example generation for portfolio reports."""

from __future__ import annotations

from typing import Any

from personal_lifelog_rag.reporting.redaction import ReportRedactor


def build_example_queries(data: dict[str, Any], *, redactor: ReportRedactor) -> list[dict[str, Any]]:
    """Return safe example query summaries without raw LINE/photo content."""

    examples: list[dict[str, Any]] = []
    event_stats = data.get("event_stats") or {}
    evidence_counts = event_stats.get("evidence_type_counts") or {}
    modality_counts = event_stats.get("modality_counts") or {}
    monthly_counts = event_stats.get("monthly_event_counts") or {}
    sample_month = next(iter(monthly_counts), "YYYY-MM")
    if evidence_counts.get("line") or evidence_counts.get("photo"):
        examples.append(
            {
                "query": "YYYY-MM-DDは何していた？" if redactor.public else "2024年12月24日は何していた？",
                "result_summary": f"{sample_month}頃のイベント候補を、LINE/photo/event evidenceの件数で説明できます。",
                "evidence_types": [key for key in ("line", "photo", "ocr", "vlm") if evidence_counts.get(key)],
            }
        )
    if modality_counts.get("photo_and_line") or evidence_counts.get("photo"):
        examples.append(
            {
                "query": "PLACE_1に行ったのはいつ？",
                "result_summary": "場所候補はGPS/場所辞書/event/location_name/LINE mentionを組み合わせて候補化します。",
                "evidence_types": ["event", "photo", "line"],
            }
        )
    embedding_stats = data.get("embedding_stats") or {}
    if embedding_stats.get("total", 0):
        examples.append(
            {
                "query": "ご飯を食べた写真はいつ？",
                "result_summary": "画像embeddingは候補検索として使い、OCR/VLM/event/LINEで再ランキングします。",
                "evidence_types": ["embedding", "vlm", "ocr", "event"],
            }
        )
    call_stats = data.get("call_stats") or {}
    if call_stats.get("total", 0):
        examples.append(
            {
                "query": "通話した日はいつ？",
                "result_summary": "構造化したLINE通話ログでcompleted/missed/unansweredを分けて検索できます。",
                "evidence_types": ["line_call_events"],
            }
        )
    return examples
