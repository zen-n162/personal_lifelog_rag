"""Report assembly and persistence for local multimodal benchmarks."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.benchmark.embedding_benchmark import (
    benchmark_image_embedding,
    engine_from_spec as embedding_engine_from_spec,
)
from personal_lifelog_rag.benchmark.schemas import (
    BenchmarkCase,
    ModelRuntimeConfig,
)
from personal_lifelog_rag.benchmark.vlm_benchmark import (
    benchmark_vlm,
    engine_from_spec as vlm_engine_from_spec,
    qwen_vlm_availability,
)


DEFAULT_BENCHMARK_OUTPUT_DIR = Path("eval_outputs/vlm_benchmark")


def model_info(config: ModelRuntimeConfig) -> dict[str, Any]:
    embedding_spec = config.multimodal_embedding
    embedding_path_exists = bool(
        embedding_spec.model_path and Path(embedding_spec.model_path).expanduser().exists()
    )
    return {
        "local_only": True,
        "network": "disabled",
        "auto_download": False,
        "vlm": qwen_vlm_availability(config.vlm),
        "multimodal_embedding": {
            "engine": embedding_spec.engine,
            "provider": embedding_spec.provider,
            "model_name": embedding_spec.model_name,
            "model_path": embedding_spec.model_path,
            "device": embedding_spec.device,
            "embedding_dim": embedding_spec.embedding_dim,
            "batch_size": embedding_spec.batch_size,
            "configured": bool(embedding_spec.engine or embedding_spec.model_name or embedding_spec.model_path),
            "model_path_exists": embedding_path_exists,
            "available": bool(embedding_spec.engine == "fake" or embedding_path_exists),
            "note": "Qwen3-VL-Embedding is for retrieval/indexing, not caption generation.",
        },
    }


def build_multimodal_benchmark_report(
    cases: list[BenchmarkCase],
    config: ModelRuntimeConfig,
    *,
    engine_override: str | None = None,
    vlm_engine_override: str | None = None,
    embedding_engine_override: str | None = None,
) -> dict[str, Any]:
    vlm_engine_name = vlm_engine_override or engine_override
    embedding_engine_name = embedding_engine_override or engine_override
    vlm_engine = vlm_engine_from_spec(config.vlm, override_engine=vlm_engine_name)
    embedding_engine = embedding_engine_from_spec(
        config.multimodal_embedding,
        override_engine=embedding_engine_name,
    )
    vlm_report = benchmark_vlm(cases, engine=vlm_engine)
    embedding_report = benchmark_image_embedding(cases, engine=embedding_engine)
    return assemble_report(
        cases=cases,
        config=config,
        vlm_report=vlm_report,
        embedding_report=embedding_report,
    )


def assemble_report(
    *,
    cases: list[BenchmarkCase],
    config: ModelRuntimeConfig,
    vlm_report: dict[str, Any] | None = None,
    embedding_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = "vlm_benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = _summary(cases, vlm_report, embedding_report)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "local_only": True,
        "auto_download": False,
        "model_config": config.to_dict(),
        "summary": summary,
        "cases": [case.to_dict() for case in cases],
        "vlm": vlm_report,
        "embedding": embedding_report,
        "recommendation": [
            "Qwen3-VL should be used for caption/tags/event cues.",
            "Qwen3-VL-Embedding should be used for text-to-image retrieval and indexing.",
            "VLM-only evidence should remain conservative and be reranked with OCR, LINE, GPS, and places.",
        ],
    }


def write_benchmark_report(report: dict[str, Any], output_dir: str | Path = DEFAULT_BENCHMARK_OUTPUT_DIR) -> dict[str, str]:
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / str(report["run_id"])
    json_path = _unique_path(base.with_suffix(".json"))
    md_path = json_path.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    md_path.write_text(format_benchmark_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def format_model_info(report: dict[str, Any]) -> str:
    lines = [
        "VLM Model Info",
        "- local_only: true",
        "- auto_download: false",
        "",
        "Qwen3-VL / caption model:",
    ]
    vlm = report["vlm"]
    lines.extend(
        [
            f"- engine: {vlm.get('engine') or ''}",
            f"- model_name: {vlm.get('model_name') or ''}",
            f"- model_path: {vlm.get('model_path') or ''}",
            f"- device: {vlm.get('device') or 'auto'}",
            f"- dtype: {vlm.get('dtype') or ''}",
            f"- local_files_only: {vlm.get('local_files_only')}",
            f"- available: {vlm.get('available')}",
            f"- note: {vlm.get('note')}",
            "",
            "Qwen3-VL-Embedding / retrieval model:",
        ]
    )
    emb = report["multimodal_embedding"]
    lines.extend(
        [
            f"- engine: {emb.get('engine') or ''}",
            f"- model_name: {emb.get('model_name') or ''}",
            f"- model_path: {emb.get('model_path') or ''}",
            f"- device: {emb.get('device') or 'auto'}",
            f"- available: {emb.get('available')}",
            f"- note: {emb.get('note')}",
        ]
    )
    return "\n".join(lines)


def format_benchmark_summary(report: dict[str, Any], *, output_paths: dict[str, str] | None = None) -> str:
    summary = report["summary"]
    lines = [
        "Qwen Multimodal Benchmark",
        "",
        "Summary:",
        f"- cases: {summary['cases']}",
        f"- VLM success rate: {summary.get('vlm_success_rate')}",
        f"- Embedding success rate: {summary.get('embedding_success_rate')}",
        f"- avg VLM latency: {summary.get('avg_vlm_latency_sec')}",
        f"- avg embedding latency: {summary.get('avg_embedding_latency_sec')}",
        f"- safety violations: {summary.get('safety_violations')}",
        f"- top1 accuracy: {summary.get('top1_accuracy')}",
        f"- recall@5: {summary.get('recall_at_5')}",
    ]
    if output_paths:
        lines.extend(["", "Saved:", f"- json: {output_paths['json']}", f"- markdown: {output_paths['markdown']}"])
    return "\n".join(lines)


def format_benchmark_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Qwen Multimodal Benchmark",
        "",
        "## Summary",
        "",
        f"- cases: {summary['cases']}",
        f"- VLM success rate: {summary.get('vlm_success_rate')}",
        f"- Embedding success rate: {summary.get('embedding_success_rate')}",
        f"- avg VLM latency: {summary.get('avg_vlm_latency_sec')}",
        f"- avg embedding latency: {summary.get('avg_embedding_latency_sec')}",
        f"- safety violations: {summary.get('safety_violations')}",
        f"- top1 accuracy: {summary.get('top1_accuracy')}",
        f"- recall@5: {summary.get('recall_at_5')}",
        "",
        "## Per-Case Results",
        "",
    ]
    vlm_cases = {row["case_id"]: row for row in (report.get("vlm") or {}).get("case_results", [])}
    embedding_cases = {row["case_id"]: row for row in (report.get("embedding") or {}).get("case_results", [])}
    for case in report.get("cases", []):
        case_id = case["id"]
        vlm = (vlm_cases.get(case_id) or {}).get("vlm", {})
        embedding = embedding_cases.get(case_id) or {}
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- image: `{case.get('image_path')}`",
                f"- VLM status: {vlm.get('status')}",
                f"- VLM matched tags: {', '.join(vlm.get('tags_matched') or []) or 'none'}",
                f"- forbidden terms: {', '.join(vlm.get('forbidden_terms_found') or []) or 'none'}",
                f"- embedding status: {embedding.get('status')}",
            ]
        )
        for query in embedding.get("query_results", [])[:5]:
            lines.append(
                f"  - query `{query.get('query')}` target_rank={query.get('target_rank')} "
                f"score={query.get('target_score')}"
            )
        lines.append("")
    lines.extend(
        [
            "## Recommendation",
            "",
            "- Qwen3-VL should be used for caption/tags/event cues.",
            "- Qwen3-VL-Embedding should be used for retrieval/indexing.",
            "- VLM-only evidence should remain conservative.",
        ]
    )
    return "\n".join(lines)


def _summary(
    cases: list[BenchmarkCase],
    vlm_report: dict[str, Any] | None,
    embedding_report: dict[str, Any] | None,
) -> dict[str, Any]:
    embedding_metrics = (embedding_report or {}).get("metrics") or {}
    return {
        "cases": len(cases),
        "vlm_success_rate": (vlm_report or {}).get("success_rate"),
        "embedding_success_rate": _embedding_success_rate(embedding_report),
        "avg_vlm_latency_sec": _average_case_latency((vlm_report or {}).get("case_results", []), "vlm"),
        "avg_embedding_latency_sec": (embedding_report or {}).get("latency_sec"),
        "safety_violations": (vlm_report or {}).get("safety_violations", 0),
        "top1_accuracy": embedding_metrics.get("top1_accuracy"),
        "recall_at_5": embedding_metrics.get("recall_at_5"),
    }


def _embedding_success_rate(report: dict[str, Any] | None) -> float | None:
    if not report or not report.get("case_results"):
        return None
    rows = report["case_results"]
    success = sum(1 for row in rows if row.get("status") == "success")
    return round(success / len(rows), 4) if rows else None


def _average_case_latency(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row.get(key, {}).get("latency_sec") or 0.0) for row in rows]
    return round(sum(values) / len(values), 4) if values else None


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique benchmark output path for {path}")
