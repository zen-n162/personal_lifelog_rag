"""Local-only VLM benchmark runner."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from personal_lifelog_rag.benchmark.schemas import BenchmarkCase, ModelSpec
from personal_lifelog_rag.vlm.base import VlmEngine
from personal_lifelog_rag.vlm.engines import get_vlm_engine
from personal_lifelog_rag.vlm.prompts import SAFE_IMAGE_ANALYSIS_PROMPT
from personal_lifelog_rag.vlm.safety import sanitize_vlm_result
from personal_lifelog_rag.vlm.schemas import VlmResult


def benchmark_vlm(cases: list[BenchmarkCase], *, engine: VlmEngine) -> dict[str, Any]:
    started = time.perf_counter()
    case_results = [_benchmark_one(case, engine=engine) for case in cases]
    success_count = sum(1 for row in case_results if row["vlm"]["status"] == "success")
    safety_violations = sum(len(row["vlm"]["forbidden_terms_found"]) for row in case_results)
    matched = sum(1 for row in case_results if row["vlm"]["tags_matched"])
    return {
        "engine": engine.name,
        "model_name": getattr(engine, "model_name", None),
        "status": "success" if success_count or not cases else _overall_status(case_results),
        "latency_sec": round(time.perf_counter() - started, 4),
        "cases": len(cases),
        "success_count": success_count,
        "success_rate": round(success_count / len(cases), 4) if cases else None,
        "tag_match_rate": round(matched / len(cases), 4) if cases else None,
        "safety_violations": safety_violations,
        "case_results": case_results,
    }


def engine_from_spec(spec: ModelSpec, *, override_engine: str | None = None) -> VlmEngine:
    engine_name = override_engine or spec.engine
    return get_vlm_engine(
        engine_name,
        model_name=spec.model_name,
        model_path=spec.model_path,
        device=spec.device,
        dtype=spec.dtype,
        local_files_only=spec.local_files_only,
        max_image_size=spec.max_image_size,
        max_new_tokens=spec.max_new_tokens,
    )


def qwen_vlm_availability(spec: ModelSpec) -> dict[str, Any]:
    """Return a conservative availability summary without loading a model."""

    model_path_exists = bool(spec.model_path and Path(spec.model_path).expanduser().exists())
    configured = bool(spec.engine or spec.model_name or spec.model_path)
    return {
        "engine": spec.engine,
        "provider": spec.provider,
        "model_name": spec.model_name,
        "model_path": spec.model_path,
        "device": spec.device,
        "dtype": spec.dtype,
        "local_files_only": spec.local_files_only,
        "configured": configured,
        "model_path_exists": model_path_exists,
        "available": bool(configured and (spec.engine == "fake" or model_path_exists)),
        "note": "Models are never downloaded automatically; provide a local model_path for real runs.",
    }


def _benchmark_one(case: BenchmarkCase, *, engine: VlmEngine) -> dict[str, Any]:
    started = time.perf_counter()
    if not engine.is_available():
        result = VlmResult(
            engine=engine.name,
            model_name=getattr(engine, "model_name", None),
            status="engine_unavailable",
            error_message=f"VLM engine '{engine.name}' is not available",
        )
    elif not case.image_path.exists() and engine.name != "fake":
        result = VlmResult(
            engine=engine.name,
            model_name=getattr(engine, "model_name", None),
            status="failed",
            error_message="image file does not exist",
        )
    else:
        try:
            result = sanitize_vlm_result(engine.analyze_image(case.image_path, SAFE_IMAGE_ANALYSIS_PROMPT))
        except Exception as exc:  # pragma: no cover - real engine boundary
            result = VlmResult(
                engine=engine.name,
                model_name=getattr(engine, "model_name", None),
                status="failed",
                error_message=f"VLM benchmark failed with {exc.__class__.__name__}",
            )
    tags = _all_tags(result)
    caption_blob = "\n".join(
        [
            result.caption or "",
            result.short_caption or "",
            "\n".join(tags),
        ]
    )
    forbidden_found = [term for term in case.forbidden_terms if term and term in caption_blob]
    tags_matched = [tag for tag in case.expected_tags_any if _tag_matches(tag, tags)]
    schema_valid = _schema_valid(result)
    return {
        "case_id": case.id,
        "image_path": str(case.image_path),
        "query_texts": list(case.query_texts),
        "vlm": {
            "engine": result.engine,
            "model_name": result.model_name,
            "status": result.status,
            "latency_sec": round(time.perf_counter() - started, 4),
            "caption": result.caption,
            "short_caption": result.short_caption,
            "scene_tags": result.scene_tags,
            "object_tags": result.object_tags,
            "activity_tags": result.activity_tags,
            "food_cues": result.food_cues,
            "location_cues": result.location_cues,
            "people_count": result.people_count,
            "contains_text_hint": result.contains_text_hint,
            "confidence": result.confidence,
            "safety_flags": result.safety_flags,
            "forbidden_terms_found": forbidden_found,
            "tags_matched": tags_matched,
            "schema_valid": schema_valid,
            "error_message": result.error_message,
        },
    }


def _overall_status(case_results: list[dict[str, Any]]) -> str:
    statuses = {str(row["vlm"]["status"]) for row in case_results}
    if statuses == {"engine_unavailable"}:
        return "engine_unavailable"
    if statuses == {"failed"}:
        return "failed"
    return "partial"


def _all_tags(result: VlmResult) -> list[str]:
    tags = result.scene_tags + result.object_tags + result.activity_tags + result.food_cues + result.location_cues
    return [str(tag) for tag in tags]


def _tag_matches(expected: str, tags: list[str]) -> bool:
    expected_lower = expected.lower()
    return any(expected_lower == tag.lower() or expected_lower in tag.lower() for tag in tags)


def _schema_valid(result: VlmResult) -> bool:
    return (
        isinstance(result.scene_tags, list)
        and isinstance(result.object_tags, list)
        and isinstance(result.activity_tags, list)
        and isinstance(result.food_cues, list)
        and isinstance(result.location_cues, list)
        and isinstance(result.safety_flags, list)
        and result.status in {"pending", "success", "skipped", "failed", "no_visual_content", "engine_unavailable"}
    )
