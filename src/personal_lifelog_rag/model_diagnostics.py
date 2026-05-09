"""Local model/runtime diagnostics without loading private data or weights."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import traceback
from typing import Any

from personal_lifelog_rag.benchmark.schemas import ModelSpec, load_model_runtime_config
from personal_lifelog_rag.embeddings.engines import get_multimodal_embedding_engine
from personal_lifelog_rag.vlm.engines import get_vlm_engine


COMMON_PROCESSOR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "generation_config.json",
)


def run_model_diagnostics(config_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect configured VLM/embedding runtimes without downloading models."""

    config = load_model_runtime_config(config_path)
    vlm = _diagnose_vlm(config.vlm)
    embedding = _diagnose_embedding(config.multimodal_embedding)
    return {
        "config_path": str(Path(config_path).expanduser()) if config_path else None,
        "vlm": vlm,
        "embedding": embedding,
        "notes": [
            "No model weights are loaded by diagnostics.",
            "No downloads are attempted; local_files_only should remain true for real runs.",
        ],
    }


def format_model_diagnostics(report: dict[str, Any]) -> str:
    lines = ["Model diagnostics", f"- config: {report.get('config_path') or '(none)'}"]
    lines.extend(_format_section("VLM", report["vlm"]))
    lines.extend(_format_section("Embedding", report["embedding"]))
    lines.append("notes:")
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _diagnose_vlm(spec: ModelSpec) -> dict[str, Any]:
    engine_report = _engine_init_report(
        lambda: get_vlm_engine(
            spec.engine,
            model_name=spec.model_name,
            model_path=spec.model_path,
            device=spec.device,
            dtype=spec.dtype,
            local_files_only=spec.local_files_only,
            max_image_size=spec.max_image_size,
            max_new_tokens=spec.max_new_tokens,
        )
    )
    path = _path_report(spec)
    dependencies = {
        "transformers": _module_report("transformers"),
        "torch": _torch_report(),
        "qwen_vl_utils": _module_report("qwen_vl_utils", distribution="qwen-vl-utils"),
        "AutoModelForImageTextToText": _transformers_attr_report("AutoModelForImageTextToText"),
        "AutoProcessor": _transformers_attr_report("AutoProcessor"),
    }
    return {
        "engine": spec.engine,
        "model_name": spec.model_name,
        "model_path": spec.model_path,
        "device": spec.device,
        "dtype": spec.dtype,
        "local_files_only": spec.local_files_only,
        "prompt_version": spec.prompt_version,
        "max_image_size": spec.max_image_size,
        "max_new_tokens": spec.max_new_tokens,
        "path": path,
        "dependencies": dependencies,
        "engine_initialization": engine_report,
        "likely_unavailable_reasons": _vlm_unavailable_reasons(spec, path, dependencies, engine_report),
    }


def _diagnose_embedding(spec: ModelSpec) -> dict[str, Any]:
    engine_report = _engine_init_report(
        lambda: get_multimodal_embedding_engine(
            spec.engine,
            model_name=spec.model_name,
            model_path=spec.model_path,
            device=spec.device,
            dtype=spec.dtype,
            local_files_only=spec.local_files_only,
            embedding_dim=spec.embedding_dim,
            batch_size=spec.batch_size,
        )
    )
    adapter = _adapter_import_report(
        "personal_lifelog_rag.embeddings.engines",
        "Qwen3VlEmbeddingEngine",
        note="Local-only adapter; qwen3_vl_embedding uses official Qwen3VLEmbedder first, while sentence-transformers is explicit opt-in.",
    )
    path = _path_report(spec)
    dependencies = {
        "sentence_transformers": _module_report("sentence_transformers", distribution="sentence-transformers"),
        "transformers": _module_report("transformers"),
        "torch": _torch_report(),
        "qwen_vl_utils": _module_report("qwen_vl_utils", distribution="qwen-vl-utils"),
        "Qwen3VlEmbeddingEngine": adapter,
    }
    return {
        "engine": spec.engine,
        "model_name": spec.model_name,
        "model_path": spec.model_path,
        "device": spec.device,
        "dtype": spec.dtype,
        "local_files_only": spec.local_files_only,
        "embedding_dim": spec.embedding_dim,
        "batch_size": spec.batch_size,
        "path": path,
        "dependencies": dependencies,
        "engine_initialization": engine_report,
        "likely_unavailable_reasons": _embedding_unavailable_reasons(spec, path, dependencies, engine_report),
    }


def _vlm_unavailable_reasons(
    spec: ModelSpec,
    path: dict[str, Any],
    dependencies: dict[str, dict[str, Any]],
    engine_report: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if spec.engine and "qwen3_vl_transformers" in spec.engine and not path.get("exists"):
        reasons.append("configured model_path/model_ref does not exist locally")
    if spec.engine and ("transformers" in spec.engine or "qwen3_vl" in spec.engine):
        if not dependencies["transformers"].get("available"):
            reasons.append("transformers is not installed in the active environment")
        if not dependencies["torch"].get("available"):
            reasons.append("torch is not installed in the active environment")
        if not dependencies["qwen_vl_utils"].get("available"):
            reasons.append("qwen-vl-utils is not installed; Qwen image preprocessing may be unavailable")
    if not engine_report.get("is_available"):
        reasons.append("engine reports is_available=false")
    return reasons


def _embedding_unavailable_reasons(
    spec: ModelSpec,
    path: dict[str, Any],
    dependencies: dict[str, dict[str, Any]],
    engine_report: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if spec.engine and "qwen3_vl_embedding" in spec.engine and not path.get("exists"):
        reasons.append("configured embedding model_path/model_ref does not exist locally")
    if spec.engine and "qwen3_vl_embedding" in spec.engine:
        bundled_runtime = (
            dependencies["transformers"].get("available")
            and dependencies["torch"].get("available")
            and dependencies["qwen_vl_utils"].get("available")
        )
        if not bundled_runtime:
            reasons.append(
                "official Qwen3VLEmbedder runtime dependencies are missing; "
                "install transformers + torch + qwen-vl-utils in the active environment"
            )
        if spec.engine and "sentence_transformers" in spec.engine and not dependencies["sentence_transformers"].get("available"):
            reasons.append("sentence-transformers runtime is not installed")
        missing_bundled = [
            name
            for name in ("transformers", "torch", "qwen_vl_utils")
            if not dependencies[name].get("available")
        ]
        if missing_bundled:
            reasons.append("bundled Qwen3VLEmbedder runtime dependencies missing: " + ", ".join(missing_bundled))
    if not engine_report.get("is_available"):
        reasons.append("engine reports is_available=false")
    return reasons


def _path_report(spec: ModelSpec) -> dict[str, Any]:
    raw_ref = spec.model_path or spec.model_name
    path = Path(raw_ref).expanduser() if raw_ref else None
    exists = bool(path and path.exists())
    is_dir = bool(path and path.is_dir())
    files: dict[str, bool] = {}
    safetensors_count = 0
    if path and exists:
        base = path if path.is_dir() else path.parent
        files = {name: (base / name).exists() for name in COMMON_PROCESSOR_FILES}
        safetensors_count = len({item.resolve() for item in base.glob("**/*.safetensors")})
    return {
        "configured_ref": raw_ref,
        "exists": exists,
        "is_dir": is_dir,
        "files": files,
        "config_json_exists": bool(files.get("config.json")),
        "processor_or_tokenizer_exists": any(files.get(name) for name in COMMON_PROCESSOR_FILES if name != "config.json"),
        "safetensors_files_count": safetensors_count,
    }


def _module_report(module_name: str, *, distribution: str | None = None) -> dict[str, Any]:
    available = importlib.util.find_spec(module_name) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(distribution or module_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return {"available": available, "version": version}


def _torch_report() -> dict[str, Any]:
    base = _module_report("torch")
    if not base["available"]:
        return {**base, "cuda_available": False}
    try:
        torch = importlib.import_module("torch")
        return {
            **base,
            "version": getattr(torch, "__version__", base.get("version")),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # pragma: no cover - depends on local torch install
        return {**base, "cuda_available": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _transformers_attr_report(attr_name: str) -> dict[str, Any]:
    if importlib.util.find_spec("transformers") is None:
        return {"available": False, "error": "transformers is not installed"}
    try:
        transformers = importlib.import_module("transformers")
        return {"available": hasattr(transformers, attr_name)}
    except Exception as exc:  # pragma: no cover - import environment dependent
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _adapter_import_report(module_name: str, attr_name: str, *, note: str | None = None) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        available = hasattr(module, attr_name)
        report: dict[str, Any] = {"available": available}
        if note:
            report["note"] = note
        return report
    except Exception as exc:  # pragma: no cover - import environment dependent
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}", "note": note}


def _engine_init_report(factory) -> dict[str, Any]:
    try:
        engine = factory()
        available = bool(engine.is_available())
        report = {
            "ok": True,
            "engine_name": getattr(engine, "name", None),
            "model_name": getattr(engine, "model_name", None),
            "is_available": available,
        }
        if hasattr(engine, "model_ref"):
            report["model_ref"] = getattr(engine, "model_ref")
        return report
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {
            "ok": False,
            "is_available": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback_short": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }


def _format_section(title: str, payload: dict[str, Any]) -> list[str]:
    lines = [f"", f"{title}:"]
    for key in ("engine", "model_name", "model_path", "device", "dtype", "local_files_only"):
        lines.append(f"- {key}: {payload.get(key)}")
    path = payload.get("path") or {}
    lines.extend(
        [
            f"- model ref exists: {path.get('exists')}",
            f"- config.json exists: {path.get('config_json_exists')}",
            f"- tokenizer/processor exists: {path.get('processor_or_tokenizer_exists')}",
            f"- safetensors files: {path.get('safetensors_files_count')}",
        ]
    )
    lines.append("- dependencies:")
    for name, row in (payload.get("dependencies") or {}).items():
        detail = f"available={row.get('available')}"
        if row.get("version"):
            detail += f" version={row.get('version')}"
        if "cuda_available" in row:
            detail += f" cuda_available={row.get('cuda_available')}"
        if row.get("error"):
            detail += f" error={row.get('error')}"
        if row.get("note"):
            detail += f" note={row.get('note')}"
        lines.append(f"  - {name}: {detail}")
    init = payload.get("engine_initialization") or {}
    lines.append("- engine initialization:")
    for key in ("ok", "engine_name", "model_name", "model_ref", "is_available", "error", "traceback_short"):
        if key in init:
            lines.append(f"  - {key}: {init.get(key)}")
    if payload.get("likely_unavailable_reasons"):
        lines.append("- likely unavailable reasons:")
        for reason in payload["likely_unavailable_reasons"]:
            lines.append(f"  - {reason}")
    return lines
