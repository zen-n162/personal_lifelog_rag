"""Local runtime health checks for model and UI execution."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
from typing import Any

from personal_lifelog_rag.benchmark.schemas import load_model_runtime_config


DEFAULT_MODEL_RUNTIME_CONFIG = Path("private_config/model_runtime.yaml")


def run_env_check(config_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect the active Python/runtime without loading model weights."""

    resolved_config = Path(config_path).expanduser() if config_path else DEFAULT_MODEL_RUNTIME_CONFIG
    config = load_model_runtime_config(resolved_config if resolved_config.exists() else None)
    executable = Path(sys.executable)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    virtual_env = os.environ.get("VIRTUAL_ENV")
    return {
        "sys_executable": str(executable),
        "conda_prefix": conda_prefix,
        "virtual_env": virtual_env,
        "using_repo_venv_python": _is_repo_venv_python(executable),
        "warnings": _warnings(executable, conda_prefix, virtual_env),
        "packages": {
            "torch": _torch_report(),
            "transformers": _package_report("transformers"),
            "pillow": _pillow_report(),
            "gradio": _package_report("gradio"),
            "sentence_transformers": _package_report("sentence-transformers", module_name="sentence_transformers"),
            "qwen_vl_utils": _package_report("qwen-vl-utils", module_name="qwen_vl_utils"),
        },
        "gradio_pillow_compatibility": _gradio_pillow_compatibility(),
        "model_paths": {
            "config_path": str(resolved_config) if resolved_config.exists() else None,
            "vlm": _model_path_report(config.vlm.model_path or config.vlm.model_name),
            "multimodal_embedding": _model_path_report(
                config.multimodal_embedding.model_path or config.multimodal_embedding.model_name
            ),
        },
    }


def format_env_check(report: dict[str, Any]) -> str:
    lines = [
        "Environment check",
        f"- sys.executable: {report['sys_executable']}",
        f"- CONDA_PREFIX: {report.get('conda_prefix') or ''}",
        f"- VIRTUAL_ENV: {report.get('virtual_env') or ''}",
        f"- using .venv Python: {report.get('using_repo_venv_python')}",
    ]
    if report.get("warnings"):
        lines.append("warnings:")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    lines.append("packages:")
    for name, payload in report.get("packages", {}).items():
        detail = f"available={payload.get('available')}"
        if payload.get("version"):
            detail += f" version={payload.get('version')}"
        if "cuda_available" in payload:
            detail += f" cuda_available={payload.get('cuda_available')}"
        if payload.get("error"):
            detail += f" error={payload.get('error')}"
        lines.append(f"- {name}: {detail}")
    compatibility = report.get("gradio_pillow_compatibility") or {}
    lines.extend(
        [
            "gradio/pillow compatibility:",
            f"- status: {compatibility.get('status')}",
            f"- detail: {compatibility.get('detail')}",
            "model paths:",
        ]
    )
    model_paths = report.get("model_paths") or {}
    lines.append(f"- config: {model_paths.get('config_path') or ''}")
    for key in ("vlm", "multimodal_embedding"):
        payload = model_paths.get(key) or {}
        lines.append(f"- {key}: {payload.get('path') or ''} exists={payload.get('exists')}")
    return "\n".join(lines)


def _is_repo_venv_python(executable: Path) -> bool:
    parts = set(executable.parts)
    return ".venv" in parts or any(part.startswith(".venv") for part in executable.parts)


def _warnings(executable: Path, conda_prefix: str | None, virtual_env: str | None) -> list[str]:
    warnings: list[str] = []
    if _is_repo_venv_python(executable):
        warnings.append("The active Python appears to be under a repo .venv; use the conda env personal_lifelog_rag.")
    if virtual_env:
        warnings.append("VIRTUAL_ENV is set; this can override the intended conda runtime.")
    if not conda_prefix:
        warnings.append("CONDA_PREFIX is not set; confirm you are running inside conda env personal_lifelog_rag.")
    return warnings


def _package_report(distribution: str, *, module_name: str | None = None) -> dict[str, Any]:
    resolved_module = module_name or distribution.replace("-", "_")
    available = importlib.util.find_spec(resolved_module) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return {"available": available, "version": version}


def _torch_report() -> dict[str, Any]:
    base = _package_report("torch")
    if not base["available"]:
        return {**base, "cuda_available": False}
    try:
        torch = importlib.import_module("torch")
        return {
            **base,
            "version": getattr(torch, "__version__", base.get("version")),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {**base, "cuda_available": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _pillow_report() -> dict[str, Any]:
    base = _package_report("Pillow", module_name="PIL")
    if not base["available"]:
        return base
    try:
        from PIL import Image

        return {**base, "version": getattr(Image, "__version__", base.get("version"))}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {**base, "error": f"{exc.__class__.__name__}: {exc}"}


def _gradio_pillow_compatibility() -> dict[str, str]:
    gradio = _package_report("gradio")
    pillow = _pillow_report()
    if not gradio.get("available"):
        return {"status": "unknown", "detail": "gradio is not installed"}
    if not pillow.get("available"):
        return {"status": "problem", "detail": "Pillow is not installed"}
    try:
        importlib.import_module("gradio")
        importlib.import_module("PIL.Image")
    except Exception as exc:
        return {"status": "problem", "detail": f"{exc.__class__.__name__}: {exc}"}
    return {
        "status": "ok",
        "detail": f"gradio {gradio.get('version') or '?'} imports with Pillow {pillow.get('version') or '?'}",
    }


def _model_path_report(raw_path: str | None) -> dict[str, Any]:
    if not raw_path:
        return {"path": None, "exists": False}
    path = Path(raw_path).expanduser()
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}
