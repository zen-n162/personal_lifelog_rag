"""Local OCR engine diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
from typing import Any

from personal_lifelog_rag.ocr.config import OcrRuntimeConfig, load_ocr_runtime_config
from personal_lifelog_rag.ocr.engines import get_ocr_engine


def run_ocr_diagnostics(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_ocr_runtime_config(config_path)
    engine_name = config.engine or "tesseract_cli"
    tesseract_path = shutil.which(config.tesseract_cmd)
    version = _run_short([config.tesseract_cmd, "--version"]) if tesseract_path else None
    langs_output = _run_short([config.tesseract_cmd, "--list-langs"]) if tesseract_path else None
    langs = _parse_tesseract_langs(langs_output or "")
    engine = get_ocr_engine(engine_name, config=config)
    paddle_importable = importlib.util.find_spec("paddleocr") is not None
    return {
        "config_path": str(config_path) if config_path else None,
        "selected_engine": engine_name,
        "config": config.to_dict(),
        "local_only": config.local_only,
        "tesseract": {
            "command": config.tesseract_cmd,
            "path": tesseract_path,
            "available": bool(tesseract_path),
            "version": version,
            "languages": langs,
            "has_jpn": "jpn" in langs,
            "has_eng": "eng" in langs,
        },
        "paddleocr": {
            "importable": paddle_importable,
            "engine": "paddleocr_local",
            "availability_note": "optional skeleton; no automatic model download",
        },
        "engine_available": engine.is_available(),
        "recommendation": _recommendation(config, tesseract_path, langs, engine_name, paddle_importable),
    }


def format_ocr_diagnostics(report: dict[str, Any]) -> str:
    tesseract = report["tesseract"]
    paddle = report["paddleocr"]
    lines = [
        "OCR diagnostics",
        f"- config path: {report.get('config_path') or '(none)'}",
        f"- selected OCR engine: {report['selected_engine']}",
        f"- local-only: {report['local_only']}",
        f"- engine available: {report['engine_available']}",
        "tesseract:",
        f"- command: {tesseract['command']}",
        f"- path: {tesseract['path'] or '(not found)'}",
        f"- available: {tesseract['available']}",
        f"- has jpn: {tesseract['has_jpn']}",
        f"- has eng: {tesseract['has_eng']}",
        f"- languages: {', '.join(tesseract['languages'][:20]) if tesseract['languages'] else '(none)'}",
        "paddleocr:",
        f"- importable: {paddle['importable']}",
        f"- note: {paddle['availability_note']}",
        "recommendation:",
        f"- {report['recommendation']}",
    ]
    if tesseract.get("version"):
        lines.insert(10, f"- version: {str(tesseract['version']).splitlines()[0]}")
    return "\n".join(lines)


def _run_short(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return f"{exc.__class__.__name__}: {exc!r}"
    text = (completed.stdout or completed.stderr or "").strip()
    return text[:2000]


def _parse_tesseract_langs(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return [line for line in lines if not line.lower().startswith("list of available languages")]


def _recommendation(
    config: OcrRuntimeConfig,
    tesseract_path: str | None,
    langs: list[str],
    engine_name: str,
    paddle_importable: bool,
) -> str:
    if engine_name in {"tesseract", "tesseract_cli"}:
        if not tesseract_path:
            return "Install the local tesseract command, then rerun ocr-diagnostics. No cloud OCR is used."
        missing = [lang for lang in config.languages.split("+") if lang and lang not in langs]
        if missing:
            return f"Tesseract is installed, but missing traineddata: {', '.join(missing)}."
        return "Tesseract CLI is ready for local OCR."
    if engine_name == "paddleocr_local":
        if not paddle_importable:
            return "paddleocr is not installed; this optional engine will report engine_unavailable."
        return "paddleocr import works, but the local adapter will not download models automatically."
    if engine_name == "fake":
        return "Fake OCR is test-only; avoid writing fake results to production data."
    return "Unknown OCR engine; noop behavior is expected."
