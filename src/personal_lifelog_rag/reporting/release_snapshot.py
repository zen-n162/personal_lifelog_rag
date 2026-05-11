"""Release freeze manifest helpers for reproducible local snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from personal_lifelog_rag.benchmark.schemas import load_model_runtime_config
from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import (
    evaluate_private_questions,
    load_private_eval_questions,
    write_private_eval_report,
)
from personal_lifelog_rag.reporting.portfolio_html import check_public_portfolio_path


DEFAULT_RELEASE_MANIFEST = Path("reports/release_v0_1_manifest.json")


def build_release_manifest(
    repository: LifelogRepository,
    *,
    version: str,
    eval_path: Path | None = None,
    save_manifest: bool = False,
    output: Path = DEFAULT_RELEASE_MANIFEST,
    run_pytest: bool = False,
    portfolio_html: Path = Path("reports/portfolio_public.html"),
    model_config: Path | None = Path("private_config/model_runtime.yaml"),
) -> dict[str, Any]:
    """Collect a public-safe release manifest.

    The manifest intentionally stores model names but not local model paths.
    """

    repository.initialize()
    db_check = run_db_check(repository.db_path)
    eval_summary: dict[str, Any] | None = None
    eval_run_path: str | None = None
    if eval_path and eval_path.exists():
        eval_report = evaluate_private_questions(repository, load_private_eval_questions(eval_path))
        eval_summary = eval_report.get("summary")
        eval_run_path = str(write_private_eval_report(eval_report, Path("private_eval/runs")))
    pytest_summary = _run_pytest_summary() if run_pytest else {"run": False}
    privacy_report = check_public_portfolio_path(portfolio_html) if portfolio_html.exists() else {
        "passed": False,
        "issue_count": None,
        "issues": [{"pattern": "missing_portfolio_html", "file": str(portfolio_html), "line": 0, "snippet": "portfolio HTML not found"}],
    }
    manifest = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pytest": pytest_summary,
        "db_check": {
            "strict_ok": bool(db_check.get("strict", {}).get("ok")),
            "severe_issue_count": len(db_check.get("strict", {}).get("severe_issues", [])),
        },
        "private_eval": {
            "path": str(eval_path) if eval_path else None,
            "run_path": eval_run_path,
            "summary": eval_summary,
        },
        "latest_reports": _latest_reports(),
        "portfolio_html": {
            "path": str(portfolio_html),
            "exists": portfolio_html.exists(),
            "privacy_check_passed": bool(privacy_report.get("passed")),
            "privacy_issue_count": privacy_report.get("issue_count"),
        },
        "model_config_summary": _model_config_summary(model_config),
        "counts": _counts(repository, db_check),
        "privacy_check": privacy_report,
    }
    if save_manifest:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def format_release_manifest(manifest: dict[str, Any], *, output: Path | None = None) -> str:
    lines = [
        f"Release check: {manifest['version']}",
        f"- generated_at: {manifest['generated_at']}",
        f"- git_commit: {manifest.get('git_commit') or 'unknown'}",
        f"- db-check strict ok: {manifest['db_check']['strict_ok']}",
        f"- portfolio privacy ok: {manifest['portfolio_html']['privacy_check_passed']}",
    ]
    eval_summary = manifest.get("private_eval", {}).get("summary")
    if eval_summary:
        lines.append(
            f"- private eval: passed={eval_summary.get('passed')} failed={eval_summary.get('failed')} skipped={eval_summary.get('skipped')}"
        )
    pytest_summary = manifest.get("pytest") or {}
    if pytest_summary.get("run"):
        lines.append(f"- pytest: exit_code={pytest_summary.get('exit_code')} summary={pytest_summary.get('summary')}")
    counts = manifest.get("counts") or {}
    lines.append(
        "- counts: "
        + ", ".join(
            f"{key}={value}"
            for key, value in counts.items()
            if key
            in {
                "media_items",
                "media_vlm_success",
                "media_vlm_failed",
                "media_embeddings_success",
                "media_ocr_success",
                "events",
                "event_evidence",
            }
        )
    )
    if output:
        lines.append(f"- manifest: {output}")
    return "\n".join(lines)


def _run_pytest_summary() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tail = "\n".join(result.stdout.splitlines()[-8:])
    summary = ""
    for line in reversed(result.stdout.splitlines()):
        if " passed" in line or " failed" in line:
            summary = line.strip()
            break
    return {"run": True, "exit_code": result.returncode, "summary": summary, "output_tail": tail}


def _git_commit() -> str | None:
    commands = [
        ["git", "--git-dir=.git-local", "--work-tree=.", "rev-parse", "HEAD"],
        ["git", "rev-parse", "HEAD"],
    ]
    for command in commands:
        try:
            result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _latest_reports() -> dict[str, str | None]:
    reports = sorted(Path("reports").glob("lifelog_rag_eval_*.md"))
    json_reports = sorted(Path("reports").glob("lifelog_rag_eval_*.json"))
    return {
        "latest_markdown": str(reports[-1]) if reports else None,
        "latest_json": str(json_reports[-1]) if json_reports else None,
        "portfolio_html": "reports/portfolio_public.html" if Path("reports/portfolio_public.html").exists() else None,
    }


def _model_config_summary(path: Path | None) -> dict[str, Any]:
    config = load_model_runtime_config(path)
    return {
        "config_present": bool(path and path.exists()),
        "vlm": {
            "engine": config.vlm.engine,
            "model_name": config.vlm.model_name,
            "device": config.vlm.device,
            "dtype": config.vlm.dtype,
            "local_files_only": config.vlm.local_files_only,
        },
        "multimodal_embedding": {
            "engine": config.multimodal_embedding.engine,
            "model_name": config.multimodal_embedding.model_name,
            "device": config.multimodal_embedding.device,
            "dtype": config.multimodal_embedding.dtype,
            "local_files_only": config.multimodal_embedding.local_files_only,
            "embedding_dim": config.multimodal_embedding.embedding_dim,
        },
    }


def _counts(repository: LifelogRepository, db_check: dict[str, Any]) -> dict[str, Any]:
    stats = repository.stats()
    media_vlm = db_check.get("media_vlm", {})
    media_ocr = db_check.get("media_ocr", {})
    media_embeddings = db_check.get("media_embeddings", {})
    vlm_status_counts = _status_count_map(media_vlm.get("status_counts") or {})
    ocr_status_counts = _status_count_map(media_ocr.get("status_counts") or {})
    embedding_status_counts = _status_count_map(
        media_embeddings.get("by_status") or media_embeddings.get("status_counts") or {}
    )
    return {
        "media_items": stats.get("media_items", 0),
        "media_vlm_success": media_vlm.get("success_count", media_vlm.get("success", vlm_status_counts.get("success", 0))),
        "media_vlm_failed": media_vlm.get("failed_count", media_vlm.get("failed", vlm_status_counts.get("failed", 0))),
        "media_embeddings_success": embedding_status_counts.get("success", 0),
        "media_embeddings_failed": embedding_status_counts.get("failed", 0),
        "media_ocr_success": media_ocr.get("success_count", media_ocr.get("success", ocr_status_counts.get("success", 0))),
        "media_ocr_failed": media_ocr.get("failed_count", media_ocr.get("failed", ocr_status_counts.get("failed", 0))),
        "events": stats.get("events", 0),
        "event_evidence": stats.get("event_evidence", 0),
    }


def _status_count_map(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(count or 0) for key, count in value.items()}
    if isinstance(value, list):
        output: dict[str, int] = {}
        for row in value:
            if isinstance(row, dict):
                key = str(row.get("status") or row.get("key") or row.get("name") or "unknown")
                output[key] = int(row.get("count") or 0)
        return output
    return {}
