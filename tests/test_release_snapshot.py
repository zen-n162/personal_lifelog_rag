from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_release_check_writes_sanitized_manifest(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "portfolio_public.html").write_text(
        "<!doctype html><html><body><h1>Personal Lifelog RAG</h1></body></html>",
        encoding="utf-8",
    )
    output_path = reports_dir / "release_v0_1_manifest.json"
    monkeypatch.chdir(tmp_path)

    code = main(
        [
            "--db-path",
            str(db_path),
            "release-check",
            "--version",
            "v0.1",
            "--save-manifest",
            "--output",
            str(output_path),
        ]
    )
    stdout = capsys.readouterr().out

    assert code == 0
    assert "Release check" in stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v0.1"
    assert payload["db_check"]["strict_ok"] is True
    assert payload["portfolio_html"]["privacy_check_passed"] is True
    assert "model_path" not in payload["model_config_summary"]["vlm"]
    assert "model_path" not in payload["model_config_summary"]["multimodal_embedding"]
