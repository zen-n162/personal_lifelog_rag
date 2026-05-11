from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_ocr_cli_dry_run_and_run_with_fake_engine(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)

    dry_code = main(
        [
            "--db-path",
            str(db_path),
            "ocr-images",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "5",
            "--dry-run",
        ]
    )
    dry_output = capsys.readouterr().out

    run_code = main(
        [
            "--db-path",
            str(db_path),
            "ocr-images",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "5",
        ]
    )
    run_output = capsys.readouterr().out

    assert dry_code == 0
    assert "- dry_run: True" in dry_output
    assert run_code == 0
    assert "- success: 1" in run_output


def test_ocr_stats_json_and_show(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    main(["--db-path", str(db_path), "ocr-images", "--date", "2024-12-24", "--engine", "fake"])
    capsys.readouterr()

    stats_code = main(["--db-path", str(db_path), "ocr-stats", "--json"])
    payload = json.loads(capsys.readouterr().out)
    show_code = main(["--db-path", str(db_path), "ocr-show", "media_cli", "--show-errors"])
    show_output = capsys.readouterr().out

    assert stats_code == 0
    assert payload["status_counts"]["success"] == 1
    assert show_code == 0
    assert "media_cli" in show_output
    assert "新宿" in show_output
    assert "error_message:" in show_output


def test_ocr_diagnostics_and_search_cli(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        """
ocr:
  engine: "tesseract_cli"
  languages: "jpn+eng"
  tesseract_cmd: "definitely_missing_tesseract_for_test"
  local_only: true
""",
        encoding="utf-8",
    )
    main(["--db-path", str(db_path), "ocr-images", "--date", "2024-12-24", "--engine", "fake"])
    capsys.readouterr()

    diag_code = main(["ocr-diagnostics", "--config", str(config_path), "--json"])
    diag_payload = json.loads(capsys.readouterr().out)
    search_code = main(["--db-path", str(db_path), "ocr-search", "新宿", "--json"])
    search_payload = json.loads(capsys.readouterr().out)

    assert diag_code == 0
    assert diag_payload["selected_engine"] == "tesseract_cli"
    assert diag_payload["local_only"] is True
    assert search_code == 0
    assert search_payload["results"][0]["media_id"] == "media_cli"


def test_ocr_images_text_cues_only_cli(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_cli",
        caption="Screen with a visible menu label",
        text_cues=["menu_label"],
        contains_text_hint=True,
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )

    code = main(
        [
            "--db-path",
            str(db_path),
            "ocr-images",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--text-cues-only",
            "--limit",
            "5",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "- selected images: 1" in output


def test_ocr_priority_cli_reports_reasons(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_cli",
        caption="A document label with visible text",
        text_cues=["document_text"],
        contains_text_hint=True,
        confidence=0.8,
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )

    code = main(
        [
            "--db-path",
            str(db_path),
            "ocr-priority",
            "--from",
            "2024-12-24",
            "--to",
            "2024-12-24",
            "--limit",
            "5",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["total"] == 1
    assert payload["results"][0]["media_id"] == "media_cli"
    assert "text_cues" in payload["results"][0]["priority_reason"]


def test_retry_ocr_failed_dry_run_lists_rows(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    repository = LifelogRepository(db_path)
    repository.upsert_media_ocr(media_id="media_cli", status="engine_unavailable", ocr_engine="tesseract_cli")

    code = main(
        [
            "--db-path",
            str(db_path),
            "retry-ocr-failed",
            "--date",
            "2024-12-24",
            "--limit",
            "5",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Retry OCR failed dry-run" in output
    assert "media_cli" in output


def _seed_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "cli.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    repository.add_media_item(
        id="media_cli",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-cli",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return db_path
