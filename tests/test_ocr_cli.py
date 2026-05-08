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
    show_code = main(["--db-path", str(db_path), "ocr-show", "media_cli"])
    show_output = capsys.readouterr().out

    assert stats_code == 0
    assert payload["status_counts"]["success"] == 1
    assert show_code == 0
    assert "media_cli" in show_output
    assert "新宿" in show_output


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
