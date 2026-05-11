from __future__ import annotations

import csv
import json
from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_missing_files_lists_exports_and_marks_unavailable(tmp_path: Path, capsys) -> None:
    db_path = _seed_missing_media_db(tmp_path)
    export_path = tmp_path / "reports" / "missing_files.csv"

    list_code = main(["--db-path", str(db_path), "missing-files", "--limit", "50"])
    list_output = capsys.readouterr().out
    export_code = main(["--db-path", str(db_path), "missing-files", "--export", str(export_path), "--json"])
    export_payload = json.loads(capsys.readouterr().out)
    dry_mark_code = main(["--db-path", str(db_path), "missing-files", "--mark-unavailable"])
    capsys.readouterr()
    mark_code = main(["--db-path", str(db_path), "missing-files", "--mark-unavailable", "--yes", "--json"])
    mark_payload = json.loads(capsys.readouterr().out)

    assert list_code == 0
    assert "- missing original files: 1" in list_output
    assert "- missing thumbnails: 1" in list_output
    assert export_code == 0
    assert export_payload["exported_path"] == str(export_path)
    with export_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["media_id"] for row in rows} == {"media_missing_file", "media_missing_thumb"}
    assert dry_mark_code == 0
    assert mark_code == 0
    assert mark_payload["marked_unavailable"] == 1

    repository = LifelogRepository(db_path)
    media = repository.list_media_items(limit=10)
    analysis = {
        row["id"]: json.loads(row["analysis_json"])
        for row in media
        if row.get("analysis_json")
    }
    assert analysis["media_missing_file"]["file_unavailable"] is True


def test_db_check_missing_files_are_optional_strict_failure(tmp_path: Path, capsys) -> None:
    db_path = _seed_missing_media_db(tmp_path)

    report = run_db_check(db_path)
    strict_ok_code = main(["--db-path", str(db_path), "db-check", "--strict"])
    capsys.readouterr()
    strict_fail_code = main(["--db-path", str(db_path), "db-check", "--strict", "--fail-on-missing-files"])
    strict_fail_output = capsys.readouterr().out

    assert report["media_items"]["missing_file_count"] == 1
    assert report["strict"]["ok"] is True
    assert strict_ok_code == 0
    assert strict_fail_code == 1
    assert "missing original media files" in strict_fail_output


def _seed_missing_media_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    existing_image = tmp_path / "existing.jpg"
    existing_image_2 = tmp_path / "existing_2.jpg"
    existing_thumb = tmp_path / "existing_thumb.jpg"
    existing_image.write_bytes(b"dummy")
    existing_image_2.write_bytes(b"dummy")
    existing_thumb.write_bytes(b"dummy")
    repository.add_media_item(
        id="media_ok",
        file_path=str(existing_image),
        file_name=existing_image.name,
        file_hash="hash-ok-missing-files",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
        thumbnail_path=str(existing_thumb),
    )
    repository.add_media_item(
        id="media_missing_file",
        file_path=str(tmp_path / "gone.jpg"),
        file_name="gone.jpg",
        file_hash="hash-gone",
        media_type="image",
        captured_at="2024-12-24T10:01:00+09:00",
    )
    repository.add_media_item(
        id="media_missing_thumb",
        file_path=str(existing_image_2),
        file_name=existing_image_2.name,
        file_hash="hash-missing-thumb",
        media_type="image",
        captured_at="2024-12-24T10:02:00+09:00",
        thumbnail_path=str(tmp_path / "missing_thumb.jpg"),
    )
    return db_path
