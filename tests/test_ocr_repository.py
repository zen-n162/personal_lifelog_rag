from __future__ import annotations

import sqlite3
from pathlib import Path

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_media_ocr_can_be_saved_and_read(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _add_media(repository, tmp_path)

    repository.upsert_media_ocr(
        media_id="media_ocr_dummy",
        ocr_text="新宿 看板",
        ocr_text_redacted="新宿 看板",
        ocr_engine="fake",
        ocr_languages=["jpn", "eng"],
        confidence=0.9,
        blocks_json=[{"text": "新宿", "confidence": 0.9}],
        status="success",
        analysis_version="test",
    )

    row = repository.get_media_ocr("media_ocr_dummy")
    assert row is not None
    assert row["ocr_text"] == "新宿 看板"
    assert row["status"] == "success"
    assert row["file_name"] == "ocr_dummy.png"

    media = repository.get_embedding_record("media_item", "media_ocr_dummy")
    assert media is not None
    assert media["ocr_text"] == "新宿 看板"


def test_list_media_ocr_filters_by_date_and_keyword(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _add_media(repository, tmp_path)
    repository.upsert_media_ocr(media_id="media_ocr_dummy", ocr_text="アルバム", status="success")

    rows = repository.list_media_ocr(start_date="2024-12-24", end_date="2024-12-24", keyword="アルバム")

    assert [row["media_id"] for row in rows] == ["media_ocr_dummy"]


def test_db_check_detects_orphan_media_ocr(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_ocr (media_id, ocr_text, status, analyzed_at)
            VALUES ('missing_media', 'dummy', 'success', CURRENT_TIMESTAMP)
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["media_ocr"]["orphan_media_refs"] == 1
    assert report["strict"]["ok"] is False


def _add_media(repository: LifelogRepository, tmp_path: Path) -> None:
    image_path = tmp_path / "ocr_dummy.png"
    image_path.write_bytes(b"dummy")
    repository.add_media_item(
        id="media_ocr_dummy",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-ocr-dummy",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
