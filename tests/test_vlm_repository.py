from __future__ import annotations

import sqlite3
from pathlib import Path

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_media_vlm_can_be_saved_read_and_searched(tmp_path: Path) -> None:
    repository = _repository_with_media(tmp_path)

    repository.upsert_media_vlm(
        media_id="media_vlm_dummy",
        caption="ラーメンの可能性がある写真",
        short_caption="ラーメン写真の可能性",
        scene_tags=["restaurant"],
        food_cues=["ramen_possible"],
        safety_flags=["low_confidence"],
        vlm_engine="fake",
        model_name="fake-vlm",
        confidence=0.82,
        status="success",
        analysis_version="test",
    )

    row = repository.get_media_vlm("media_vlm_dummy")
    rows = repository.list_media_vlm(start_date="2024-12-24", end_date="2024-12-24", keyword="ラーメン")
    records = repository.search_text_records(terms=["ラーメン"], limit=10)
    media = repository.get_embedding_record("media_item", "media_vlm_dummy")

    assert row is not None
    assert row["status"] == "success"
    assert [item["media_id"] for item in rows] == ["media_vlm_dummy"]
    assert records["media_vlm"][0]["media_id"] == "media_vlm_dummy"
    assert "ラーメン" in (media["caption"] or "")


def test_db_check_detects_orphan_media_vlm(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_vlm (media_id, caption, short_caption, status, analyzed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            ("orphan_vlm", "caption", "caption", "success"),
        )
        connection.commit()

    report = run_db_check(repository.db_path)

    assert report["media_vlm"]["orphan_media_refs"] == 1
    assert report["strict"]["ok"] is False


def _repository_with_media(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_dummy",
        file_path="/local/photos/vlm.jpg",
        file_name="vlm.jpg",
        file_hash="hash-vlm-dummy",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    return repository
