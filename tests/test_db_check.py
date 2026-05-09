from __future__ import annotations

import json
import sqlite3

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_db_check_passes_for_normal_db(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_normal_db(repository, tmp_path)

    exit_code = main(["--db-path", str(db_path), "db-check"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DB integrity check" in output
    assert "media_items:" in output
    assert "line_messages:" in output
    assert "strict:" in output
    assert "- ok: True" in output


def test_db_check_detects_orphan_evidence(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_evidence (event_id, evidence_type, evidence_id, weight)
            VALUES ('missing_event', 'line', 'missing_line', 1.0)
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["event_evidence"]["orphan_event_refs"] == 1
    assert report["event_evidence"]["missing_line_refs"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_invalid_vlm_event_evidence(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_normal_db(repository, tmp_path)
    repository.upsert_media_vlm(
        media_id="media_ok",
        caption="failed VLM candidate",
        short_caption="failed",
        status="failed",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
    )
    repository.add_event_evidence(event_id="event_ok", evidence_type="vlm", evidence_id="media_ok", weight=0.2)

    report = run_db_check(db_path)

    assert report["event_evidence"]["non_success_vlm_refs"] == 1
    assert report["event_evidence"]["failed_vlm_refs"] == 1
    assert report["event_evidence"]["invalid_vlm_refs"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_orphan_line_call_event(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO line_call_events (
                message_id, chat_id, sent_at, sender, call_status, duration_sec, raw_text_short
            )
            VALUES ('missing_line', 'chat_dummy', '2024-12-24T10:00:00+09:00', '自分', 'completed', 638, 'call')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["line_call_events"]["orphan_message_refs"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_bad_media_embedding(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_embeddings (
                media_id, embedding_type, embedding_model, embedding_dim,
                embedding, embedding_format, status
            )
            VALUES ('missing_media', 'image', 'fake', 3, X'0000', 'float32_numpy', 'success')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["media_embeddings"]["orphan_media_refs"] == 1
    assert report["media_embeddings"]["dimension_mismatch_count"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_bad_vlm_override(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_vlm_overrides (
                media_id, scene_tags_override_json, review_status
            )
            VALUES ('missing_media', '{bad json', 'mystery')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["media_vlm_overrides"]["orphan_media_refs"] == 1
    assert report["media_vlm_overrides"]["unknown_status_count"] == 1
    assert report["media_vlm_overrides"]["invalid_json_count"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_duplicate_file_path(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    shared_file = tmp_path / "shared.jpg"
    shared_file.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_a",
        file_path=str(shared_file),
        file_name=shared_file.name,
        file_hash="hash-a",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    repository.add_media_item(
        id="media_b",
        file_path=str(shared_file),
        file_name=shared_file.name,
        file_hash="hash-b",
        media_type="image",
        captured_at="2024-12-24T11:00:00+09:00",
    )

    report = run_db_check(db_path)

    assert report["media_items"]["duplicate_file_path_groups"] == 1
    assert set(report["media_items"]["duplicate_file_path_sample_ids"]) == {"media_a", "media_b"}
    assert not report["strict"]["ok"]


def test_db_check_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_normal_db(repository, tmp_path)

    exit_code = main(["--db-path", str(db_path), "db-check", "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["media_items"]["total"] == 1
    assert payload["line_messages"]["total"] == 1
    assert payload["strict"]["ok"] is True


def test_db_check_strict_fails_when_severe_issue_exists(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    shared_file = tmp_path / "shared.jpg"
    shared_file.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_a",
        file_path=str(shared_file),
        file_name=shared_file.name,
        file_hash="hash-a",
        media_type="image",
    )
    repository.add_media_item(
        id="media_b",
        file_path=str(shared_file),
        file_name=shared_file.name,
        file_hash="hash-b",
        media_type="image",
    )

    exit_code = main(["--db-path", str(db_path), "db-check", "--strict"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "duplicate file_path groups" in output


def _seed_normal_db(repository: LifelogRepository, tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    thumb_path = tmp_path / "thumb.jpg"
    image_path.write_bytes(b"dummy image")
    thumb_path.write_bytes(b"dummy thumb")
    media_id = repository.add_media_item(
        id="media_ok",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-ok",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
        fallback_captured_at="2024-12-24T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
        thumbnail_path=str(thumb_path),
    )
    line_id = repository.add_line_message(
        id="line_ok",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T10:05:00+09:00",
        sender="自分",
        text="dummy",
    )
    event_id = repository.add_event(
        id="event_ok",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:10:00",
        title="dummy event",
        confidence=0.8,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="photo",
        evidence_id=media_id,
        weight=0.8,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="line",
        evidence_id=line_id,
        weight=0.8,
    )
