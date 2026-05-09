from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_vlm_review_queue_and_update_cli(tmp_path: Path, capsys) -> None:
    db_path = _seed(tmp_path)

    queue_code = main(
        [
            "--db-path",
            str(db_path),
            "vlm-review-queue",
            "--date",
            "2024-12-24",
            "--unreviewed",
            "--json",
        ]
    )
    queue_payload = json.loads(capsys.readouterr().out)
    update_code = main(
        [
            "--db-path",
            str(db_path),
            "update-vlm-result",
            "media_vlm_cli_review",
            "--accepted",
            "--verified",
            "--caption",
            "確認済みcaption",
            "--tag",
            "meal_possible",
            "--json",
        ]
    )
    update_payload = json.loads(capsys.readouterr().out)

    assert queue_code == 0
    assert queue_payload["results"][0]["media_id"] == "media_vlm_cli_review"
    assert update_code == 0
    assert update_payload["review_status"] == "accepted"
    assert update_payload["is_verified"] == 1


def test_make_vlm_eval_case_cli_outputs_yaml(tmp_path: Path, capsys) -> None:
    code = main(["make-vlm-eval-case", "--query", "ご飯を食べた写真", "--expected-media-id", "media_x"])
    output = capsys.readouterr().out

    assert code == 0
    assert "type: image_search" in output
    assert "media_x" in output


def _seed(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_cli_review",
        file_path="/local/photos/cli_review.jpg",
        file_name="cli_review.jpg",
        file_hash="hash-vlm-cli-review",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_cli_review",
        caption="ご飯の可能性がある写真",
        short_caption="ご飯候補",
        food_cues=["meal_possible"],
        vlm_engine="fake",
        status="success",
        confidence=0.7,
    )
    return db_path
