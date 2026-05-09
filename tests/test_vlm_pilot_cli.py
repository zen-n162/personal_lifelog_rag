from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.vlm.pilot import VlmPilotOptions, run_vlm_pilot
from personal_lifelog_rag.vlm.schemas import VlmResult


class UnavailablePilotEngine:
    name = "unavailable_pilot"
    model_name = None

    def is_available(self) -> bool:
        return False

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        raise AssertionError("unavailable engine must not analyze images")


def test_vlm_pilot_dry_run_does_not_write_db(tmp_path: Path) -> None:
    repository = _seed_db(tmp_path)

    report = run_vlm_pilot(
        repository,
        repository.db_path,
        VlmPilotOptions(date="2024-12-24", limit=2, engine_name="fake", dry_run=True, backup_dir=tmp_path / "backups"),
    )

    assert report["run_info"]["dry_run"] is True
    assert len(report["selected_images"]) == 2
    assert repository.stats()["media_vlm"] == 0


def test_vlm_pilot_fake_engine_completes_and_saves_reports(tmp_path: Path) -> None:
    repository = _seed_db(tmp_path)

    report = run_vlm_pilot(
        repository,
        repository.db_path,
        VlmPilotOptions(
            date="2024-12-24",
            limit=2,
            engine_name="fake",
            save_report=True,
            force=True,
            output_dir=tmp_path / "eval_outputs",
            backup_dir=tmp_path / "backups",
        ),
    )

    assert report["vlm_report"]["success"] == 2
    assert report["vlm_report"]["success_rate"] == 1.0
    assert report["vlm_report"]["recommendation"] == "continue_to_20"
    assert report["recommendation"] == "continue_to_20"
    assert report["safety_summary"]["flag_counts"]["low_confidence"] == 2
    assert "image-search ご飯" in report["search_smoke_tests"]
    assert Path(report["output_paths"]["json"]).exists()
    assert Path(report["output_paths"]["markdown"]).exists()
    assert any((tmp_path / "backups").glob("lifelog_before_vlm_pilot_20241224_*.sqlite"))


def test_vlm_pilot_engine_unavailable_records_status(tmp_path: Path) -> None:
    repository = _seed_db(tmp_path)

    report = run_vlm_pilot(
        repository,
        repository.db_path,
        VlmPilotOptions(date="2024-12-24", limit=1, engine_name="unavailable", backup_dir=tmp_path / "backups"),
        engine=UnavailablePilotEngine(),
    )

    assert report["vlm_report"]["engine_unavailable"] == 1
    assert report["vlm_report"]["engine_unavailable_rate"] == 1.0
    assert report["vlm_report"]["recommendation"] == "inspect_failures"
    assert repository.get_media_vlm("media_pilot_1")["status"] == "engine_unavailable"


def test_vlm_pilot_cli_json_and_report_output(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path).db_path

    code = main(
        [
            "--db-path",
            str(db_path),
            "vlm-pilot",
            "--date",
            "2024-12-24",
            "--limit",
            "2",
            "--engine",
            "fake",
            "--save-report",
            "--force",
            "--output-dir",
            str(tmp_path / "eval_outputs"),
            "--backup-dir",
            str(tmp_path / "backups"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["run_info"]["date"] == "2024-12-24"
    assert payload["vlm_report"]["success"] == 2
    assert payload["vlm_report"]["recommendation"] == "continue_to_20"
    assert Path(payload["output_paths"]["json"]).exists()


def _seed_db(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for index in range(1, 4):
        image_path = tmp_path / f"pilot_{index}.png"
        Image.new("RGB", (16, 16), "white").save(image_path)
        repository.add_media_item(
            id=f"media_pilot_{index}",
            file_path=str(image_path),
            file_name=image_path.name,
            file_hash=f"hash-pilot-{index}",
            media_type="image",
            captured_at=f"2024-12-24T{9 + index:02d}:00:00+09:00",
            gps_lat=35.0 if index == 1 else None,
            gps_lon=139.0 if index == 1 else None,
        )
    repository.upsert_media_ocr(
        media_id="media_pilot_2",
        ocr_text="ご飯",
        ocr_text_redacted="ご飯",
        ocr_engine="fake",
        status="success",
    )
    repository.add_event(
        id="event_pilot",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="11:00:00",
        title="写真撮影の記録",
        summary="dummy",
        confidence=0.7,
    )
    repository.add_event_evidence(event_id="event_pilot", evidence_type="photo", evidence_id="media_pilot_1")
    return repository
