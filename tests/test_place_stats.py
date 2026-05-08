from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.schemas import Place
from personal_lifelog_rag.places.stats import place_stats


def test_place_stats_counts_location_names_without_gps_details(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_place_stats(repository)

    report = place_stats(repository, places=[_sensitive_place()])

    assert report["location_counts"] == {"候補地点001": 1}
    assert report["unset_location_count"] == 1
    assert report["sensitive_location_event_count"] == 1
    assert report["photo_evidence_event_count"] == 1
    assert report["gps_event_count"] == 1
    assert "gps_lat" not in json.dumps(report, ensure_ascii=False)
    assert "35.123456" not in json.dumps(report, ensure_ascii=False)


def test_place_stats_cli_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_place_stats(repository)

    exit_code = main(["--db-path", str(db_path), "place-stats", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["total_events"] == 2
    assert payload["location_counts"]["候補地点001"] == 1


def _seed_place_stats(repository: LifelogRepository) -> None:
    media_id = repository.add_media_item(
        id="media_place_stats",
        file_path="/local/photos/place.jpg",
        file_name="place.jpg",
        file_hash="hash-place-stats",
        captured_at="2024-12-24T10:00:00+09:00",
        gps_lat=35.123456,
        gps_lon=139.654321,
    )
    event_id = repository.add_event(
        id="event_place_stats",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:30:00",
        title="位置情報付き写真の記録",
        location_name="候補地点001",
        gps_lat=35.123456,
        gps_lon=139.654321,
        confidence=0.8,
    )
    repository.add_event(
        id="event_no_place",
        date="2024-12-25",
        start_time="10:00:00",
        end_time="10:30:00",
        title="場所未設定イベント",
        confidence=0.4,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="photo",
        evidence_id=media_id,
        weight=0.8,
    )


def _sensitive_place() -> Place:
    return Place(
        id="candidate_place_001",
        name="candidate_place_001",
        display_name="候補地点001",
        lat=35.123456,
        lon=139.654321,
        radius_m=500,
        category="unknown",
        privacy_level="sensitive",
        show_exact_location=False,
    )
