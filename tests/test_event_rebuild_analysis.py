from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_rebuild_analysis import (
    EventRebuildOptions,
    analysis_snapshot,
    rebuild_events_with_analysis,
)


def test_rebuild_events_with_analysis_dry_run_does_not_change_db(tmp_path: Path) -> None:
    repository = _seed_repository(tmp_path)
    before_stats = repository.stats()

    report = rebuild_events_with_analysis(
        repository,
        repository.db_path,
        EventRebuildOptions(date="2024-12-24", dry_run=True, force=True, output_dir=tmp_path / "reports"),
    )

    assert report["run_info"]["dry_run"] is True
    assert repository.stats() == before_stats
    assert report["event_diff"]["event_count_delta"] == 0


def test_rebuild_events_with_analysis_saves_report_and_preserves_overrides(tmp_path: Path) -> None:
    repository = _seed_repository(tmp_path)

    report = rebuild_events_with_analysis(
        repository,
        repository.db_path,
        EventRebuildOptions(
            date="2024-12-24",
            force=True,
            save_report=True,
            output_dir=tmp_path / "reports",
            backup_dir=tmp_path / "backups",
        ),
    )
    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24", include_hidden=True)
    event = events[0]

    assert report["build_report"]["events_created"] == 1
    assert report["after_snapshot"]["summary"]["ocr_evidence_count"] == 1
    assert report["after_snapshot"]["summary"]["vlm_evidence_count"] == 1
    assert Path(report["output_paths"]["json"]).exists()
    assert Path(report["output_paths"]["markdown"]).exists()
    assert event["is_pinned"] == 1
    assert event["is_verified"] == 1


def test_analysis_snapshot_contains_search_and_qa_samples(tmp_path: Path) -> None:
    repository = _seed_repository(tmp_path)

    snapshot = analysis_snapshot(repository, start_date="2024-12-24", end_date="2024-12-24")

    assert snapshot["from"] == "2024-12-24"
    assert "search ご飯" in snapshot["search_samples"]
    assert "ご飯を食べた写真はいつ？" in snapshot["qa_samples"]


def _seed_repository(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_rebuild",
        file_path="/local/rebuild.jpg",
        file_name="rebuild.jpg",
        file_hash="hash-rebuild",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
        ocr_text="新宿 ラーメン",
        caption="ラーメンやご飯の可能性がある料理写真",
        analysis_json={"food_cues": ["ramen_possible"], "scene_tags": ["restaurant"]},
    )
    repository.add_line_message(
        id="line_rebuild",
        chat_id="chat",
        source_file="sample.txt",
        sent_at="2024-12-24T12:10:00+09:00",
        sender="自分",
        text="ご飯を食べるかも",
    )
    repository.add_event(
        id="event_existing",
        date="2024-12-24",
        start_time="12:00:00",
        end_time="12:10:00",
        title="既存イベント",
        summary="既存",
        confidence=0.5,
    )
    # build_events creates a stable id from evidence. Override it before rebuild
    # so preservation can be tested once the same id is recreated.
    from personal_lifelog_rag.timeline.event_builder import build_event_drafts_for_date

    stable_id = build_event_drafts_for_date(repository, "2024-12-24")[0].event_id
    repository.upsert_event_override(str(stable_id), is_pinned=True, is_verified=True)
    return repository
