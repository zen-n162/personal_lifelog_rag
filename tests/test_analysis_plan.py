from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.jobs.planners import plan_analysis
from personal_lifelog_rag.jobs.schemas import AnalysisPlanOptions


def test_analysis_plan_counts_existing_statuses(tmp_path: Path) -> None:
    repository = _repository_with_media(tmp_path, count=3)
    repository.upsert_media_vlm(
        media_id="media_plan_1",
        caption="caption",
        short_caption="caption",
        scene_tags=[],
        object_tags=[],
        activity_tags=[],
        location_cues=[],
        food_cues=[],
        text_cues=[],
        uncertainty_notes=[],
        evidence_strength="weak",
        people_count=0,
        contains_text_hint=False,
        safety_flags=[],
        vlm_engine="fake",
        model_name="fake-vlm",
        prompt_version="lifelog_structured_tags_v1",
        confidence=0.8,
        status="success",
        error_message=None,
        analysis_version="vlm_v1",
    )
    repository.upsert_media_vlm(
        media_id="media_plan_2",
        caption=None,
        short_caption=None,
        scene_tags=[],
        object_tags=[],
        activity_tags=[],
        location_cues=[],
        food_cues=[],
        text_cues=[],
        uncertainty_notes=[],
        evidence_strength="weak",
        people_count=0,
        contains_text_hint=False,
        safety_flags=[],
        vlm_engine="fake",
        model_name="fake-vlm",
        prompt_version="lifelog_structured_tags_v1",
        confidence=None,
        status="failed",
        error_message="boom",
        analysis_version="vlm_v1",
    )

    plan = plan_analysis(
        repository,
        AnalysisPlanOptions(job_type="vlm", start_date="2024-12-24", end_date="2024-12-24", limit=10),
    )

    assert plan.total_candidates == 3
    assert plan.already_success == 1
    assert plan.failed == 1
    assert [item.item_id for item in plan.selected_items] == ["media_plan_2", "media_plan_3"]


def test_analysis_plan_detects_version_changed_only(tmp_path: Path) -> None:
    repository = _repository_with_media(tmp_path, count=1)
    repository.upsert_media_vlm(
        media_id="media_plan_1",
        caption="caption",
        short_caption="caption",
        scene_tags=[],
        object_tags=[],
        activity_tags=[],
        location_cues=[],
        food_cues=[],
        text_cues=[],
        uncertainty_notes=[],
        evidence_strength="weak",
        people_count=0,
        contains_text_hint=False,
        safety_flags=[],
        vlm_engine="fake",
        model_name="fake-vlm",
        prompt_version="old_prompt",
        confidence=0.8,
        status="success",
        error_message=None,
        analysis_version="vlm_v1",
    )

    plan = plan_analysis(
        repository,
        AnalysisPlanOptions(
            job_type="vlm",
            start_date="2024-12-24",
            end_date="2024-12-24",
            prompt_version="lifelog_structured_tags_v1",
            version_changed_only=True,
        ),
    )

    assert plan.version_changed == 1
    assert [item.item_id for item in plan.selected_items] == ["media_plan_1"]


def _repository_with_media(tmp_path: Path, *, count: int) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for index in range(1, count + 1):
        image_path = tmp_path / f"plan_{index}.jpg"
        image_path.write_bytes(b"not-a-real-image")
        repository.add_media_item(
            id=f"media_plan_{index}",
            file_path=str(image_path),
            file_name=image_path.name,
            file_hash=f"hash-plan-{index}",
            media_type="image",
            captured_at=f"2024-12-24T1{index}:00:00+09:00",
        )
    return repository
