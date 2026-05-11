from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.vlm.engines import FakeVlmEngine
from personal_lifelog_rag.vlm.schemas import VlmResult
from personal_lifelog_rag.vlm.vlm_service import VlmImagesOptions, run_vlm_images, vlm_stats


class UnavailableVlmEngine:
    name = "unavailable_vlm"
    model_name = None

    def is_available(self) -> bool:
        return False

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        raise AssertionError("analyze_image should not be called")


class EmptyCaptionEngine:
    name = "empty_caption_test"
    model_name = "empty-caption-test"

    def is_available(self) -> bool:
        return True

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        return VlmResult(engine=self.name, model_name=self.model_name, status="success", caption="")


def test_vlm_dry_run_does_not_write_db(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    report = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", dry_run=True, limit=10),
        engine=FakeVlmEngine(),
    )

    assert report.selected_images == 1
    assert report.processed == 0
    assert repository.get_media_vlm("media_vlm_service") is None


def test_vlm_success_skip_existing_and_force(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    first = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=FakeVlmEngine(caption="ラーメンの可能性"),
    )
    second = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10, skip_existing=True),
        engine=FakeVlmEngine(caption="上書きされない"),
    )
    forced = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10, force=True),
        engine=FakeVlmEngine(caption="カフェの可能性"),
    )

    assert first.success == 1
    assert second.skipped == 1
    assert forced.success == 1
    assert "カフェ" in repository.get_media_vlm("media_vlm_service")["caption"]


def test_unavailable_vlm_engine_records_status_without_crashing(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    report = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=UnavailableVlmEngine(),
    )

    assert report.engine_unavailable == 1
    assert repository.get_media_vlm("media_vlm_service")["status"] == "engine_unavailable"


def test_empty_caption_success_is_recorded_as_failed(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    report = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=EmptyCaptionEngine(),
    )

    row = repository.get_media_vlm("media_vlm_service")
    assert report.failed == 1
    assert row["status"] == "failed"
    assert "empty caption" in row["error_message"].lower()


def test_vlm_stats_returns_tag_counts(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)
    run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=FakeVlmEngine(),
    )

    stats = vlm_stats(repository, start_date="2024-12-24", end_date="2024-12-24")

    assert stats["total_media_vlm"] == 1
    assert stats["status_counts"]["success"] == 1
    assert stats["food_cues_top"]["ramen_possible"] == 1


def test_vlm_images_skip_missing_original_files(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)
    repository.add_media_item(
        id="missing_vlm_file",
        file_path=str(tmp_path / "missing_vlm.png"),
        file_name="missing_vlm.png",
        file_hash="hash-missing-vlm",
        media_type="image",
        captured_at="2024-12-24T10:05:00+09:00",
    )

    report = run_vlm_images(
        repository,
        VlmImagesOptions(start_date="2024-12-24", end_date="2024-12-24", dry_run=True, limit=10),
        engine=FakeVlmEngine(),
    )

    assert report.selected_images == 1
    assert report.rows[0]["media_id"] == "media_vlm_service"


def _repository_with_image(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "vlm_service.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    repository.add_media_item(
        id="media_vlm_service",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-vlm-service",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    return repository
