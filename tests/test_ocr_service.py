from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ocr.engines import FakeOcrEngine
from personal_lifelog_rag.ocr.ocr_service import OcrImagesOptions, ocr_priority_candidates, ocr_stats, run_ocr_images
from personal_lifelog_rag.ocr.schemas import OcrResult


class UnavailableEngine:
    name = "unavailable_test"

    def is_available(self) -> bool:
        return False

    def recognize(self, image_path: Path, languages: list[str]) -> OcrResult:
        raise AssertionError("recognize should not be called")


def test_ocr_dry_run_does_not_write_db(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    report = run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", dry_run=True, limit=10),
        engine=FakeOcrEngine(default_text="新宿"),
    )

    assert report.selected_images == 1
    assert report.processed == 0
    assert repository.get_media_ocr("media_service") is None


def test_ocr_success_skip_existing_and_force(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    first = run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=FakeOcrEngine(default_text="新宿"),
    )
    second = run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10, skip_existing=True),
        engine=FakeOcrEngine(default_text="上書きされない"),
    )
    forced = run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10, force=True),
        engine=FakeOcrEngine(default_text="上書きされた"),
    )

    assert first.success == 1
    assert second.skipped == 1
    assert forced.success == 1
    assert repository.get_media_ocr("media_service")["ocr_text"] == "上書きされた"


def test_unavailable_engine_records_status_without_crashing(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    report = run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=UnavailableEngine(),
    )

    assert report.engine_unavailable == 1
    assert repository.get_media_ocr("media_service")["status"] == "engine_unavailable"


def test_ocr_stats_returns_counts(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)
    run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=FakeOcrEngine(default_text="新宿"),
    )

    stats = ocr_stats(repository, start_date="2024-12-24", end_date="2024-12-24")

    assert stats["ocr_done_photos"] == 1
    assert stats["status_counts"]["success"] == 1
    assert stats["text_present_count"] == 1
    assert "text_length_distribution" in stats


def test_no_text_detected_status_is_counted(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)

    report = run_ocr_images(
        repository,
        OcrImagesOptions(start_date="2024-12-24", end_date="2024-12-24", limit=10),
        engine=FakeOcrEngine(default_text=""),
    )

    assert report.no_text == 1
    assert repository.get_media_ocr("media_service")["status"] == "no_text"


def test_ocr_priority_selects_vlm_text_candidates_and_skips_missing_files(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)
    missing_path = tmp_path / "missing_text.png"
    repository.add_media_item(
        id="media_missing_text",
        file_path=str(missing_path),
        file_name=missing_path.name,
        file_hash="hash-missing-text",
        media_type="image",
        captured_at="2024-12-24T10:05:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_service",
        caption="A poster with visible text",
        text_cues=["poster_text"],
        contains_text_hint=True,
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )
    repository.upsert_media_vlm(
        media_id="media_missing_text",
        caption="A menu with text",
        text_cues=["menu_text"],
        contains_text_hint=True,
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )

    report = run_ocr_images(
        repository,
        OcrImagesOptions(
            start_date="2024-12-24",
            end_date="2024-12-24",
            dry_run=True,
            limit=10,
            text_cues_only=True,
            ocr_priority=True,
        ),
        engine=FakeOcrEngine(default_text="poster"),
    )

    assert report.selected_images == 1
    assert report.rows[0]["media_id"] == "media_service"


def test_ocr_priority_filters_by_caption_keywords_confidence_and_review_flags(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path)
    repository.upsert_media_vlm(
        media_id="media_service",
        caption="A low confidence poster with a menu label",
        text_cues=["menu_text"],
        contains_text_hint=True,
        confidence=0.8,
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )

    rows = ocr_priority_candidates(
        repository,
        start_date="2024-12-24",
        end_date="2024-12-24",
        caption_keywords=("menu",),
        min_vlm_confidence=0.7,
    )

    assert len(rows) == 1
    assert rows[0]["media_id"] == "media_service"
    assert "caption:menu" in rows[0]["priority_reason"]

    repository.upsert_media_vlm_override(media_id="media_service", is_wrong=1, review_status="wrong")
    assert (
        ocr_priority_candidates(
            repository,
            start_date="2024-12-24",
            end_date="2024-12-24",
            caption_keywords=("menu",),
            min_vlm_confidence=0.7,
        )
        == []
    )


def _repository_with_image(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "service.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    repository.add_media_item(
        id="media_service",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-service",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return repository
