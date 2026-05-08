from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search


def test_ocr_text_is_searchable_and_reported_as_ocr_evidence(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_ocr_record(repository)

    report = local_text_search(repository, LocalSearchOptions(query="新宿", intent="place_visit", limit=10))
    result = report["results"][0]

    assert result["date"] == "2024-12-24"
    assert result["ocr_match_count"] == 1
    assert "ocr" in result["evidence_types"]
    assert "新宿" in result["ocr_samples"][0]["text"]


def test_private_eval_ocr_quality_case_can_pass(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_ocr_record(repository)

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="ocr_quality_dummy",
                question="2024-12-24",
                case_type="ocr_quality",
                date="2024-12-24",
                expected_min_ocr_success=1,
                expected_evidence_types=["ocr"],
            )
        ],
    )

    assert report["summary"]["passed"] == 1
    assert report["case_results"][0]["ocr_success_count"] == 1


def _seed_ocr_record(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_ocr_search",
        file_path="/local/photos/sign.jpg",
        file_name="sign.jpg",
        file_hash="hash-ocr-search",
        media_type="image",
        captured_at="2024-12-24T14:37:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_ocr(
        media_id="media_ocr_search",
        ocr_text="新宿駅 看板",
        ocr_text_redacted="新宿駅 看板",
        ocr_engine="fake",
        ocr_languages=["jpn"],
        status="success",
        confidence=0.9,
        analysis_version="test",
    )
