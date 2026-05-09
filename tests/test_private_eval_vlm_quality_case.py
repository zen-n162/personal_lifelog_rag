from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_vlm_quality_checks_engine_flags_and_empty_captions(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _add_vlm_media(repository, "media_vlm_quality_ok", caption="Meal possible", flags=["people_present"])
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: vlm_quality_ok
    type: vlm_quality
    date: "2024-12-24"
    expected_min_vlm_success: 1
    expected_engine: "qwen3_vl_transformers"
    allowed_safety_flags:
      - "people_present"
    forbidden_terms:
      - "彼女"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    case = report["case_results"][0]
    assert case["status"] == "pass"
    assert case["success_engines"]["qwen3_vl_transformers"] == 1


def test_vlm_quality_detects_forbidden_terms_and_disallowed_flags(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _add_vlm_media(
        repository,
        "media_vlm_quality_bad",
        caption="彼女と楽しそうな写真",
        flags=["relationship_inference_removed"],
    )
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: vlm_quality_bad
    type: vlm_quality
    date: "2024-12-24"
    expected_min_vlm_success: 1
    expected_engine: "qwen3_vl_transformers"
    allowed_safety_flags:
      - "people_present"
    forbidden_terms:
      - "彼女"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    case = report["case_results"][0]
    assert case["status"] == "fail"
    assert any("forbidden VLM terms" in issue for issue in case["issues"])
    assert any("disallowed VLM safety flags" in issue for issue in case["issues"])


def _add_vlm_media(repository: LifelogRepository, media_id: str, *, caption: str, flags: list[str]) -> None:
    repository.add_media_item(
        id=media_id,
        file_path=f"/local/private/{media_id}.jpg",
        file_name=f"{media_id}.jpg",
        file_hash=f"hash-{media_id}",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id=media_id,
        caption=caption,
        short_caption=caption[:24],
        status="success",
        vlm_engine="qwen3_vl_transformers",
        safety_flags=flags,
    )
