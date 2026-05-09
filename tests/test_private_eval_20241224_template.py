from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_make_private_eval_template_writes_sanitized_yaml(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    output_path = tmp_path / "private_eval" / "questions_20241224.yaml"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_20241224_baseline(repository, tmp_path)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "make-private-eval-template",
            "--date",
            "2024-12-24",
            "--output",
            str(output_path),
        ]
    )
    output = capsys.readouterr().out
    yaml_text = output_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert "Private eval template generated" in output
    assert "cases:" in yaml_text
    assert "date_20241224_summary" in yaml_text
    assert "mm_food_photo_20241224" in yaml_text
    assert "18時に新宿着く！" not in yaml_text
    assert str(tmp_path) not in yaml_text
    assert "food.jpg" not in yaml_text


def test_generated_20241224_template_can_be_evaluated(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    output_path = tmp_path / "questions_20241224.yaml"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_20241224_baseline(repository, tmp_path)

    assert main(["--db-path", str(db_path), "make-private-eval-template", "--date", "2024-12-24", "--output", str(output_path)]) == 0
    questions = load_private_eval_questions(output_path)
    report = evaluate_private_questions(repository, questions)

    assert report["summary"]["cases"] >= 6
    assert report["summary"]["failed"] == 0
    assert report["by_type"]["vlm_quality"]["passed"] == 1
    assert report["by_type"]["event_quality"]["passed"] == 1


def _seed_20241224_baseline(repository: LifelogRepository, tmp_path: Path) -> None:
    media_id = repository.add_media_item(
        id="media_20241224_food",
        file_path=str(tmp_path / "private_photos" / "food.jpg"),
        file_name="food.jpg",
        file_hash="hash-20241224-food",
        media_type="image",
        captured_at="2024-12-24T15:53:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    line_id = repository.add_line_message(
        id="line_20241224_shinjuku",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.upsert_media_vlm(
        media_id=media_id,
        caption="A meal possible photo in a dining area",
        short_caption="meal possible",
        scene_tags=["indoor"],
        activity_tags=["meal_possible"],
        food_cues=["meal_possible", "rice_possible"],
        safety_flags=["people_present"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
        model_name="Qwen3-VL",
        confidence=0.68,
    )
    event_id = repository.add_event(
        id="event_20241224_food_shinjuku",
        date="2024-12-24",
        start_time="15:50:00",
        end_time="18:30:00",
        title="食事・カフェの可能性",
        summary="場所候補: 新宿。画像解析では食事の可能性があります。",
        location_name="新宿",
        confidence=0.72,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="line", evidence_id=line_id, weight=0.8)
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id=media_id, weight=0.8)
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id=media_id, weight=0.3)
