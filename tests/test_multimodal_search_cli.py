from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.query_router import _format_multimodal_image_answer


def test_multimodal_search_cli_and_image_search_hybrid(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    image_path = tmp_path / "ramen.jpg"
    image_path.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_cli_mm",
        file_path=str(image_path),
        file_name="ramen.jpg",
        file_hash="hash-cli-mm",
        media_type="image",
        captured_at="2024-12-24T18:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_cli_mm",
        caption="ラーメンまたはご飯の可能性がある写真",
        short_caption="ご飯候補",
        food_cues=["ramen_possible"],
        status="success",
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
    )

    assert main(["--db-path", str(db_path), "build-image-embeddings", "--date", "2024-12-24", "--engine", "fake", "--model", "unit-test-embedding", "--force", "--allow-fake-write"]) == 0
    capsys.readouterr()
    assert main(["--db-path", str(db_path), "build-text-embeddings", "--date", "2024-12-24", "--engine", "fake", "--model", "unit-test-embedding", "--type", "combined_text", "--force", "--allow-fake-write"]) == 0
    capsys.readouterr()
    assert main(["--db-path", str(db_path), "embedding-stats", "--json"]) == 0
    stats_payload = json.loads(capsys.readouterr().out)

    assert stats_payload["total"] == 2

    assert main(["--db-path", str(db_path), "multimodal-search", "ご飯を食べた写真", "--backend", "hybrid", "--engine", "fake", "--model", "unit-test-embedding", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["media_id"] == "media_cli_mm"
    assert "score_components" in payload["results"][0]

    assert main(["--db-path", str(db_path), "image-search", "カフェ", "--backend", "hybrid", "--limit", "10"]) == 0
    output = capsys.readouterr().out
    assert "Multimodal Search" in output

    assert main(["--db-path", str(db_path), "image-search", "ご飯", "--backend", "vlm_sql", "--limit", "10"]) == 0
    output = capsys.readouterr().out
    assert "media_cli_mm" in output


def test_qa_routes_food_photo_query_to_multimodal_search(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    image_path = tmp_path / "food.jpg"
    image_path.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_cli_qa_mm",
        file_path=str(image_path),
        file_name="food.jpg",
        file_hash="hash-cli-qa-mm",
        media_type="image",
        captured_at="2024-12-24T19:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_cli_qa_mm",
        caption="ご飯の可能性がある写真",
        short_caption="ご飯候補",
        food_cues=["meal_possible"],
        status="success",
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
    )

    exit_code = main(["--db-path", str(db_path), "qa", "ご飯を食べた写真はいつ？"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "routing: multimodal-search" in output
    assert "画像解析では" in output
    assert "media_cli_qa_mm" in output


def test_qa_routes_performance_photo_query_to_multimodal_search(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    image_path = tmp_path / "stage.jpg"
    image_path.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_cli_qa_performance",
        file_path=str(image_path),
        file_name="stage.jpg",
        file_hash="hash-cli-qa-performance",
        media_type="image",
        captured_at="2024-12-24T20:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_cli_qa_performance",
        caption="Stage performance candidate",
        short_caption="Performance candidate",
        scene_tags=["stage_possible"],
        activity_tags=["performance_possible"],
        status="success",
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
    )

    exit_code = main(["--db-path", str(db_path), "qa", "パフォーマンスっぽい写真はいつ？"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "意図: multimodal_image_search" in output
    assert "routing: multimodal-search" in output
    assert "media_cli_qa_performance" in output


def test_multimodal_image_answer_summary_uses_rank_one_date() -> None:
    answer = _format_multimodal_image_answer(
        "カフェっぽい写真はいつ？",
        {
            "results": [
                {
                    "date": "2024-12-25",
                    "media_id": "media_rank1",
                    "file_name": "rank1.jpg",
                    "captured_at": "2024-12-25T09:00:00+09:00",
                    "caption": "Cafe candidate",
                    "food_cues": ["cafe_possible"],
                    "matched_terms": ["cafe_possible"],
                    "evidence_strength": "weak",
                    "confidence_label": "低",
                    "score_components": {"final_score": 0.4},
                    "evidence_types": ["vlm"],
                    "reasons": ["VLM tag match"],
                },
                {
                    "date": "2024-12-24",
                    "media_id": "media_rank2",
                    "file_name": "rank2.jpg",
                    "captured_at": "2024-12-24T20:00:00+09:00",
                    "caption": "Cafe candidate",
                    "food_cues": ["cafe_possible"],
                    "matched_terms": ["cafe_possible"],
                    "evidence_strength": "weak",
                    "confidence_label": "低",
                    "score_components": {"final_score": 0.35},
                    "evidence_types": ["vlm"],
                    "reasons": ["VLM tag match"],
                },
            ],
            "backend": "hybrid",
            "query": "カフェっぽい写真はいつ？",
            "total": 2,
            "embedding_engine": "",
            "embedding_status": {"available": False},
        },
    )

    assert "画像解析では、2024-12-25" in answer
    assert "主な候補は 2024-12-25, 2024-12-24 です。" in answer
