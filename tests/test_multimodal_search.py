from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.engines import FakeEmbeddingEngine
from personal_lifelog_rag.embeddings.base import MultimodalEmbeddingEngine
from personal_lifelog_rag.embeddings.multimodal_search import (
    compute_multimodal_evidence_strength,
    multimodal_search,
)
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions


def test_multimodal_search_hybrid_combines_embedding_vlm_ocr_line_event(tmp_path: Path) -> None:
    repository = _seed_multimodal_records(tmp_path)

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="ご飯を食べた写真", backend="hybrid", limit=5),
        engine=FakeEmbeddingEngine(),
    )
    result = report["results"][0]

    assert result["date"] == "2024-12-24"
    assert "embedding" in result["evidence_types"]
    assert "vlm" in result["evidence_types"]
    assert "ocr" in result["evidence_types"]
    assert "line" in result["evidence_types"]
    assert result["evidence_strength"] in {"medium", "strong"}
    assert "final_score" in result["score_components"]


def test_multimodal_embedding_only_is_weak(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_embedding_only",
        file_path=str(tmp_path / "food_only.jpg"),
        file_name="food_only.jpg",
        file_hash="hash-embedding-only",
        media_type="image",
        captured_at="2024-12-25T10:00:00+09:00",
    )
    vector = FakeEmbeddingEngine().embed_text("ご飯").vector
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_embedding_only",
        embedding_type="image",
            embedding_model="unit-test-embedding",
        vector=vector,
        status="success",
    )

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="ご飯", backend="embedding", limit=5),
        engine=FakeEmbeddingEngine(model_name="unit-test-embedding"),
    )

    assert report["results"][0]["evidence_strength"] == "weak"
    assert report["results"][0]["score_components"]["final_score"] <= 0.44


def test_multimodal_vlm_sql_fallback_returns_vlm_food_cues_without_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_sql_food",
        file_path=str(tmp_path / "food.jpg"),
        file_name="food.jpg",
        file_hash="hash-vlm-sql-food",
        media_type="image",
        captured_at="2024-12-24T15:53:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_sql_food",
        caption="Table with meal-like food items",
        short_caption="Meal candidate",
        food_cues=["meal_possible", "rice_possible"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
    )

    report = multimodal_search(repository, MultimodalSearchOptions(query="ご飯を食べた写真", backend="vlm_sql", limit=5))

    assert report["results"][0]["media_id"] == "media_vlm_sql_food"
    assert report["results"][0]["evidence_strength"] == "weak"
    assert "meal_possible" in report["results"][0]["matched_terms"]
    assert report["results"][0]["score_components"]["embedding_score"] == 0.0


def test_multimodal_vlm_sql_fallback_returns_performance_tags(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_sql_performance",
        file_path=str(tmp_path / "stage.jpg"),
        file_name="stage.jpg",
        file_hash="hash-vlm-sql-performance",
        media_type="image",
        captured_at="2024-12-24T20:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_sql_performance",
        caption="A stage or performance venue possible",
        short_caption="Performance candidate",
        scene_tags=["stage_possible", "theater_possible"],
        activity_tags=["performance_possible"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
    )

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="パフォーマンスっぽい写真", backend="vlm_sql", limit=5),
    )

    assert report["results"][0]["media_id"] == "media_vlm_sql_performance"
    assert "performance_possible" in report["results"][0]["matched_terms"]


def test_multimodal_vlm_sql_with_related_event_is_medium(tmp_path: Path) -> None:
    repository = _seed_multimodal_records(tmp_path)

    report = multimodal_search(repository, MultimodalSearchOptions(query="ご飯を食べた写真", backend="vlm_sql", limit=5))

    assert report["results"][0]["media_id"] == "media_multimodal_food"
    assert report["results"][0]["evidence_strength"] in {"medium", "strong"}


def test_hybrid_falls_back_to_vlm_sql_when_embedding_engine_unavailable(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_fallback_food",
        file_path=str(tmp_path / "food.jpg"),
        file_name="food.jpg",
        file_hash="hash-fallback-food",
        media_type="image",
        captured_at="2024-12-24T15:53:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_fallback_food",
        caption="Meal candidate",
        food_cues=["meal_possible"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_fallback_food",
        embedding_type="image",
        embedding_model="real-model",
        status="engine_unavailable",
    )

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="ご飯を食べた写真", backend="hybrid", limit=5),
        engine=UnavailableEmbeddingEngine(),
    )

    assert report["results"][0]["media_id"] == "media_fallback_food"
    assert report["results"][0]["score_components"]["embedding_score"] == 0.0


def test_embedding_backend_reports_unavailable_without_fallback(tmp_path: Path) -> None:
    repository = _seed_multimodal_records(tmp_path)

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="ご飯を食べた写真", backend="embedding", limit=5),
        engine=UnavailableEmbeddingEngine(),
    )

    assert report["results"] == []
    assert "not available" in report["embedding_status"]["reason"]


def test_people_present_only_does_not_become_high_confidence(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_people_only",
        file_path=str(tmp_path / "people.jpg"),
        file_name="people.jpg",
        file_hash="hash-people-only",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_people_only",
        caption="people_present",
        safety_flags=["people_present"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
        confidence=0.99,
    )

    report = multimodal_search(repository, MultimodalSearchOptions(query="people_present", backend="vlm_sql", limit=5))

    assert report["results"][0]["confidence_label"] != "高"


def test_hybrid_visual_mismatch_context_does_not_become_strong(tmp_path: Path) -> None:
    repository = _seed_visual_match_records(tmp_path)

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="ダンスの写真", backend="hybrid", limit=10),
        engine=FixedQueryEmbeddingEngine(),
    )
    nonvisual = next(row for row in report["results"] if row["media_id"] == "media_nonvisual_context")

    assert nonvisual["score_components"]["visual_match"] == 0.0
    assert nonvisual["score_components"]["line_score"] <= 0.2
    assert nonvisual["score_components"]["event_score"] <= 0.2
    assert nonvisual["evidence_strength"] == "weak"
    assert nonvisual["confidence_label"] == "低"


def test_hybrid_visual_match_with_context_can_be_strong(tmp_path: Path) -> None:
    repository = _seed_visual_match_records(tmp_path)

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="ダンスの写真", backend="hybrid", limit=10),
        engine=FixedQueryEmbeddingEngine(),
    )
    visual = next(row for row in report["results"] if row["media_id"] == "media_visual_dance")

    assert visual["score_components"]["visual_match"] == 1.0
    assert visual["evidence_strength"] == "strong"
    assert report["results"][0]["media_id"] == "media_visual_dance"


def test_evidence_strength_rules() -> None:
    assert compute_multimodal_evidence_strength(evidence_types=["photo", "embedding"]) == "weak"
    assert compute_multimodal_evidence_strength(evidence_types=["photo", "embedding", "vlm"]) == "medium"
    assert (
        compute_multimodal_evidence_strength(
            evidence_types=["photo", "embedding", "vlm", "event", "line", "gps"],
            visual_match=False,
        )
        == "weak"
    )
    assert (
        compute_multimodal_evidence_strength(evidence_types=["photo", "embedding", "vlm", "ocr", "line"])
        == "strong"
    )


def _seed_multimodal_records(tmp_path: Path) -> LifelogRepository:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_multimodal_food",
        file_path=str(tmp_path / "ramen.jpg"),
        file_name="ramen.jpg",
        file_hash="hash-multimodal-food",
        media_type="image",
        captured_at="2024-12-24T19:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_ocr(media_id="media_multimodal_food", ocr_text="ご飯 メニュー", status="success")
    repository.upsert_media_vlm(
        media_id="media_multimodal_food",
        caption="ご飯または料理の可能性がある写真",
        short_caption="ご飯候補",
        food_cues=["meal_possible"],
        status="success",
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
    )
    repository.add_line_message(
        id="line_multimodal_food",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T19:30:00+09:00",
        sender="自分",
        text="今日のご飯おいしかったね",
    )
    event_id = repository.add_event(
        id="event_multimodal_food",
        date="2024-12-24",
        start_time="19:00:00",
        end_time="20:00:00",
        title="食事・カフェの可能性",
        summary="ご飯に関する可能性",
        confidence=0.7,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id="media_multimodal_food")
    vector = FakeEmbeddingEngine().embed_text("ご飯").vector
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_multimodal_food",
        embedding_type="image",
        embedding_model="unit-test-embedding",
        vector=vector,
        status="success",
    )
    return repository


def _seed_visual_match_records(tmp_path: Path) -> LifelogRepository:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_nonvisual_context",
        file_path=str(tmp_path / "children_collage.jpg"),
        file_name="children_collage.jpg",
        file_hash="hash-nonvisual-context",
        media_type="image",
        captured_at="2024-12-24T22:21:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(
        media_id="media_nonvisual_context",
        caption="Black-and-white photo collage of children",
        short_caption="Photo collage",
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )
    nonvisual_event_id = repository.add_event(
        id="event_nonvisual_context",
        date="2024-12-24",
        start_time="22:00:00",
        end_time="23:00:00",
        title="食事・カフェの可能性",
        summary="同日イベント",
        confidence=0.95,
        location_name="テスト場所",
    )
    repository.add_event_evidence(event_id=nonvisual_event_id, evidence_type="photo", evidence_id="media_nonvisual_context")
    repository.add_line_message(
        id="line_nonvisual_dance",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T22:30:00+09:00",
        sender="自分",
        text="ダンスの話をした",
    )
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_nonvisual_context",
        embedding_type="image",
        embedding_model="fixed-test-embedding",
        vector=[0.27, 0.96286136],
        status="success",
    )

    repository.add_media_item(
        id="media_visual_dance",
        file_path=str(tmp_path / "dance.jpg"),
        file_name="dance.jpg",
        file_hash="hash-visual-dance",
        media_type="image",
        captured_at="2024-12-14T17:20:00+09:00",
        gps_lat=35.1,
        gps_lon=139.1,
    )
    repository.upsert_media_vlm(
        media_id="media_visual_dance",
        caption="Stage performance with dancing",
        short_caption="Dance candidate",
        activity_tags=["dancing_possible"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )
    visual_event_id = repository.add_event(
        id="event_visual_dance",
        date="2024-12-14",
        start_time="17:00:00",
        end_time="18:00:00",
        title="写真撮影の記録",
        summary="ステージの写真",
        confidence=0.8,
        location_name="テスト会場",
    )
    repository.add_event_evidence(event_id=visual_event_id, evidence_type="photo", evidence_id="media_visual_dance")
    repository.add_line_message(
        id="line_visual_dance",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-14T17:30:00+09:00",
        sender="自分",
        text="ダンスの写真を撮った",
    )
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_visual_dance",
        embedding_type="image",
        embedding_model="fixed-test-embedding",
        vector=[0.3, 0.9539392],
        status="success",
    )
    return repository


class UnavailableEmbeddingEngine(MultimodalEmbeddingEngine):
    name = "unavailable-test"
    model_name = "unavailable-test"

    def is_available(self) -> bool:
        return False

    def availability_error(self) -> str:
        return "embedding engine is not available in test"

    def embed_image(self, image_path: Path):
        raise AssertionError("embed_image should not be called")

    def embed_text(self, text: str):
        raise AssertionError("embed_text should not be called")


class FixedQueryEmbeddingEngine(MultimodalEmbeddingEngine):
    name = "fixed-test"
    model_name = "fixed-test-embedding"

    def is_available(self) -> bool:
        return True

    def embed_image(self, image_path: Path):
        raise AssertionError("embed_image should not be called")

    def embed_text(self, text: str):
        from personal_lifelog_rag.embeddings.schemas import EmbeddingResult

        return EmbeddingResult(vector=[1.0, 0.0], model_name=self.model_name, embedding_dim=2, status="success")
