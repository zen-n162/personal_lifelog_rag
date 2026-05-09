from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.benchmark.embedding_benchmark import (
    FakeMultimodalEmbeddingEngine,
    benchmark_image_embedding,
    get_multimodal_embedding_engine,
)
from personal_lifelog_rag.benchmark.schemas import BenchmarkCase


def test_fake_embedding_benchmark_computes_ranking_metrics() -> None:
    cases = [
        BenchmarkCase(
            id="food_sample",
            image_path=Path("private_eval/vlm_benchmark/images/food/sample.jpg"),
            query_texts=["料理の写真", "ご飯を食べた写真"],
        ),
        BenchmarkCase(
            id="station_sample",
            image_path=Path("private_eval/vlm_benchmark/images/place/sample.jpg"),
            query_texts=["駅の写真"],
        ),
    ]

    report = benchmark_image_embedding(cases, engine=FakeMultimodalEmbeddingEngine())

    assert report["status"] == "success"
    assert report["case_results"][0]["image_embedding_dim"] == 8
    assert report["metrics"]["top1_accuracy"] == 1.0
    assert report["metrics"]["recall_at_5"] == 1.0


def test_unavailable_embedding_engine_returns_engine_unavailable() -> None:
    engine = get_multimodal_embedding_engine(
        "qwen3_vl_embedding_sentence_transformers",
        model_name="Qwen/Qwen3-VL-Embedding-2B",
        model_path=None,
    )

    report = benchmark_image_embedding(
        [BenchmarkCase(id="food_sample", image_path=Path("food/sample.jpg"), query_texts=["料理"])],
        engine=engine,
    )

    assert report["status"] == "engine_unavailable"
    assert report["case_results"][0]["status"] == "engine_unavailable"

