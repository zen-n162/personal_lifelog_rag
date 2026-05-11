from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.query_router import RoutedQueryResult


def test_batch_qa_runs_multiple_queries_and_writes_outputs(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    json_path = tmp_path / "batch.json"
    md_path = tmp_path / "batch.md"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_batch_qa_records(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "batch-qa",
            "--query",
            "ご飯を食べた写真はいつ？",
            "--query",
            "2025年1月は何していた？",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Batch QA" in stdout
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["success"] == 2
    assert payload["queries"][0]["routing"] == "multimodal-search"
    assert payload["queries"][1]["routing"] == "monthly-summary"
    assert "answer" in payload["queries"][0]
    assert "error_message" in payload["queries"][0]
    assert "# Batch QA Run" in md_path.read_text(encoding="utf-8")


def test_batch_qa_continues_when_one_query_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    json_path = tmp_path / "batch_failure.json"

    def fake_route_query(repository, query: str, **kwargs):
        if "失敗" in query:
            raise RuntimeError("intentional test failure")
        return RoutedQueryResult(
            query=query,
            intent="monthly_summary",
            intent_confidence=0.9,
            entities={},
            routing="monthly-summary",
            answer="月次要約です。",
            results=[{"date": "2025-01"}],
            intent_reasons=["test"],
        )

    monkeypatch.setattr("personal_lifelog_rag.app.cli.route_query", fake_route_query)
    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "batch-qa",
            "--query",
            "2025年1月は何していた？",
            "--query",
            "失敗するquery",
            "--output-json",
            str(json_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["success"] == 1
    assert payload["summary"]["failed"] == 1
    assert payload["queries"][1]["success"] is False
    assert "RuntimeError" in payload["queries"][1]["error_message"]


def test_batch_qa_reuses_configured_multimodal_engine(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  multimodal_embedding:",
                '    engine: "qwen3_vl_embedding"',
                '    model_name: "Qwen/Qwen3-VL-Embedding-8B"',
                f'    model_path: "{tmp_path / "local-model"}"',
                "    local_files_only: true",
                "    embedding_dim: 4096",
                "    batch_size: 4",
            ]
        ),
        encoding="utf-8",
    )
    engine = object()
    calls: list[dict[str, object]] = []
    seen_engine_ids: list[int] = []

    def fake_get_cached_engine(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return engine

    def fake_route_query(repository, query: str, **kwargs):
        seen_engine_ids.append(id(kwargs.get("multimodal_engine")))
        return RoutedQueryResult(
            query=query,
            intent="multimodal_image_search",
            intent_confidence=0.9,
            entities={},
            routing="multimodal-search",
            answer=f"{query} answer",
            results=[],
            intent_reasons=["test"],
        )

    monkeypatch.setattr("personal_lifelog_rag.app.cli.get_cached_multimodal_embedding_engine", fake_get_cached_engine)
    monkeypatch.setattr("personal_lifelog_rag.app.cli.route_query", fake_route_query)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "batch-qa",
            "--config",
            str(config_path),
            "--query",
            "ご飯を食べた写真はいつ？",
            "--query",
            "ステージの写真はいつ？",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["kwargs"]["model_path"] == str(tmp_path / "local-model")
    assert calls[0]["kwargs"]["local_files_only"] is True
    assert seen_engine_ids == [id(engine), id(engine)]


def test_batch_qa_reads_queries_file_and_summary_only(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    queries_path = tmp_path / "queries.txt"
    json_path = tmp_path / "queries.json"
    queries_path.write_text(
        "\n".join(
            [
                "# comment",
                "2025年1月は何していた？",
                "ステージの写真はいつ？",
            ]
        ),
        encoding="utf-8",
    )

    def fake_route_query(repository, query: str, **kwargs):
        return RoutedQueryResult(
            query=query,
            intent="monthly_summary" if "1月" in query else "multimodal_image_search",
            intent_confidence=0.9,
            entities={},
            routing="monthly-summary" if "1月" in query else "multimodal-search",
            answer=f"{query} answer",
            results=[{"date": "2025-01-03"}],
            intent_reasons=["test"],
        )

    monkeypatch.setattr("personal_lifelog_rag.app.cli.route_query", fake_route_query)
    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "batch-qa",
            "--queries-file",
            str(queries_path),
            "--summary-only",
            "--output-json",
            str(json_path),
        ]
    )
    capsys.readouterr()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["summary"]["total"] == 2
    assert payload["queries"][0]["answer"] == ""
    assert payload["queries"][0]["answer_summary"]
    assert payload["queries"][0]["top_dates"] == ["2025-01-03"]


def test_batch_qa_fail_fast_stops_after_first_error(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    json_path = tmp_path / "fail_fast.json"

    def fake_route_query(repository, query: str, **kwargs):
        if "失敗" in query:
            raise RuntimeError("intentional test failure")
        return RoutedQueryResult(
            query=query,
            intent="monthly_summary",
            intent_confidence=0.9,
            entities={},
            routing="monthly-summary",
            answer="月次要約です。",
            results=[],
            intent_reasons=["test"],
        )

    monkeypatch.setattr("personal_lifelog_rag.app.cli.route_query", fake_route_query)
    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "batch-qa",
            "--query",
            "失敗するquery",
            "--query",
            "実行されないquery",
            "--fail-fast",
            "--output-json",
            str(json_path),
        ]
    )
    capsys.readouterr()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["failed"] == 1


def _seed_batch_qa_records(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_batch_food",
        file_path="/local/private/batch_food.jpg",
        file_name="batch_food.jpg",
        file_hash="hash-batch-food",
        captured_at="2025-01-05T12:30:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(
        media_id="media_batch_food",
        caption="meal possible",
        short_caption="meal possible",
        food_cues=["meal_possible", "rice_possible"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )
    repository.add_line_message(
        id="line_batch_food",
        chat_id="chat_batch",
        source_file="sample_chat.txt",
        sent_at="2025-01-05T12:00:00+09:00",
        sender="自分",
        text="カフェでご飯",
    )
    repository.add_event(
        id="event_batch_food",
        date="2025-01-05",
        start_time="12:00:00",
        end_time="13:00:00",
        title="食事・カフェの可能性",
        summary="LINEと画像解析による候補",
        confidence=0.7,
    )
    repository.add_event_evidence(event_id="event_batch_food", evidence_type="line", evidence_id="line_batch_food")
    repository.add_event_evidence(event_id="event_batch_food", evidence_type="photo", evidence_id="media_batch_food")
    repository.add_event_evidence(event_id="event_batch_food", evidence_type="vlm", evidence_id="media_batch_food")
