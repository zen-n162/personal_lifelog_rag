from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import (
    _forbidden_claims_found,
    evaluate_private_questions,
    load_private_eval_questions,
    write_private_eval_report,
)


def test_load_private_eval_questions_from_simple_yaml(tmp_path) -> None:
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """questions:
  - id: dummy_day
    question: "2024年12月24日は何していた？"
    expected_date: "2024-12-24"
    require_events: true
    min_events: 1
    expected_keywords: ["根拠:", "信頼度:"]
    forbidden_claims: ["デートしていた"]
""",
        encoding="utf-8",
    )

    questions = load_private_eval_questions(questions_path)

    assert len(questions) == 1
    assert questions[0].id == "dummy_day"
    assert questions[0].expected_date == "2024-12-24"
    assert questions[0].require_events is True
    assert questions[0].min_events == 1


def test_load_private_eval_cases_fixture() -> None:
    questions = load_private_eval_questions("tests/fixtures/eval/private_eval_sample.yaml")

    assert len(questions) == 3
    assert questions[0].id == "date_001"
    assert questions[0].case_type == "date_qa"
    assert questions[0].expected_dates == ["2024-12-24"]
    assert questions[0].expected_keywords == ["通話・連絡"]
    assert questions[0].expected_evidence_types == ["line", "photo"]
    assert questions[0].forbidden_claims == ["確実に", "デートしていた"]
    assert questions[0].max_confidence_for_activity == "高"


def test_load_private_eval_questions_from_bare_yaml_list_with_aliases(tmp_path) -> None:
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """  - id: date_001
    type: date_qa
    question: "2024年12月24日は何していた？"
    expected_dates:
      - "2024-12-24"
    expected_event_keywords:
      - "根拠:"
      - "信頼度:"
    expected_evidence_types:
      - line
      - photo
    should_not_include:
      - "デートしていた"
    expected_min_events: 1
  - id: date_002
    question: "2024年12月25日は何していた？"
""",
        encoding="utf-8",
    )

    questions = load_private_eval_questions(questions_path)

    assert len(questions) == 2
    assert questions[0].id == "date_001"
    assert questions[0].case_type == "date_qa"
    assert questions[0].expected_date == "2024-12-24"
    assert questions[0].expected_dates == ["2024-12-24"]
    assert questions[0].expected_keywords == ["根拠:", "信頼度:"]
    assert questions[0].expected_evidence_types == ["line", "photo"]
    assert questions[0].forbidden_claims == ["デートしていた"]
    assert questions[0].min_events == 1


def test_private_eval_keyword_search_matches_result_dates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_keyword",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T18:00:00+09:00",
        sender="自分",
        text="新宿に着いた",
    )
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """questions:
  - id: search_001
    type: keyword_search
    question: "新宿に行ったのはいつ？"
    query: "新宿"
    expected_dates:
      - "2024-12-24"
    expected_evidence_types:
      - line
""",
        encoding="utf-8",
    )
    questions = load_private_eval_questions(questions_path)

    report = evaluate_private_questions(repository, questions)

    case = report["cases"][0]
    assert case["passed"] is True
    assert case["status"] == "pass"
    assert case["parsed_date"] is None
    assert case["matched_dates"] == ["2024-12-24"]


def test_private_eval_keyword_search_can_skip_when_unavailable(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: search_001
    type: keyword_search
    question: "新宿に行ったのはいつ？"
    query: "新宿"
""",
        encoding="utf-8",
    )
    questions = load_private_eval_questions(questions_path)

    report = evaluate_private_questions(repository, questions, keyword_search_available=False)

    assert report["summary"] == {"cases": 1, "total": 1, "passed": 0, "failed": 0, "skipped": 1}
    assert report["cases"][0]["status"] == "skip"
    assert report["cases"][0]["skip_reason"] == "search command is not implemented yet"


def test_private_eval_passes_with_dummy_event_evidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_eval_records(repository)
    questions = load_private_eval_questions(_write_questions(tmp_path))

    report = evaluate_private_questions(repository, questions)

    assert report["summary"] == {"cases": 1, "total": 1, "passed": 1, "failed": 0, "skipped": 0}
    case = report["cases"][0]
    assert case["status"] == "pass"
    assert case["parsed_date"] == "2024-12-24"
    assert case["events_count"] == 1
    assert case["event_evidence_count"] == 2
    assert case["unsupported_claims"] == []


def test_private_eval_detects_expected_keyword_and_forbidden_phrase(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: keyword_and_forbidden
    type: date_qa
    question: "2024年12月24日は何していた？"
    expected_dates:
      - "2024-12-24"
    expected_event_keywords:
      - "通話・連絡"
      - "存在しないキーワード"
    should_not_include:
      - "通話・連絡"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    case = report["cases"][0]
    assert case["status"] == "fail"
    assert case["matched_expected_keywords"] == ["通話・連絡"]
    assert case["missing_expected_keywords"] == ["存在しないキーワード"]
    assert case["forbidden_claims_found"] == ["通話・連絡"]


def test_forbidden_claims_ignore_cautious_negative_context() -> None:
    assert _forbidden_claims_found("具体的な活動内容は断定できません。", ["断定"]) == []
    assert _forbidden_claims_found("食事をしていたと断定できます。", ["食事"]) == ["食事"]


def test_private_eval_cli_writes_run_file(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = _write_questions(tmp_path)
    output_dir = tmp_path / "runs"

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "private-eval",
            "--questions",
            str(questions_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    output = capsys.readouterr().out
    run_files = list(output_dir.glob("*.json"))

    assert exit_code == 0
    assert "Private Eval" in output
    assert "[PASS] dummy_day" in output
    assert len(run_files) == 0


def test_private_eval_cli_save_run_writes_eval_named_file(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = _write_questions(tmp_path)
    output_dir = tmp_path / "runs"

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "private-eval",
            "--questions",
            str(questions_path),
            "--output-dir",
            str(output_dir),
            "--save-run",
        ]
    )
    output = capsys.readouterr().out
    run_files = list(output_dir.glob("eval_*.json"))

    assert exit_code == 0
    assert "run_file:" in output
    assert len(run_files) == 1
    payload = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] == 1


def test_write_private_eval_report_does_not_overwrite_same_run_id(tmp_path) -> None:
    report = {
        "run_id": "eval_20990101_000000",
        "summary": {"cases": 0, "total": 0, "passed": 0, "failed": 0, "skipped": 0},
        "cases": [],
        "case_results": [],
    }

    first = write_private_eval_report(report, tmp_path)
    second = write_private_eval_report(report, tmp_path)

    assert first.name == "eval_20990101_000000.json"
    assert second.name == "eval_20990101_000000_1.json"
    assert first.read_text(encoding="utf-8")
    assert second.read_text(encoding="utf-8")


def test_private_eval_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = _write_questions(tmp_path)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "private-eval",
            "--questions",
            str(questions_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["skipped"] == 0
    assert payload["cases"][0]["id"] == "dummy_day"


def test_eval_private_alias_accepts_path_and_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = _write_questions(tmp_path)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "eval-private",
            "--path",
            str(questions_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["summary"]["passed"] == 1
    assert payload["cases"][0]["id"] == "dummy_day"


def test_eval_private_save_run_option_is_accepted(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = _write_questions(tmp_path)
    output_dir = tmp_path / "runs"

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "eval-private",
            "--path",
            str(questions_path),
            "--output-dir",
            str(output_dir),
            "--save-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[PASS] dummy_day" in output
    assert len(list(output_dir.glob("*.json"))) == 1


def test_eval_private_case_id_filters_questions(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """questions:
  - id: date_001
    question: "2024年12月24日は何していた？"
    expected_date: "2024-12-24"
    require_events: true
    expected_keywords: ["根拠:", "信頼度:"]
  - id: date_002
    question: "2024年12月25日は何していた？"
    expected_date: "2024-12-25"
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "eval-private",
            "--path",
            str(questions_path),
            "--case-id",
            "date_001",
            "--no-save-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "cases: 1" in output
    assert "[PASS] date_001" in output
    assert "date_002" not in output


def test_private_eval_strict_fails_when_checks_fail(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_eval_records(repository)
    questions_path = tmp_path / "bad_questions.yaml"
    questions_path.write_text(
        """questions:
  - id: bad_expectation
    question: "2024年12月24日は何していた？"
    expected_date: "2024-12-24"
    require_events: true
    expected_keywords: ["存在しないキーワード"]
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "private-eval",
            "--questions",
            str(questions_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--strict",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "[FAIL] bad_expectation" in output


def _write_questions(tmp_path):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """questions:
  - id: dummy_day
    question: "2024年12月24日は何していた？"
    expected_date: "2024-12-24"
    require_events: true
    min_events: 1
    expected_keywords: ["根拠:", "信頼度:"]
    forbidden_claims: ["デートしていた", "喧嘩した"]
""",
        encoding="utf-8",
    )
    return questions_path


def _seed_eval_records(repository: LifelogRepository) -> None:
    media_id = repository.add_media_item(
        id="media_dummy",
        file_path="/local/photos/dummy.jpg",
        file_name="dummy.jpg",
        file_hash="hash-dummy",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    line_id = repository.add_line_message(
        id="line_dummy",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T10:05:00+09:00",
        sender="自分",
        text="☎ 通話時間 1:00",
    )
    event_id = repository.add_event(
        id="event_dummy",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:10:00",
        title="通話・連絡",
        summary="LINE 1件、写真 1枚から作成したイベント候補です。",
        confidence=0.7,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="photo",
        evidence_id=media_id,
        weight=0.8,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="line",
        evidence_id=line_id,
        weight=0.8,
    )
