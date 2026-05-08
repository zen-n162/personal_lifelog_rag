from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.event_review_service import make_eval_case_yaml


def test_make_eval_case_from_event(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_eval",
        date="2024-12-24",
        start_time="18:00:00",
        end_time="19:00:00",
        title="新宿の出来事",
        summary="manual eval seed",
        confidence=0.7,
    )

    yaml_text = make_eval_case_yaml(repository, event_id="event_eval", case_type="date_qa")

    assert "type: date_qa" in yaml_text
    assert "2024-12-24" in yaml_text
    assert "should_not_include" in yaml_text


def test_make_eval_case_from_query_cli(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "make-eval-case",
            "--query",
            "新宿に行ったのはいつ？",
            "--expected-date",
            "2024-12-24",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "type: routed_qa" in output
    assert "expected_top_dates" in output
    assert "2024-12-24" in output
