from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person, public_person_name
from personal_lifelog_rag.reporting.report_builder import build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions
from personal_lifelog_rag.retrieval.query_router import route_query


def test_public_person_name_uses_public_alias_or_hides_private_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    private_person = create_person(repository, name="Private Real Name", privacy_level="private")
    alias_person = create_person(repository, name="Another Private Name", public_name="人物A", privacy_level="public_alias")
    hidden_person = create_person(repository, name="Hidden Private Name", privacy_level="public_hidden")

    assert public_person_name(private_person, index=1) == ""
    assert public_person_name(alias_person, index=2) == "人物A"
    assert public_person_name(hidden_person, index=3) == ""


def test_public_report_does_not_include_private_person_display_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    create_person(repository, name="Private Real Name", public_name="人物A", privacy_level="public_alias")

    report = build_report(repository, ReportOptions(mode="public"))
    result = write_report(report, output_path=tmp_path / "report.md", save_json=True)
    text = result.markdown_path.read_text(encoding="utf-8")

    assert "Private Real Name" not in text
    assert "人物A" not in text


def test_unreviewed_person_label_is_not_qa_evidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    create_person(repository, name="Private Real Name", public_name="人物A", privacy_level="public_alias")

    result = route_query(repository, "誰が写っている？")

    assert "Private Real Name" not in result.answer
    assert "人物A" not in result.answer
