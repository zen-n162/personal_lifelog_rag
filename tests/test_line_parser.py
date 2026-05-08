from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ingest.line_parser import (
    classify_message_type,
    derive_chat_id,
    generate_message_id,
    parse_line_chat_file,
    parse_line_chat_file_with_warnings,
)


def test_parse_line_chat_fixture() -> None:
    fixture = Path("tests/fixtures/line/sample_chat.txt")

    messages = parse_line_chat_file(fixture, chat_name="Dummy Chat")

    assert len(messages) == 9
    assert messages[0].sent_at == "2024-12-24T17:30:00+09:00"
    assert messages[0].sender_name == "自分"
    assert messages[0].message_text == "18時に新宿着く！"
    assert messages[0].chat_name == "Dummy Chat"
    assert messages[0].chat_id == derive_chat_id(fixture)
    assert messages[0].source_file == "sample_chat.txt"
    assert messages[0].message_id == generate_message_id(
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )


def test_multiline_message_is_joined_to_previous_message() -> None:
    fixture = Path("tests/fixtures/line/sample_chat.txt")

    messages = parse_line_chat_file(fixture)

    assert messages[2].message_text == "今日のご飯おいしかったね\nこのあと少し散歩した"
    assert messages[2].message_type == "text"


def test_special_messages_are_classified() -> None:
    assert classify_message_type("[写真]") == "image"
    assert classify_message_type("[動画]") == "video"
    assert classify_message_type("[スタンプ]") == "sticker"
    assert classify_message_type("[ファイル] receipt.pdf") == "file"
    assert classify_message_type("相手が参加しました", sender_name="LINE") == "system"
    assert classify_message_type("[未対応]") == "unknown"


def test_unparsed_lines_are_kept_as_warnings() -> None:
    fixture = Path("tests/fixtures/line/sample_chat.txt")

    result = parse_line_chat_file_with_warnings(fixture)

    assert len(result.warnings) == 2
    assert all(warning.reason for warning in result.warnings)


def test_ingest_line_cli_saves_fixture_and_is_idempotent(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    fixture_dir = Path("tests/fixtures/line")

    assert main(["--db-path", str(db_path), "ingest-line", "--path", str(fixture_dir)]) == 0
    first_output = capsys.readouterr().out
    assert "Imported LINE messages: 9 new, 0 duplicate, 2 warning(s), 1 file(s)" in first_output
    assert "18時に新宿着く" not in first_output

    assert main(["--db-path", str(db_path), "ingest-line", "--path", str(fixture_dir)]) == 0
    second_output = capsys.readouterr().out
    assert "Imported LINE messages: 0 new, 9 duplicate, 2 warning(s), 1 file(s)" in second_output

    repository = LifelogRepository(db_path)
    assert repository.stats()["line_messages"] == 9
    rows = repository.list_line_messages(limit=20)
    assert rows[0]["chat_id"] == derive_chat_id(Path("tests/fixtures/line/sample_chat.txt"))
    assert rows[0]["id"] == generate_message_id(
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
