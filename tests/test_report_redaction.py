from __future__ import annotations

from personal_lifelog_rag.reporting.redaction import ReportRedactor


def test_public_redaction_hides_gps_paths_media_ids_and_contacts() -> None:
    redactor = ReportRedactor(public=True)

    text = redactor.text(
        "/home/user/private/photo.jpg media_abcdef1234567890 35.123456, 139.123456 "
        "mail@example.com 090-1234-5678",
        max_chars=200,
    )

    assert "[PATH]" in text
    assert "MEDIA_ID_REDACTED" in text
    assert "[GPS]" in text
    assert "[EMAIL]" in text
    assert "[PHONE]" in text


def test_public_redaction_maps_people_senders_and_places() -> None:
    redactor = ReportRedactor(public=True)

    assert redactor.person("いおり") == "PERSON_1"
    assert redactor.person("いおり") == "PERSON_1"
    assert redactor.sender("ぜん") == "SENDER_1"
    assert redactor.place("新宿駅") == "PLACE_1"
    assert redactor.place("自宅周辺", sensitive=True) == "SENSITIVE_PLACE"

