from __future__ import annotations

from personal_lifelog_rag.places.location_store import public_place_label


def test_private_place_public_label_hides_specific_display_name() -> None:
    place = {
        "display_name": "具体的な自宅名",
        "public_name": None,
        "category": "home",
        "privacy_level": "private",
    }

    label = public_place_label(place)

    assert label == "非公開の場所"
    assert "具体的な自宅名" not in label


def test_public_label_place_uses_public_name() -> None:
    place = {
        "display_name": "実際の店名",
        "public_name": "カフェ周辺",
        "category": "cafe",
        "privacy_level": "public_label",
    }

    assert public_place_label(place) == "カフェ周辺"
