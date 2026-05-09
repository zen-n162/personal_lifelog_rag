from __future__ import annotations

from personal_lifelog_rag.retrieval.visual_query_expansion import expand_visual_query_terms


def test_gohan_expands_to_vlm_food_terms() -> None:
    terms = expand_visual_query_terms("ご飯を食べた写真")

    assert "ご飯" in terms
    assert "meal_possible" in terms
    assert "rice_possible" in terms
    assert "food_cues" in terms


def test_visual_query_expands_place_cafe_and_car_terms() -> None:
    cafe_terms = expand_visual_query_terms("カフェっぽい写真")
    shinjuku_terms = expand_visual_query_terms("新宿の写真")
    car_terms = expand_visual_query_terms("車内の写真")

    assert "cafe_possible" in cafe_terms
    assert "glass_cup" in cafe_terms
    assert "shinjuku" in shinjuku_terms
    assert "urban" in shinjuku_terms
    assert "vehicle_interior_possible" in car_terms
    assert "headrest" in car_terms
