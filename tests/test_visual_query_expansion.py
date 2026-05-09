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


def test_visual_query_expands_performance_stage_and_dance_terms() -> None:
    performance_terms = expand_visual_query_terms("パフォーマンスっぽい写真")
    stage_terms = expand_visual_query_terms("ステージの写真")
    dance_terms = expand_visual_query_terms("ダンスの写真")

    assert "performance_possible" in performance_terms
    assert "performing_possible" in performance_terms
    assert "stage_possible" in stage_terms
    assert "performance_venue_possible" in stage_terms
    assert "dancing_possible" in dance_terms


def test_visual_query_expands_ocr_document_terms() -> None:
    receipt_terms = expand_visual_query_terms("レシートの写真")
    menu_terms = expand_visual_query_terms("メニューが写っている写真")
    sign_terms = expand_visual_query_terms("看板に書いてある写真")

    assert "receipt" in receipt_terms
    assert "ocr_text" in receipt_terms
    assert "restaurant_menu" in menu_terms
    assert "signboard" in sign_terms
