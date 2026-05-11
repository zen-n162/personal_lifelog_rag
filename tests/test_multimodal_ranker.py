from __future__ import annotations

from personal_lifelog_rag.retrieval.evidence_strength import confidence_label_for_score, compute_evidence_strength
from personal_lifelog_rag.retrieval.multimodal_ranker import score_multimodal_components
from personal_lifelog_rag.retrieval.visual_query_expansion import expand_visual_query_terms


def test_vlm_only_score_is_capped_below_high_confidence() -> None:
    terms = expand_visual_query_terms("ご飯を食べた写真")
    components = score_multimodal_components(
        {"food_cues_json": '["meal_possible", "rice_possible"]'},
        expanded_terms=terms,
        embedding_score=0.0,
        related_event=None,
        line_matches=[],
    )

    assert components["vlm_text_score"] > 0
    assert components["final_score"] <= 0.44
    assert confidence_label_for_score(0.9, evidence_types=["photo", "vlm"]) == "中"


def test_score_uses_embedding_vlm_ocr_line_event_components() -> None:
    terms = expand_visual_query_terms("ご飯を食べた写真")
    components = score_multimodal_components(
        {"food_cues_json": '["meal_possible"]', "ocr_text": "ご飯 メニュー"},
        expanded_terms=terms,
        embedding_score=0.8,
        related_event={"title": "食事・カフェの可能性", "summary": "ご飯", "confidence": 0.8},
        line_matches=[{"text": "ご飯おいしかった"}],
    )

    assert components["embedding_score"] == 0.8
    assert components["vlm_text_score"] > 0
    assert components["ocr_score"] > 0
    assert components["line_score"] > 0
    assert components["event_score"] > 0
    assert components["final_score"] > 0.45


def test_visual_mismatch_caps_line_and_event_boosts() -> None:
    terms = expand_visual_query_terms("ダンスの写真")
    components = score_multimodal_components(
        {"caption": "Black-and-white photo collage of children"},
        expanded_terms=terms,
        embedding_score=0.27,
        related_event={"title": "食事・カフェの可能性", "summary": "同日イベント", "confidence": 0.95},
        line_matches=[{"text": "ダンスの話"} for _ in range(5)],
    )

    assert components["visual_match"] == 0.0
    assert components["line_score"] <= 0.2
    assert components["event_score"] <= 0.2
    assert components["final_score"] < 0.3


def test_visual_match_allows_context_boosts() -> None:
    terms = expand_visual_query_terms("ダンスの写真")
    components = score_multimodal_components(
        {"activity_tags_json": '["dancing_possible"]'},
        expanded_terms=terms,
        embedding_score=0.2,
        related_event={"title": "写真撮影の記録", "summary": "同日イベント", "confidence": 0.8},
        line_matches=[{"text": "ダンスの話"}],
    )

    assert components["visual_match"] == 1.0
    assert components["line_score"] > 0.2
    assert components["event_score"] > 0.2
    assert components["final_score"] > 0.2


def test_specific_food_generic_only_match_stays_weak() -> None:
    terms = expand_visual_query_terms("ラーメンの写真")
    components = score_multimodal_components(
        {"caption": "A bowl of food on a restaurant table", "food_cues_json": '["meal_possible"]'},
        expanded_terms=terms,
        embedding_score=0.2,
        related_event={"title": "ラーメンの話", "summary": "同日イベント", "confidence": 0.95},
        line_matches=[{"text": "ラーメン食べたい"} for _ in range(5)],
        sql_score=0.8,
        matched_terms=["food", "bowl", "meal"],
        query_intent="specific_food_search",
        specific_terms=["ラーメン", "ramen", "ramen noodle", "noodle soup", "chashu", "soft-boiled egg"],
        generic_terms=["food", "meal", "bowl", "dish", "cafe", "restaurant", "rice", "noodle", "soup"],
    )

    assert components["specific_food_score"] == 0.0
    assert components["generic_food_score"] > 0
    assert components["specific_visual_match"] == 0.0
    assert components["visual_match"] == 0.0
    assert components["line_score"] <= 0.2
    assert components["event_score"] <= 0.2


def test_specific_food_match_gets_bonus() -> None:
    terms = expand_visual_query_terms("オムライスの写真")
    generic = score_multimodal_components(
        {"caption": "A rice dish on a table", "food_cues_json": '["meal_possible"]'},
        expanded_terms=terms,
        embedding_score=0.1,
        related_event=None,
        line_matches=[],
        sql_score=0.5,
        query_intent="specific_food_search",
        specific_terms=["オムライス", "omurice", "omelette rice", "fried rice with omelette"],
        generic_terms=["food", "meal", "bowl", "dish", "rice"],
    )
    specific = score_multimodal_components(
        {"caption": "Omelette rice with ketchup sauce", "food_cues_json": '["omurice"]'},
        expanded_terms=terms,
        embedding_score=0.1,
        related_event=None,
        line_matches=[],
        sql_score=0.5,
        query_intent="specific_food_search",
        specific_terms=["オムライス", "omurice", "omelette rice", "fried rice with omelette"],
        generic_terms=["food", "meal", "bowl", "dish", "rice"],
    )

    assert specific["specific_food_score"] > 0
    assert specific["specific_visual_match"] == 1.0
    assert specific["final_score"] > generic["final_score"]


def test_evidence_strength_rules_for_pr35() -> None:
    assert compute_evidence_strength(["photo", "vlm"]) == "weak"
    assert compute_evidence_strength(["photo", "embedding"]) == "weak"
    assert compute_evidence_strength(["photo", "vlm", "event"]) == "medium"
    assert compute_evidence_strength(["photo", "vlm", "ocr", "line", "event"]) == "strong"
    assert compute_evidence_strength(["photo", "embedding", "vlm", "ocr", "event"]) == "strong"
