from __future__ import annotations

from personal_lifelog_rag.retrieval.query_intent import infer_query_intent


def test_query_intent_place_visit() -> None:
    assert infer_query_intent("新宿に行ったのはいつ？") == "place_visit"


def test_query_intent_food_activity() -> None:
    assert infer_query_intent("ご飯を食べたのはいつ？") == "food_activity"


def test_query_intent_call_activity() -> None:
    assert infer_query_intent("通話した日は？") == "call_activity"


def test_query_intent_topic_mention() -> None:
    assert infer_query_intent("アルバム") == "topic_mention"


def test_query_intent_override() -> None:
    assert infer_query_intent("新宿", override="place_visit") == "place_visit"

