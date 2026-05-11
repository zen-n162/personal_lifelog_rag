from __future__ import annotations

from personal_lifelog_rag.evaluation.private_eval import (
    PERSON_QA_OVERCLAIM_TERMS,
    _forbidden_claims_found,
)


def test_person_overclaim_terms_are_detected() -> None:
    text = "顔認証で確定し、確実に一緒にいたと断定しました。"

    found = _forbidden_claims_found(text, list(PERSON_QA_OVERCLAIM_TERMS))

    assert "顔認証で確定" in found
    assert "確実に一緒にいた" in found
