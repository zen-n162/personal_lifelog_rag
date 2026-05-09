from __future__ import annotations

from personal_lifelog_rag.reporting.examples import build_example_queries
from personal_lifelog_rag.reporting.redaction import ReportRedactor


def test_examples_are_generic_and_evidence_based() -> None:
    examples = build_example_queries(
        {
            "event_stats": {
                "evidence_type_counts": {"line": 2, "photo": 1},
                "modality_counts": {"photo_and_line": 1},
                "monthly_event_counts": {"2024-12": 3},
            },
            "embedding_stats": {"total": 1},
            "call_stats": {"total": 1},
        },
        redactor=ReportRedactor(public=True),
    )

    text = "\n".join(example["query"] + example["result_summary"] for example in examples)
    assert "PLACE_1" in text
    assert "/home/" not in text
    assert "35." not in text
