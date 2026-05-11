from __future__ import annotations

from personal_lifelog_rag.reporting.portfolio_html import check_public_portfolio_text, format_safety_report


def test_public_portfolio_safety_detects_dangerous_patterns() -> None:
    report = check_public_portfolio_text(
        "\n".join(
            [
                "<p>/home/zennakamura/private.jpg</p>",
                "<p>private_config/model_runtime.yaml</p>",
                "<p>data/raw/photos/sample.jpg</p>",
                "<p>contact: user@example.com</p>",
                "<p>share=True</p>",
            ]
        ),
        file_name="sample.html",
    )

    assert report["passed"] is False
    patterns = {issue["pattern"] for issue in report["issues"]}
    assert {"home_path", "private_config", "raw_data_path", "email", "share_true"} <= patterns
    assert "FAIL" in format_safety_report(report)


def test_public_portfolio_safety_passes_public_summary() -> None:
    report = check_public_portfolio_text(
        "<h1>Personal Lifelog RAG</h1><p>Local-first aggregate public report without raw evidence.</p>",
        file_name="safe.html",
    )

    assert report["passed"] is True
    assert format_safety_report(report).startswith("PASS")
