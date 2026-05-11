from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.reporting.portfolio_html import (
    PortfolioHtmlOptions,
    build_portfolio_html,
    check_public_portfolio_path,
)


def test_build_portfolio_html_generates_single_file_and_build_json(tmp_path: Path) -> None:
    source = tmp_path / "evaluation_summary.md"
    source.write_text(
        "\n".join(
            [
                "# Evaluation Summary",
                "pytest: 384 passed",
                "db-check --strict: ok",
                "cases: 17",
                "passed: 17",
                "failed: 0",
                "media_vlm total: 400",
                "success: 389",
                "failed: 11",
                "engine_unavailable: 0",
                "media_embeddings total: 471",
                "success: 471",
                "embedding_dim: 4096",
                "media_ocr total: 10",
                "success: 5",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "portfolio_public.html"

    payload = build_portfolio_html(
        PortfolioHtmlOptions(
            output_html=output,
            source_files=(source,),
            check_privacy=True,
            force=True,
        )
    )

    html = output.read_text(encoding="utf-8")
    build_json = output.with_name("portfolio_public_build.json")
    build_payload = json.loads(build_json.read_text(encoding="utf-8"))

    assert output.exists()
    assert build_json.exists()
    assert payload["privacy_check_passed"] is True
    assert build_payload["metrics"]["pytest"] == "384 passed"
    assert "Personal Lifelog RAG" in html
    assert "Architecture overview" in html
    assert "Data model overview" in html
    assert "Demo scenarios" in html
    assert "Evaluation summary" in html
    assert "Privacy and safety design" in html
    assert "Roadmap" in html
    assert "https://cdn" not in html
    assert "share=True" not in html
    assert "private_config" not in html
    assert "data/raw" not in html
    assert "/home/zennakamura" not in html
    assert check_public_portfolio_path(output)["passed"] is True
