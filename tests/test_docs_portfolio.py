from __future__ import annotations

from pathlib import Path


PORTFOLIO_DOCS = [
    "docs/portfolio_summary.md",
    "docs/system_architecture.md",
    "docs/demo_scenarios.md",
    "docs/privacy_and_safety.md",
    "docs/evaluation_summary.md",
    "docs/roadmap.md",
]


def test_portfolio_docs_exist_and_are_linked_from_readme() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for doc in PORTFOLIO_DOCS:
        path = Path(doc)
        assert path.exists(), doc
        assert doc in readme


def test_portfolio_docs_do_not_include_private_paths_or_bulk_private_artifacts() -> None:
    forbidden = [
        "data/raw",
        "private_config/",
        "private_config/model_runtime.yaml",
        "/home/",
        "raw database path",
    ]
    combined = "\n".join(Path(doc).read_text(encoding="utf-8") for doc in PORTFOLIO_DOCS)
    for phrase in forbidden:
        assert phrase not in combined


def test_portfolio_summary_contains_job_hunting_description() -> None:
    text = Path("docs/portfolio_summary.md").read_text(encoding="utf-8")
    assert "privacy-preserving multimodal RAG" in text
    assert "portfolio description" in text


def test_architecture_doc_contains_mermaid_flowchart() -> None:
    text = Path("docs/system_architecture.md").read_text(encoding="utf-8")
    assert "```mermaid" in text
    assert "Qwen3-VL" in text
    assert "Qwen3-VL-Embedding" in text

