from __future__ import annotations

from pathlib import Path


RELEASE_AND_CAREER_DOCS = [
    "RELEASE_NOTES.md",
    "docs/releases/v0.1.md",
    "docs/reproducibility.md",
    "docs/ui_review_workflow.md",
    "docs/final_publication_checklist.md",
    "docs/job_hunting_pitch.md",
    "docs/technical_interview_notes.md",
    "docs/ml_learning_takeaways.md",
    "docs/es_self_pr_examples.md",
    "docs/demo_script_for_interview.md",
]


def test_pr56_63_docs_exist_and_are_linked_from_readme() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for doc in RELEASE_AND_CAREER_DOCS:
        path = Path(doc)
        assert path.exists(), doc
        if doc.startswith("docs/"):
            assert doc in readme


def test_pr56_63_docs_avoid_private_paths_and_raw_artifact_markers() -> None:
    forbidden = [
        "/home/zennakamura",
        "data/raw",
        "private_config/",
        "file://",
        ".sqlite",
        "share=True",
        "line_9f",
    ]
    combined = "\n".join(Path(doc).read_text(encoding="utf-8") for doc in RELEASE_AND_CAREER_DOCS)
    for phrase in forbidden:
        assert phrase not in combined


def test_job_hunting_docs_contain_required_pitch_lengths() -> None:
    pitch = Path("docs/job_hunting_pitch.md").read_text(encoding="utf-8")
    es = Path("docs/es_self_pr_examples.md").read_text(encoding="utf-8")
    assert "30-Second Pitch" in pitch
    assert "1-Minute Pitch" in pitch
    assert "3-Minute Pitch" in pitch
    for label in ("100 Characters", "200 Characters", "400 Characters", "800 Characters"):
        assert label in es
