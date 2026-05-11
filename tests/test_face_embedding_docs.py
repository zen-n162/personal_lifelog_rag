from __future__ import annotations

from pathlib import Path


def test_face_embedding_doc_exists_and_is_linked() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/face_embedding_clustering.md")
    people_doc = Path("docs/face_review_people.md")

    assert doc.exists()
    assert people_doc.exists()
    assert "docs/face_embedding_clustering.md" in readme
    assert "docs/face_review_people.md" in readme


def test_face_embedding_doc_avoids_private_paths() -> None:
    text = Path("docs/face_embedding_clustering.md").read_text(encoding="utf-8")
    text += "\n" + Path("docs/face_review_people.md").read_text(encoding="utf-8")
    forbidden = ["/home/", "data/raw", "file://"]
    for phrase in forbidden:
        assert phrase not in text
