from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.privacy_controls import privacy_audit


def test_privacy_audit_detects_public_artifact_patterns(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    bad_html = tmp_path / "bad.html"
    bad_html.write_text("<html>/home/example data/raw file://x</html>", encoding="utf-8")

    report = privacy_audit(repository, public=True, public_paths=[bad_html], log_action=False)

    assert report["passed"] is False
    assert report["issue_count"] >= 1


def test_privacy_audit_passes_safe_public_artifact(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    safe_html = tmp_path / "safe.html"
    safe_html.write_text("<html><body>Public aggregate portfolio</body></html>", encoding="utf-8")

    report = privacy_audit(repository, public=True, public_paths=[safe_html], log_action=False)

    assert report["passed"] is True
    assert report["issue_count"] == 0
