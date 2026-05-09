from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag import env_check
from personal_lifelog_rag.env_check import run_env_check


def test_env_check_reports_repo_venv_warning(tmp_path: Path, monkeypatch) -> None:
    fake_python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(env_check.sys, "executable", str(fake_python))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))

    report = run_env_check()

    assert report["using_repo_venv_python"] is True
    assert any(".venv" in warning for warning in report["warnings"])


def test_env_check_cli_outputs_model_paths(tmp_path: Path, capsys) -> None:
    vlm_path = tmp_path / "models" / "vlm"
    embedding_path = tmp_path / "models" / "embedding"
    vlm_path.mkdir(parents=True)
    embedding_path.mkdir(parents=True)
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        f"""
models:
  vlm:
    model_path: "{vlm_path}"
  multimodal_embedding:
    model_path: "{embedding_path}"
""",
        encoding="utf-8",
    )

    code = main(["env-check", "--config", str(config_path)])
    output = capsys.readouterr().out

    assert code == 0
    assert "Environment check" in output
    assert str(vlm_path) in output
    assert str(embedding_path) in output
