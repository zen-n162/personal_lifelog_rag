from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app import cli as cli_module
from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.engines import Qwen3VlEmbeddingEngine


def test_qwen3_embedding_engine_reports_unavailable_without_runtime_deps(tmp_path: Path, monkeypatch) -> None:
    model_path = _make_model_dir(tmp_path)
    monkeypatch.setattr(
        "personal_lifelog_rag.embeddings.engines._has_module",
        lambda module_name: False,
    )

    engine = Qwen3VlEmbeddingEngine(model_name="Qwen/Qwen3-VL-Embedding-8B", model_path=str(model_path))

    assert engine.is_available() is False
    result = engine.embed_text("ご飯を食べた写真")
    assert result.status == "engine_unavailable"
    assert "missing local runtime dependencies" in str(result.error_message)


def test_qwen3_embedding_engine_uses_bundled_qwen_embedder_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = _make_model_dir(
        tmp_path,
        script_text="""
from pathlib import Path

class Qwen3VLEmbedder:
    def __init__(self, model_name_or_path, **kwargs):
        Path(model_name_or_path, "capture.json").write_text(
            __import__("json").dumps({"model": model_name_or_path, "local_files_only": kwargs.get("local_files_only")}),
            encoding="utf-8",
        )

    def process(self, inputs):
        return [[1.0, 2.0, 3.0, 4.0] for _ in inputs]
""",
    )
    monkeypatch.setattr(
        "personal_lifelog_rag.embeddings.engines._has_module",
        lambda module_name: module_name in {"torch", "transformers", "qwen_vl_utils"},
    )

    engine = Qwen3VlEmbeddingEngine(
        model_name="Qwen/Qwen3-VL-Embedding-8B",
        model_path=str(model_path),
        device="cpu",
        local_files_only=True,
        embedding_dim=3,
        batch_size=2,
    )

    text_result = engine.embed_text("新宿の写真")
    assert text_result.status == "success"
    assert text_result.embedding_dim == 3
    captured = json.loads((model_path / "capture.json").read_text(encoding="utf-8"))
    assert captured["model"] == str(model_path)
    assert captured["local_files_only"] is True

    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    image_result = engine.embed_image(image_path)
    assert image_result.status == "success"


def test_qwen3_embedding_engine_path_mode_uses_absolute_path(tmp_path: Path) -> None:
    model_path = _make_model_dir(tmp_path)
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class FakeRuntime:
        def __init__(self):
            self.inputs = []

        def process(self, inputs):
            self.inputs.append(inputs)
            return [[1.0, 0.0, 0.0, 0.0]]

    runtime = FakeRuntime()
    engine = Qwen3VlEmbeddingEngine(model_path=str(model_path))
    engine._runtime_model = runtime
    engine._runtime_kind = "bundled_qwen3_vl_embedding"

    result = engine.embed_image(image_path)

    assert result.status == "success"
    assert runtime.inputs[0] == [{"image": str(image_path.resolve())}]


def test_qwen3_embedding_engine_falls_back_to_pil_when_path_mode_fails(tmp_path: Path) -> None:
    model_path = _make_model_dir(tmp_path)
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class FakeRuntime:
        def __init__(self):
            self.inputs = []

        def process(self, inputs):
            self.inputs.append(inputs)
            image_value = inputs[0]["image"]
            if isinstance(image_value, str):
                raise ValueError("path mode failed")
            assert image_value.mode == "RGB"
            return [[1.0, 0.0, 0.0, 0.0]]

    runtime = FakeRuntime()
    engine = Qwen3VlEmbeddingEngine(model_path=str(model_path))
    engine._runtime_model = runtime
    engine._runtime_kind = "bundled_qwen3_vl_embedding"

    result = engine.embed_image(image_path)

    assert result.status == "success"
    assert len(runtime.inputs) == 2


def test_qwen3_embedding_engine_text_embedding_adds_retrieval_instruction(tmp_path: Path) -> None:
    model_path = _make_model_dir(tmp_path)

    class FakeRuntime:
        def __init__(self):
            self.inputs = []

        def process(self, inputs):
            self.inputs.append(inputs)
            return [[0.0, 1.0, 0.0, 0.0]]

    runtime = FakeRuntime()
    engine = Qwen3VlEmbeddingEngine(model_path=str(model_path))
    engine._runtime_model = runtime
    engine._runtime_kind = "bundled_qwen3_vl_embedding"

    result = engine.embed_text("ご飯を食べた写真")

    assert result.status == "success"
    assert runtime.inputs[0][0]["text"] == "ご飯を食べた写真"
    assert "Retrieve images relevant" in runtime.inputs[0][0]["instruction"]


def test_build_image_embeddings_config_passes_runtime_fields(tmp_path: Path, monkeypatch) -> None:
    db_path = _seed_image_db(tmp_path)
    model_path = _make_model_dir(tmp_path)
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        f"""
models:
  multimodal_embedding:
    engine: qwen3_vl_embedding
    model_name: Qwen/Qwen3-VL-Embedding-8B
    model_path: "{model_path}"
    device: cpu
    dtype: bfloat16
    local_files_only: true
    embedding_dim: 4096
    batch_size: 4
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_get_engine(engine_name, **kwargs):
        captured["engine_name"] = engine_name
        captured.update(kwargs)
        return Qwen3VlEmbeddingEngine(
            model_name=kwargs.get("model_name"),
            model_path=kwargs.get("model_path"),
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
            local_files_only=kwargs.get("local_files_only"),
            embedding_dim=kwargs.get("embedding_dim"),
            batch_size=kwargs.get("batch_size"),
        )

    monkeypatch.setattr(cli_module, "get_multimodal_embedding_engine", fake_get_engine)

    code = main(
        [
            "--db-path",
            str(db_path),
            "build-image-embeddings",
            "--date",
            "2024-12-24",
            "--config",
            str(config_path),
            "--dry-run",
        ]
    )

    assert code == 0
    assert captured["engine_name"] == "qwen3_vl_embedding"
    assert captured["model_path"] == str(model_path)
    assert captured["device"] == "cpu"
    assert captured["dtype"] == "bfloat16"
    assert captured["local_files_only"] is True
    assert captured["embedding_dim"] == 4096
    assert captured["batch_size"] == 4


def test_build_image_embeddings_cli_engine_overrides_config(tmp_path: Path, monkeypatch) -> None:
    db_path = _seed_image_db(tmp_path)
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        """
models:
  multimodal_embedding:
    engine: qwen3_vl_embedding
    model_path: "/tmp/not-used"
    local_files_only: true
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_get_engine(engine_name, **kwargs):
        captured["engine_name"] = engine_name
        captured.update(kwargs)
        from personal_lifelog_rag.embeddings.engines import FakeEmbeddingEngine

        return FakeEmbeddingEngine(model_name="fake-qwen3-vl-embedding")

    monkeypatch.setattr(cli_module, "get_multimodal_embedding_engine", fake_get_engine)

    code = main(
        [
            "--db-path",
            str(db_path),
            "build-image-embeddings",
            "--date",
            "2024-12-24",
            "--config",
            str(config_path),
            "--engine",
            "fake",
            "--dry-run",
        ]
    )

    assert code == 0
    assert captured["engine_name"] == "fake"


def _make_model_dir(tmp_path: Path, *, script_text: str = "") -> Path:
    model_path = tmp_path / "models" / "Qwen3-VL-Embedding-8B"
    (model_path / "scripts").mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "scripts" / "qwen3_vl_embedding.py").write_text(script_text, encoding="utf-8")
    return model_path


def _seed_image_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    repository.add_media_item(
        id="media_qwen_config",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-media-qwen-config",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return db_path
