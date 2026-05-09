from __future__ import annotations

from pathlib import Path
import sys
import types

from personal_lifelog_rag.vlm.engines import FakeVlmEngine, NoopVlmEngine, Qwen3VlTransformersEngine, get_vlm_engine


def test_fake_vlm_engine_returns_safe_structured_result(tmp_path: Path) -> None:
    image_path = tmp_path / "dummy.jpg"
    image_path.write_bytes(b"not-used-by-fake")

    result = FakeVlmEngine().analyze_image(image_path, "prompt")

    assert result.status == "success"
    assert result.engine == "fake"
    assert "ramen_possible" in result.food_cues
    assert result.caption


def test_noop_vlm_engine_skips_without_crashing(tmp_path: Path) -> None:
    result = NoopVlmEngine().analyze_image(tmp_path / "missing.jpg", "prompt")

    assert result.status == "skipped"
    assert result.engine == "noop"


def test_ollama_engine_requires_localhost_url() -> None:
    try:
        get_vlm_engine("ollama", model_name="dummy")
    except ValueError:
        raise AssertionError("default Ollama URL should be localhost-only")


def test_qwen_transformers_uses_absolute_path_without_file_scheme(tmp_path: Path, monkeypatch) -> None:
    fake_transformers = _install_fake_qwen_transformers(monkeypatch)
    model_path = tmp_path / "model"
    model_path.mkdir()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-image-not-opened")

    engine = Qwen3VlTransformersEngine(model_name=None, model_path=str(model_path), device="cpu", local_files_only=True)
    result = engine.analyze_image(image_path, "Return JSON only.")

    assert result.status == "success"
    assert result.engine == "qwen3_vl_transformers"
    assert result.caption
    captured_image = fake_transformers.FakeProcessor.messages[0]["content"][0]["image"]
    assert captured_image == str(image_path.resolve())
    assert not captured_image.startswith("file://")
    assert result.raw["image_loading_mode"] == "path"
    assert fake_transformers.FakeModel.from_pretrained_kwargs["local_files_only"] is True


def test_qwen_transformers_falls_back_to_pil_image(tmp_path: Path, monkeypatch) -> None:
    fake_transformers = _install_fake_qwen_transformers(monkeypatch, fail_path_mode=True)
    model_path = tmp_path / "model"
    model_path.mkdir()
    image_path = tmp_path / "photo.jpg"
    from PIL import Image

    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(image_path)

    engine = Qwen3VlTransformersEngine(model_name=None, model_path=str(model_path), device="cpu", local_files_only=True)
    result = engine.analyze_image(image_path, "Return JSON only.")

    assert result.status == "success"
    assert result.raw["image_loading_mode"] == "pil"
    captured_image = fake_transformers.FakeProcessor.messages[-1]["content"][0]["image"]
    assert isinstance(captured_image, Image.Image)


def test_qwen_transformers_extracts_json_from_surrounding_text(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_transformers(
        monkeypatch,
        decoded_text='Here is the result:\n{"caption": "カフェのような場所の可能性があります", "short_caption": "カフェ候補", "food_cues": ["cafe_possible"], "confidence": 0.6}\nDone.',
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-image-not-opened")

    engine = Qwen3VlTransformersEngine(model_name=None, model_path=str(model_path), device="cpu", local_files_only=True)
    result = engine.analyze_image(image_path, "Return valid JSON only.")

    assert result.status == "success"
    assert result.caption == "カフェのような場所の可能性があります"
    assert "cafe_possible" in result.food_cues


def test_qwen_transformers_failed_json_saves_detailed_error(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_transformers(monkeypatch, decoded_text="I can see a vehicle interior, but this is not JSON.")
    model_path = tmp_path / "model"
    model_path.mkdir()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-image-not-opened")

    engine = Qwen3VlTransformersEngine(model_name=None, model_path=str(model_path), device="cpu", local_files_only=True)
    result = engine.analyze_image(image_path, "Return valid JSON only.")

    assert result.status == "failed"
    assert result.error_message
    assert "exception_class: ValueError" in result.error_message
    assert "exception_repr: ValueError" in result.error_message
    assert "raw_output_head:" in result.error_message
    assert "prompt_template:" in result.error_message
    assert "image_input_mode: path" in result.error_message
    assert "traceback_tail:" in result.error_message
    assert "json_parse_failed" in result.safety_flags


def test_qwen_transformers_runs_safety_filter(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_transformers(
        monkeypatch,
        decoded_text='{"caption": "彼女と楽しそうにご飯を食べている写真です", "short_caption": "彼女とご飯", "food_cues": ["meal_possible"], "confidence": 0.8}',
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-image-not-opened")

    engine = Qwen3VlTransformersEngine(model_name=None, model_path=str(model_path), device="cpu", local_files_only=True)
    result = engine.analyze_image(image_path, "Return valid JSON only.")

    assert result.status == "success"
    assert "彼女" not in (result.caption or "")
    assert "楽しそう" not in (result.caption or "")
    assert "relationship_inference_removed" in result.safety_flags
    assert "emotion_inference_removed" in result.safety_flags


def test_qwen_thinking_model_uses_processor_prefill_for_json(tmp_path: Path, monkeypatch) -> None:
    fake_transformers = _install_fake_qwen_transformers(
        monkeypatch,
        decoded_text='"caption": "車内の写真の可能性があります", "short_caption": "車内候補", "confidence": 0.5}',
    )
    model_path = tmp_path / "Qwen3-VL-8B-Thinking"
    model_path.mkdir()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-image-not-opened")

    engine = Qwen3VlTransformersEngine(model_name=None, model_path=str(model_path), device="cpu", local_files_only=True)
    result = engine.analyze_image(image_path, "Return valid JSON only.")

    assert result.status == "success"
    assert result.caption == "車内の写真の可能性があります"
    assert fake_transformers.FakeProcessor.template_kwargs["continue_final_message"] is True
    assert fake_transformers.FakeProcessor.template_kwargs["add_generation_prompt"] is False
    assert fake_transformers.FakeProcessor.messages[-1]["role"] == "assistant"
    assert fake_transformers.FakeProcessor.messages[-1]["content"] == "{"


def _install_fake_qwen_transformers(
    monkeypatch,
    *,
    fail_path_mode: bool = False,
    decoded_text: str = '{"caption": "駅の看板の可能性があります", "short_caption": "駅の看板", "confidence": 0.7}',
):
    module = types.ModuleType("transformers")

    class FakeInputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            self.device = device
            return self

    class FakeModel:
        from_pretrained_kwargs: dict = {}
        device = "cpu"

        @classmethod
        def from_pretrained(cls, model_ref, **kwargs):
            cls.model_ref = model_ref
            cls.from_pretrained_kwargs = kwargs
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return None

        def generate(self, **kwargs):
            return [[1, 2, 3, 4, 5]]

    class FakeProcessor:
        messages: list = []
        template_kwargs: dict = {}

        @classmethod
        def from_pretrained(cls, model_ref, **kwargs):
            cls.model_ref = model_ref
            cls.from_pretrained_kwargs = kwargs
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            self.__class__.messages.extend(messages)
            self.__class__.template_kwargs = kwargs
            image_content = messages[0]["content"][0]["image"]
            if fail_path_mode and isinstance(image_content, str):
                raise ValueError("path mode failed")
            assert "<think>" not in str(kwargs.get("chat_template") or "")
            assert kwargs["tokenize"] is True
            if kwargs.get("continue_final_message"):
                assert kwargs["add_generation_prompt"] is False
                assert messages[-1]["role"] == "assistant"
                assert messages[-1]["content"] == "{"
            else:
                assert kwargs["add_generation_prompt"] is True
            assert kwargs["return_dict"] is True
            assert kwargs["return_tensors"] == "pt"
            return FakeInputs({"input_ids": [[1, 2, 3]], "token_type_ids": [[0, 0, 0]]})

        def batch_decode(self, generated_ids, **kwargs):
            assert generated_ids == [[4, 5]]
            return [decoded_text]

    module.AutoModelForImageTextToText = FakeModel
    module.AutoProcessor = FakeProcessor
    FakeProcessor.chat_template = "{{- '<|im_start|>assistant\\n<think>\\n' }}"
    module.FakeModel = FakeModel
    module.FakeProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module
