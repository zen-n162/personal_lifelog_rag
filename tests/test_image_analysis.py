import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.captioning.image_analysis import analyze_images
from personal_lifelog_rag.captioning.local_vlm import UnconfiguredVLMAdapter, VLMAnalysisResult
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ocr.local_ocr import OCRResult, UnconfiguredOCRAdapter


class FakeOCRAdapter:
    engine = "fake-ocr"
    available = True

    def extract_text(self, image_path):
        return OCRResult(text="看板の文字", engine=self.engine)


class FakeVLMAdapter:
    engine = "fake-vlm"
    available = True

    def analyze_image(self, image_path, *, ocr_text=None):
        return VLMAnalysisResult(
            engine=self.engine,
            caption="駅前で撮った写真",
            scene="駅前",
            objects=("sign", "building"),
            possible_activity="待ち合わせの可能性",
            text_in_image=ocr_text,
        )


def test_analyze_images_skips_safely_when_models_are_unconfigured(tmp_path):
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_dummy",
        file_path=str(tmp_path / "missing.jpg"),
        file_name="missing.jpg",
        file_hash="dummy-hash",
        media_type="image",
    )

    report = analyze_images(
        repository,
        ocr_adapter=UnconfiguredOCRAdapter(),
        vlm_adapter=UnconfiguredVLMAdapter(),
    )

    assert report.scanned == 1
    assert report.updated == 0
    assert report.skipped == 1
    assert report.reason is not None


def test_analyze_images_persists_caption_ocr_and_analysis_json(tmp_path):
    image_path = tmp_path / "dummy.jpg"
    image_path.write_bytes(b"dummy local image placeholder")
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_dummy",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="dummy-hash",
        media_type="image",
    )

    report = analyze_images(
        repository,
        ocr_adapter=FakeOCRAdapter(),
        vlm_adapter=FakeVLMAdapter(),
    )

    row = repository.list_media_items(limit=1)[0]
    analysis = json.loads(row["analysis_json"])
    assert report.updated == 1
    assert row["caption"] == "駅前で撮った写真"
    assert row["ocr_text"] == "看板の文字"
    assert analysis["scene"] == "駅前"
    assert analysis["objects"] == ["sign", "building"]
    assert analysis["possible_activity"] == "待ち合わせの可能性"
    assert analysis["text_in_image"] == "看板の文字"


def test_cli_analyze_images_reports_safe_skip(tmp_path, capsys):
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_dummy",
        file_path=str(tmp_path / "missing.jpg"),
        file_name="missing.jpg",
        file_hash="dummy-hash",
        media_type="image",
    )

    exit_code = main(["--db-path", str(db_path), "analyze-images", "--limit", "100"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "未解析" in captured.out
    assert "1 file(s)" in captured.out

