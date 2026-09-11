"""图片接入测试：质量、EXIF、预处理、OCR 适配与感知哈希。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
pytest.importorskip("cv2")
pytest.importorskip("imagehash")

from typer.testing import CliRunner  # noqa: E402

from scoreproof.cli import app  # noqa: E402
from scoreproof.errors import DataSourceError  # noqa: E402
from scoreproof.ingest.image_loader import (  # noqa: E402
    OcrResult,
    check_quality,
    phash,
    phash_distance,
    preprocess,
    run_ocr,
)


def _make_text_image(path: Path, *, size: tuple[int, int] = (800, 500)) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, size[0] - 20, size[1] - 20), outline="black", width=8)
    for y in range(80, size[1] - 40, 45):
        draw.line((80, y, size[0] - 80, y), fill="black", width=5)
    image.save(path)
    return path


class TestQuality:
    def test_sharp_image(self, tmp_path: Path) -> None:
        path = _make_text_image(tmp_path / "sharp.png")
        result = check_quality(path, blurry_threshold=20)
        assert result.width == 800 and result.height == 500
        assert result.sharpness is not None and result.sharpness > 20
        assert result.is_blurry is False
        assert result.needs_retake is False

    def test_flat_image_is_blurry_and_small(self, tmp_path: Path) -> None:
        path = tmp_path / "flat.png"
        Image.new("RGB", (320, 240), "white").save(path)
        result = check_quality(path)
        assert result.is_blurry is True
        assert result.needs_retake is True
        assert any(note.startswith("分辨率不足") for note in result.notes)

    def test_exif_orientation_is_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "rotated.jpg"
        image = Image.new("RGB", (80, 40), "white")
        exif = image.getexif()
        exif[274] = 6  # 展示时需顺时针旋转 90°
        image.save(path, exif=exif)
        result = check_quality(path, min_width=1, min_height=1)
        assert (result.width, result.height) == (40, 80)
        assert result.rotation_applied == 90

    @pytest.mark.parametrize("value", [-1, 0])
    def test_bad_threshold_rejected(self, tmp_path: Path, value: int) -> None:
        path = _make_text_image(tmp_path / "input.png")
        with pytest.raises(ValueError):
            check_quality(path, min_width=value)


class TestPreprocess:
    def test_writes_new_png_without_overwriting_source(self, tmp_path: Path) -> None:
        source = _make_text_image(tmp_path / "原图.png")
        original = source.read_bytes()
        out = preprocess(
            source,
            out_dir=tmp_path / "处理结果",
            correct_perspective=False,
            denoise=False,
            enhance_contrast=False,
        )
        assert out.exists() and out.suffix == ".png" and out != source
        assert source.read_bytes() == original
        with Image.open(out) as image:
            assert image.size == (800, 500)

    def test_default_output_directory(self, tmp_path: Path) -> None:
        source = _make_text_image(tmp_path / "input.png")
        out = preprocess(
            source, correct_perspective=False, denoise=False, enhance_contrast=False
        )
        assert out.parent == tmp_path / "processed"


class FakeOcrEngine:
    def __init__(self, result: Any) -> None:
        self.result = result

    def __call__(self, _: str) -> Any:
        return self.result


class BrokenOcrEngine:
    def __call__(self, _: str) -> Any:
        raise RuntimeError("engine failed")


class TestOcr:
    def test_parses_text_bbox_confidence_and_elapsed(self, tmp_path: Path) -> None:
        source = _make_text_image(tmp_path / "cert.png")
        raw = (
            [
                [[[0, 0], [100, 0], [100, 20], [0, 20]], "荣誉证书", 0.98],
                [[[5, 30], [80, 30], [80, 50], [5, 50]], "学生001", 1.2],
                [None, "", 0.5],
                ["bad"],
            ],
            [0.1, 0.2, 0.3],
        )
        result = run_ocr(source, engine=FakeOcrEngine(raw))
        assert isinstance(result, OcrResult)
        assert result.text == "荣誉证书\n学生001"
        assert result.lines[0].bbox == (0.0, 0.0, 100.0, 20.0)
        assert result.lines[1].confidence == 1.0  # 防御性截断到 [0,1]
        assert result.elapsed_seconds == pytest.approx(0.6)

    def test_empty_result(self, tmp_path: Path) -> None:
        source = _make_text_image(tmp_path / "empty.png")
        result = run_ocr(source, engine=FakeOcrEngine((None, 0.1)))
        assert result.text == "" and result.mean_confidence == 0.0

    def test_engine_error_is_domain_error(self, tmp_path: Path) -> None:
        source = _make_text_image(tmp_path / "error.png")
        with pytest.raises(DataSourceError, match="OCR 识别失败"):
            run_ocr(source, engine=BrokenOcrEngine())


class TestPerceptualHash:
    def test_same_image_has_same_hash(self, tmp_path: Path) -> None:
        first = _make_text_image(tmp_path / "first.png")
        second = tmp_path / "second.png"
        second.write_bytes(first.read_bytes())
        left, right = phash(first), phash(second)
        assert left == right and len(left) == 16
        assert phash_distance(left, right) == 0

    def test_changed_image_has_nonzero_distance(self, tmp_path: Path) -> None:
        first = _make_text_image(tmp_path / "first.png")
        second = _make_text_image(tmp_path / "second.png")
        with Image.open(second) as image:
            draw = ImageDraw.Draw(image)
            draw.rectangle((300, 150, 500, 350), fill="black")
            image.save(second)
        assert phash_distance(phash(first), phash(second)) > 0

    @pytest.mark.parametrize(("left", "right"), [("", "aa"), ("aa", "bbb"), ("zz", "aa")])
    def test_invalid_hashes_rejected(self, left: str, right: str) -> None:
        with pytest.raises(ValueError):
            phash_distance(left, right)


class TestInputErrors:
    @pytest.mark.parametrize("function", [check_quality, preprocess, run_ocr, phash])
    def test_missing_file(self, tmp_path: Path, function: Any) -> None:
        with pytest.raises(DataSourceError, match="不存在"):
            function(tmp_path / "missing.png")

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "fake.txt"
        path.write_text("not an image", encoding="utf-8")
        with pytest.raises(DataSourceError, match="不支持"):
            check_quality(path)


class TestImageCli:
    def test_parse_image_without_ocr(self, tmp_path: Path) -> None:
        source = _make_text_image(tmp_path / "cli.png")
        result = CliRunner().invoke(app, ["parse-image", str(source), "--no-ocr"])
        assert result.exit_code == 0, result.output
        assert '"phash"' in result.output
        assert '"ocr": null' in result.output
