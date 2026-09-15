"""图片接入：质量检测、预处理、OCR 与感知哈希。

本模块负责本地、确定性的图像接入；文本 LLM 字段抽取、字段校验与 VLM
触发决策由 ``scoreproof.evidence.certificate`` 实现并在本模块保留兼容入口。
实现顺序（与项目总结第 4.2 节一致）：

    图片
     ↓ ① 预处理：EXIF 旋转、去噪、透视矫正、清晰度检查（模糊→提示重拍）
     ↓ ② OCR：RapidOCR / PaddleOCR → 文本 + 坐标 + 置信度
     ↓ ③ 文本 LLM 结构化（DeepSeek）：文本 → JSON 字段 + 逐字段置信度
     ↓ ④ 置信度判断：高 → 落库；低 → ⑤ VLM 兜底（Qwen-VL-Plus / GLM-4V）
     ↓ ⑥ 人工校对界面（图片 + 抽取结果并排，一键修正）
     ↓ ⑦ 落库：字段 + OCR 原文 + 图片路径 + 感知哈希 + 出处

为什么不做成"VLM 一步到位"：成本高、中间过程不可见（无法溯源）、
结构化输出不稳定。OCR 打底保可控与成本，VLM 只在低置信度时兜底。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from ..errors import DataSourceError, UnsupportedModality

DEFAULT_BLUR_THRESHOLD = 80.0
DEFAULT_MIN_WIDTH = 640
DEFAULT_MIN_HEIGHT = 480
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _load_optional_module(name: str) -> Any:
    """运行时加载可选依赖，不把第三方类型存根纳入核心 mypy 图。"""
    return import_module(name)

# 奖状/证书要抽出的字段（与 Evidence.fields 对齐）
CERTIFICATE_FIELDS: tuple[str, ...] = (
    "姓名",
    "赛事名称",
    "级别",
    "奖项/名次",
    "获奖日期",
    "颁发单位",
    "团队属性",
)


@dataclass
class ImageQuality:
    """清晰度检查结果：模糊就直接提示重拍，别浪费 OCR 与人工时间。"""

    width: int
    height: int
    is_blurry: bool = False
    sharpness: float | None = None
    rotation_applied: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def needs_retake(self) -> bool:
        """模糊或分辨率不足时建议重拍，而不是继续消耗模型调用。"""
        return self.is_blurry or any(note.startswith("分辨率不足") for note in self.notes)


@dataclass
class OcrLine:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0


@dataclass
class OcrResult:
    lines: list[OcrLine] = field(default_factory=list)
    engine: str = "rapidocr"
    elapsed_seconds: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        return sum(line.confidence for line in self.lines) / len(self.lines) if self.lines else 0.0


def _require_pillow() -> tuple[Any, Any]:
    try:
        Image = _load_optional_module("PIL.Image")
        ImageOps = _load_optional_module("PIL.ImageOps")
    except ImportError as exc:  # pragma: no cover - 由无 extra 的安装环境覆盖
        raise UnsupportedModality(
            "缺少 Pillow，无法处理图片",
            detail={"hint": "uv sync --extra multimodal", "dependency": "pillow"},
        ) from exc
    return Image, ImageOps


def _require_cv2_numpy() -> tuple[Any, Any]:
    try:
        # 动态加载可选依赖，避免 Python 3.11 的 mypy 解析仅兼容 3.12 的
        # 新版 NumPy 类型存根；运行时行为与普通 import 相同。
        cv2 = _load_optional_module("cv2")
        np = _load_optional_module("numpy")
    except ImportError as exc:  # pragma: no cover - 由无 extra 的安装环境覆盖
        raise UnsupportedModality(
            "缺少 OpenCV/NumPy，无法检查或预处理图片",
            detail={"hint": "uv sync --extra multimodal", "dependency": "opencv-python-headless"},
        ) from exc
    return cv2, np


def _image_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        raise DataSourceError("图片文件不存在", detail={"path": str(candidate)})
    if candidate.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise DataSourceError(
            "不支持的图片格式",
            detail={"path": str(candidate), "suffix": candidate.suffix.lower()},
        )
    return candidate


def _open_or_raise(path: Path) -> Any:
    Image, _ = _require_pillow()
    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()
    except Exception as exc:
        raise DataSourceError(
            "无法读取图片文件",
            detail={"path": str(path), "error": str(exc)},
        ) from exc


def _exif_rotation(image: Any) -> tuple[int, str | None]:
    """返回 ImageOps.exif_transpose 将执行的顺时针旋转角度与镜像说明。"""
    try:
        orientation = int(image.getexif().get(274, 1))
    except (AttributeError, TypeError, ValueError):
        orientation = 1
    rotations = {3: 180, 6: 90, 8: 270, 5: 90, 7: 270}
    mirrored = orientation in {2, 4, 5, 7}
    return rotations.get(orientation, 0), "检测到 EXIF 镜像并已校正" if mirrored else None


def check_quality(
    path: str | Path,
    *,
    blurry_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> ImageQuality:
    """检查方向、分辨率与清晰度。

    清晰度采用灰度图 Laplacian 方差。该数值依赖图像内容，阈值必须在真实
    证书验证集上校准；默认值只用于第一版筛查。
    """
    candidate = _image_path(path)
    if blurry_threshold < 0 or min_width <= 0 or min_height <= 0:
        raise ValueError("质量阈值和最小尺寸必须为正数")
    cv2, np = _require_cv2_numpy()
    _, ImageOps = _require_pillow()
    image = _open_or_raise(candidate)
    rotation, mirror_note = _exif_rotation(image)
    corrected = ImageOps.exif_transpose(image).convert("RGB")
    width, height = corrected.size
    gray = cv2.cvtColor(np.asarray(corrected), cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    notes: list[str] = []
    if rotation:
        notes.append(f"已按 EXIF 顺时针旋转 {rotation}°")
    if mirror_note:
        notes.append(mirror_note)
    if width < min_width or height < min_height:
        notes.append(f"分辨率不足：{width}×{height}，建议至少 {min_width}×{min_height}")
    is_blurry = sharpness < blurry_threshold
    if is_blurry:
        notes.append(f"图像可能模糊：清晰度 {sharpness:.2f} < 阈值 {blurry_threshold:.2f}")
    return ImageQuality(
        width=width,
        height=height,
        is_blurry=is_blurry,
        sharpness=sharpness,
        rotation_applied=rotation,
        notes=notes,
    )


def _order_quad(points: Any, np: Any) -> Any:
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left
    return ordered


def _warp_document(image: Any, *, min_area_ratio: float = 0.2) -> tuple[Any, bool]:
    """检测最大的四边形文档轮廓并做透视矫正；找不到时保持原图。"""
    cv2, np = _require_cv2_numpy()
    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    quad = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(contour) < image_area * min_area_ratio:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype("float32") / scale
            break
    if quad is None:
        return image, False

    rect = _order_quad(quad, np)
    top_left, top_right, bottom_right, bottom_left = rect
    max_width = int(
        max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    )
    max_height = int(
        max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    )
    if max_width < 32 or max_height < 32:
        return image, False
    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, transform, (max_width, max_height)), True


def preprocess(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    correct_perspective: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
) -> Path:
    """EXIF 旋转、可选透视矫正/去噪/对比度增强，输出新的 PNG。

    原图永不覆盖。``out_dir`` 未指定时写入原图旁的 ``processed`` 子目录。
    """
    candidate = _image_path(path)
    cv2, np = _require_cv2_numpy()
    _, ImageOps = _require_pillow()
    source = _open_or_raise(candidate)
    rgb = ImageOps.exif_transpose(source).convert("RGB")
    image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if correct_perspective:
        image, _ = _warp_document(image)
    if denoise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    if enhance_contrast:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lightness)
        image = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    destination_dir = Path(out_dir) if out_dir is not None else candidate.parent / "processed"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{candidate.stem}_processed.png"
    try:
        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise ValueError("OpenCV 编码返回失败")
        buffer.tofile(str(destination))  # 支持 Windows 中文路径
    except Exception as exc:
        raise DataSourceError(
            "无法写出预处理图片",
            detail={"source": str(candidate), "destination": str(destination), "error": str(exc)},
        ) from exc
    return destination


@lru_cache(maxsize=1)
def _rapidocr_engine() -> Any:
    try:
        module = _load_optional_module("rapidocr_onnxruntime")
    except ImportError as exc:  # pragma: no cover - 由无 extra 的安装环境覆盖
        raise UnsupportedModality(
            "缺少 RapidOCR，无法识别图片文字",
            detail={"hint": "uv sync --extra multimodal", "dependency": "rapidocr-onnxruntime"},
        ) from exc
    return module.RapidOCR()


def _elapsed_seconds(raw: Any) -> float | None:
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    if isinstance(raw, (list, tuple)):
        numbers = [float(value) for value in raw if isinstance(value, (int, float))]
        return sum(numbers) if numbers else None
    return None


def _bbox(raw_box: Any) -> tuple[float, float, float, float] | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in raw_box]
    except (TypeError, ValueError, IndexError):
        return None
    if not points:
        return None
    xs, ys = zip(*points, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def run_ocr(path: str | Path, *, engine: Any | None = None) -> OcrResult:
    """用 RapidOCR 抽取文本、位置和置信度，保持识别顺序。

    ``engine`` 是测试/替换引擎注入口；生产默认复用进程级 RapidOCR 实例。
    """
    candidate = _image_path(path)
    selected_engine = engine or _rapidocr_engine()
    try:
        raw_result = selected_engine(str(candidate))
    except Exception as exc:
        raise DataSourceError(
            "OCR 识别失败",
            detail={"path": str(candidate), "engine": "rapidocr", "error": str(exc)},
        ) from exc

    items: Any
    elapsed: Any = None
    if isinstance(raw_result, tuple):
        items = raw_result[0]
        elapsed = raw_result[1] if len(raw_result) > 1 else None
    else:
        items = raw_result
    lines: list[OcrLine] = []
    for item in items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        try:
            confidence = min(1.0, max(0.0, float(item[2])))
        except (TypeError, ValueError):
            confidence = 0.0
        lines.append(OcrLine(text=text, bbox=_bbox(item[0]), confidence=confidence))
    return OcrResult(
        lines=lines,
        engine="rapidocr",
        elapsed_seconds=_elapsed_seconds(elapsed),
    )


def phash(path: str | Path, *, hash_size: int = 8) -> str:
    """计算感知哈希；相似图片应再结合字段指纹判断，不能只靠一个阈值。"""
    candidate = _image_path(path)
    if not 4 <= hash_size <= 32:
        raise ValueError("hash_size 必须位于 [4, 32]")
    try:
        imagehash = _load_optional_module("imagehash")
    except ImportError as exc:  # pragma: no cover - 由无 extra 的安装环境覆盖
        raise UnsupportedModality(
            "缺少 imagehash，无法计算感知哈希",
            detail={"hint": "uv sync --extra multimodal", "dependency": "imagehash"},
        ) from exc
    _, ImageOps = _require_pillow()
    image = _open_or_raise(candidate)
    return str(imagehash.phash(ImageOps.exif_transpose(image).convert("RGB"), hash_size=hash_size))


def phash_distance(left: str, right: str) -> int:
    """返回两个同长度十六进制 pHash 的汉明距离。"""
    if not left or len(left) != len(right):
        raise ValueError("两个 pHash 必须非空且长度相同")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("pHash 必须是十六进制字符串") from exc


def extract_certificate_fields(ocr_text: str | OcrResult, **kwargs: Any) -> Any:
    """兼容入口：OCR -> 严格 LLM 草稿 -> 代码校验与逐字段置信度。"""
    from ..evidence.certificate import extract_certificate_fields as _extract

    return _extract(ocr_text, **kwargs)


def extract_with_vlm(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """兼容入口：仅允许低置信字段及必要裁剪区域进入 VLM。"""
    from ..evidence.certificate import extract_with_vlm as _extract

    return _extract(path, **kwargs)


__all__ = [
    "CERTIFICATE_FIELDS",
    "ImageQuality",
    "OcrLine",
    "OcrResult",
    "check_quality",
    "extract_certificate_fields",
    "extract_with_vlm",
    "phash",
    "phash_distance",
    "preprocess",
    "run_ocr",
]
