"""图片接入（P2 待实现）：预处理 + OCR + 字段抽取 + 查重。

本文件目前只定义**流水线骨架与数据结构**，不假装已经能识别奖状。
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import UnsupportedModality

# 奖状/证书要抽出的字段（与 Evidence.fields 对齐）
CERTIFICATE_FIELDS: tuple[str, ...] = (
    "赛事名称",
    "等级",
    "获奖时间",
    "颁发单位",
    "姓名",
    "是否团队",
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


@dataclass
class OcrLine:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0


@dataclass
class OcrResult:
    lines: list[OcrLine] = field(default_factory=list)
    engine: str = "rapidocr"

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        return sum(line.confidence for line in self.lines) / len(self.lines) if self.lines else 0.0


def check_quality(path: str | Path) -> ImageQuality:  # pragma: no cover - P2
    """清晰度/方向检查（Pillow + OpenCV）。"""
    raise UnsupportedModality(
        "图片质量检查未实现（P2）",
        detail={"hint": "uv sync --extra multimodal", "path": str(path)},
    )


def preprocess(path: str | Path, *, out_dir: str | Path | None = None) -> Path:  # pragma: no cover - P2
    """EXIF 旋转 + 去噪 + 透视矫正，输出规范化图片。"""
    raise UnsupportedModality("图片预处理未实现（P2）", detail={"path": str(path)})


def run_ocr(path: str | Path) -> OcrResult:  # pragma: no cover - P2
    """RapidOCR（onnxruntime，pip 即装，无需显卡）。"""
    raise UnsupportedModality(
        "OCR 未实现（P2）", detail={"engine": "rapidocr-onnxruntime", "path": str(path)}
    )


def phash(path: str | Path) -> str:  # pragma: no cover - P2
    """感知哈希（imagehash.pHash），用于"一证多报"检测。"""
    raise UnsupportedModality("pHash 未实现（P2）", detail={"hint": "uv sync --extra multimodal"})


def extract_certificate_fields(ocr_text: str, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
    """OCR 文本 -> 结构化字段 + 逐字段置信度（走文本 LLM，非 VLM）。"""
    raise UnsupportedModality(
        "奖状字段抽取未实现（P2）",
        detail={"fields": list(CERTIFICATE_FIELDS), "hint": "复用 rules.extractor.LLMExtractor 模式"},
    )


def extract_with_vlm(path: str | Path, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
    """低置信度兜底：把图片交给 VLM（Qwen-VL-Plus / GLM-4V）。"""
    raise UnsupportedModality(
        "VLM 兜底未实现（P2）",
        detail={"providers": ["qwen-vl-plus", "glm-4v"], "path": str(path)},
    )


__all__ = [
    "CERTIFICATE_FIELDS",
    "ImageQuality",
    "OcrLine",
    "OcrResult",
    "check_quality",
    "extract_certificate_fields",
    "extract_with_vlm",
    "phash",
    "preprocess",
    "run_ocr",
]
