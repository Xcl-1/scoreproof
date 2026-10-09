"""真实视觉模型适配器：只处理调用方生成的必要裁剪图。"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import DataSourceError, SchemaValidationError, UnsupportedModality

DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_CROP_BYTES = 7 * 1024 * 1024
MAX_TOTAL_CROP_BYTES = 7 * 1024 * 1024
MAX_FIELD_VALUE_LENGTH = 200


@dataclass(frozen=True)
class VlmProviderResponse:
    """适配器内部结果；原始模型回复不进入业务产物。"""

    values: dict[str, str | None]
    usage: dict[str, int]


def _validated_values(payload: Any, fields: list[str]) -> dict[str, str | None]:
    if not isinstance(payload, dict) or set(payload) != set(fields):
        raise SchemaValidationError(
            "VLM 返回字段集合与请求不一致",
            detail={"expected_fields": fields},
        )
    values: dict[str, str | None] = {}
    for field in fields:
        value = payload[field]
        if value is not None and not isinstance(value, str):
            raise SchemaValidationError("VLM 字段值必须是字符串或 null", detail={"field": field})
        if isinstance(value, str):
            value = value.strip()
            if not value:
                value = None
            elif len(value) > MAX_FIELD_VALUE_LENGTH:
                raise SchemaValidationError("VLM 字段值过长", detail={"field": field})
        values[field] = value
    return values


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    output: dict[str, int] = {}
    for source, target in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage, source, None)
        if isinstance(value, int) and value >= 0:
            output[target] = value
    return output


class QwenVlmClient:
    """阿里云百炼 OpenAI 兼容视觉客户端。"""

    def __init__(self, *, api_key: str | None = None, model: str = "qwen-vl-plus") -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model

    def invoke(self, payload: dict[str, Any]) -> VlmProviderResponse:
        if not self.api_key:
            raise UnsupportedModality("视觉模型缺少 DASHSCOPE_API_KEY，已转人工复核")
        fields = payload.get("fields")
        crops = payload.get("crops")
        if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
            raise ValueError("VLM fields 无效")
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("VLM fields 不能为空或重复")
        if not isinstance(crops, list) or not crops:
            raise ValueError("VLM crops 无效")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - llm extra 缺失
            raise DataSourceError(
                "真实 VLM 不可用：未安装 openai 客户端",
                detail={"hint": "uv sync --extra llm"},
            ) from exc

        content: list[dict[str, Any]] = []
        total_crop_bytes = 0
        for index, crop in enumerate(crops):
            if (
                not isinstance(crop, dict)
                or not isinstance(crop.get("crop_path"), str)
                or crop.get("field") not in fields
            ):
                raise ValueError("VLM crop 元数据无效")
            path = Path(crop["crop_path"])
            data = path.read_bytes()
            if len(data) > MAX_CROP_BYTES:
                raise ValueError("VLM 单个裁剪图超过 7MB 安全上限")
            total_crop_bytes += len(data)
            if total_crop_bytes > MAX_TOTAL_CROP_BYTES:
                raise ValueError("VLM 裁剪图总量超过 7MB 安全上限")
            encoded = base64.b64encode(data).decode("ascii")
            content.extend(
                [
                    {"type": "text", "text": f"裁剪 {index + 1} 对应字段：{crop['field']}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ]
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "只识别裁剪图中清晰可见的目标字段。请按照 JSON 格式输出一个对象，"
                    f"键必须且只能是 {json.dumps(fields, ensure_ascii=False)}；"
                    "值必须是裁剪图中的逐字文本，无法确认时为 null。不要推断、解释或增加字段。"
                ),
            }
        )
        try:
            response = OpenAI(
                api_key=self.api_key,
                base_url=DASHSCOPE_COMPATIBLE_BASE_URL,
                timeout=60.0,
            ).chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
                presence_penalty=1.5,
            )
        except Exception as exc:
            raise DataSourceError(
                "千问视觉模型调用失败",
                detail={"provider": "qwen-vl-plus", "error_type": type(exc).__name__},
            ) from exc
        text = response.choices[0].message.content
        if not isinstance(text, str):
            raise SchemaValidationError("VLM 返回内容不是 JSON 文本")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError("VLM 返回内容不是有效 JSON") from exc
        return VlmProviderResponse(
            values=_validated_values(decoded, fields),
            usage=_usage_dict(response),
        )


def make_vlm_client(provider: str) -> QwenVlmClient:
    normalized = provider.strip().lower()
    if normalized == "qwen-vl-plus":
        return QwenVlmClient()
    raise UnsupportedModality(
        "该视觉模型尚无真实客户端适配器，已转人工复核",
        detail={"provider": normalized, "supported": ["qwen-vl-plus"]},
    )


__all__ = [
    "DASHSCOPE_COMPATIBLE_BASE_URL",
    "QwenVlmClient",
    "VlmProviderResponse",
    "make_vlm_client",
]
