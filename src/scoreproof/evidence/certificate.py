"""奖状 OCR 文本结构化、代码级校验、置信度与 VLM 触发决策。

文本模型只提交严格的字段草稿。规范化值、原文支持、定位与置信度全部由
本模块确定性计算；任何无法被 OCR 原文支持的值都会被清空并进入人工复核。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from ..config import get_settings
from ..errors import DataSourceError, SchemaValidationError, ScoreProofError, UnsupportedModality
from ..ingest.image_loader import ImageQuality, OcrLine, OcrResult, check_quality, phash, preprocess, run_ocr
from ..normalize import parse_prize, parse_tier
from ..observability import CostLedger, classify_model_tier, pricing_from_settings
from ..schema import Evidence

CERTIFICATE_FIELD_NAMES: tuple[str, ...] = (
    "姓名",
    "赛事名称",
    "级别",
    "奖项/名次",
    "获奖日期",
    "颁发单位",
    "团队属性",
)


class CertificateFieldDraft(BaseModel):
    """文本模型允许提交的单字段草稿；模型不得提交规范值或置信度。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    raw_value: StrictStr | None
    evidence_text: StrictStr | None
    line_indexes: list[StrictInt]


class CertificateDraft(BaseModel):
    """DeepSeek 输出契约。七个字段必须出现，但未知值必须显式为 null。"""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    name: CertificateFieldDraft = Field(alias="姓名")
    event_name: CertificateFieldDraft = Field(alias="赛事名称")
    level: CertificateFieldDraft = Field(alias="级别")
    award: CertificateFieldDraft = Field(alias="奖项/名次")
    award_date: CertificateFieldDraft = Field(alias="获奖日期")
    issuer: CertificateFieldDraft = Field(alias="颁发单位")
    team: CertificateFieldDraft = Field(alias="团队属性")


CERTIFICATE_DRAFT_SCHEMA: dict[str, Any] = CertificateDraft.model_json_schema(by_alias=True)


class FieldLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    line_indexes: list[int] = Field(default_factory=list, description="OCR 行号，0-based")
    bboxes: list[tuple[float, float, float, float]] = Field(default_factory=list)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)


class FieldSignals(BaseModel):
    """字段置信度的可审计信号；没有适用信号时用 null，不偷偷加分。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    ocr_line_confidence: float = Field(ge=0.0, le=1.0)
    source_supported: bool
    format_valid: bool
    dictionary_match: bool | None = None
    multi_source_agreement: bool | None = None


class CertificateField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    raw_value: str | None = None
    normalized_value: str | None = None
    evidence_text: str | None = None
    location: FieldLocation = Field(default_factory=FieldLocation)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: FieldSignals
    validation_errors: list[str] = Field(default_factory=list)


class CertificateFieldSet(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    name: CertificateField = Field(alias="姓名")
    event_name: CertificateField = Field(alias="赛事名称")
    level: CertificateField = Field(alias="级别")
    award: CertificateField = Field(alias="奖项/名次")
    award_date: CertificateField = Field(alias="获奖日期")
    issuer: CertificateField = Field(alias="颁发单位")
    team: CertificateField = Field(alias="团队属性")

    def by_name(self) -> dict[str, CertificateField]:
        return {
            "姓名": self.name,
            "赛事名称": self.event_name,
            "级别": self.level,
            "奖项/名次": self.award,
            "获奖日期": self.award_date,
            "颁发单位": self.issuer,
            "团队属性": self.team,
        }


class VlmRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field: str
    bbox: tuple[float, float, float, float]
    reason: str


VlmStatus = Literal[
    "not_needed",
    "provider_missing",
    "key_missing",
    "crop_missing",
    "ready",
    "called",
    "failed",
]


class VlmDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requested: bool = False
    called: bool = False
    provider: str | None = None
    status: VlmStatus = "not_needed"
    low_confidence_fields: list[str] = Field(default_factory=list)
    eligible_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    regions: list[VlmRegion] = Field(default_factory=list)


class CertificateExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fields: CertificateFieldSet
    extractor: Literal["ocr+llm"] = "ocr+llm"
    model: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    vlm: VlmDecision
    manual_review_required: bool
    manual_review_status: Literal["not_required", "pending_manual_review"]
    manual_review_reasons: list[str] = Field(default_factory=list)


class OcrLinePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class OcrPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    engine: str
    elapsed_seconds: float | None = None
    mean_confidence: float = Field(ge=0.0, le=1.0)
    text: str
    lines: list[OcrLinePayload]


class QualityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    width: int
    height: int
    sharpness: float | None = None
    is_blurry: bool
    needs_retake: bool
    rotation_applied: int
    notes: list[str]


class CertificatePipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: str
    processed: str | None
    quality: QualityPayload
    phash: str
    ocr: OcrPayload
    extraction: CertificateExtraction
    evidence: Evidence


class CertificateTextExtractor:
    """通过 OpenAI 兼容接口调用 DeepSeek，只抽 OCR 文本中的可见字段。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        cost_ledger: CostLedger | None = None,
        material_id: str | None = None,
        batch_id: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.base_url = settings.llm_base_url
        self.client = client
        self.cost_ledger = cost_ledger
        self.material_id = material_id
        self.batch_id = batch_id

    def available(self) -> bool:
        if self.client is not None:
            return True
        if not get_settings().llm_configured:
            return False
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        raise SchemaValidationError("证书字段模型返回内容不是文本")

    def build_prompt(self, ocr: OcrResult) -> list[dict[str, str]]:
        lines = [
            {"line_index": index, "text": line.text, "ocr_confidence": line.confidence}
            for index, line in enumerate(ocr.lines)
        ]
        system = (
            "你是奖状 OCR 文本的保守结构化抽取器。只输出符合 schema 的单个 JSON 对象；"
            "七个字段都必须出现。raw_value 和 evidence_text 必须逐字来自 OCR 行，"
            "line_indexes 使用 0-based 行号。不能从缺失信息推断：没有明确原文就把"
            "raw_value/evidence_text 设为 null、line_indexes 设为空数组。"
            "级别仅指国家级/省级/市级/校级/院级/国际级等范围；奖项或名次单独填写。"
            "团队属性只有原文明确出现团队/团体/个人/单人时才填写。"
            "不要输出规范值、置信度、解释或 schema 外字段。"
        )
        payload = {"schema": CERTIFICATE_DRAFT_SCHEMA, "ocr_lines": lines}
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _client_for(self) -> Any:
        if self.client is not None:
            return self.client
        settings = get_settings()
        if not settings.llm_api_key:
            raise DataSourceError("奖状文本抽取不可用：缺少 DEEPSEEK_API_KEY")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - 依赖安装状态
            raise DataSourceError(
                "奖状文本抽取不可用：未安装 langchain-openai",
                detail={"hint": "uv sync --extra llm"},
            ) from exc
        return ChatOpenAI(
            model=self.model,
            api_key=settings.llm_api_key,
            base_url=self.base_url,
            temperature=0,
            max_tokens=2048,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def extract(self, ocr: OcrResult) -> CertificateDraft:
        if not self.available():
            raise DataSourceError(
                "奖状文本抽取不可用：缺少 DEEPSEEK_API_KEY 或 langchain-openai",
                detail={"hint": "在 .env 配置 DeepSeek，并安装 llm extra"},
            )
        try:
            response = self._client_for().invoke(self.build_prompt(ocr))
        except ScoreProofError:
            raise
        except Exception as exc:
            if self.cost_ledger is not None:
                self.cost_ledger.record_failure(
                    provider="deepseek",
                    model=self.model,
                    model_tier=classify_model_tier(self.model),
                    purpose="certificate_text_extraction",
                    error=exc,
                    material_id=self.material_id,
                    batch_id=self.batch_id,
                )
            raise DataSourceError(
                "DeepSeek 奖状文本抽取调用失败",
                detail={"model": self.model, "error_type": type(exc).__name__},
            ) from exc
        if self.cost_ledger is not None:
            self.cost_ledger.record_response(
                response,
                provider="deepseek",
                model=self.model,
                model_tier=classify_model_tier(self.model),
                purpose="certificate_text_extraction",
                material_id=self.material_id,
                batch_id=self.batch_id,
                pricing=pricing_from_settings(get_settings()),
            )
        raw = self._response_text(response).strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw = fenced.group(1)
        try:
            payload = json.loads(raw)
            return CertificateDraft.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SchemaValidationError(
                "奖状字段模型输出未通过严格 schema 校验",
                detail={"model": self.model},
            ) from exc


def _as_ocr(value: str | OcrResult) -> OcrResult:
    if isinstance(value, OcrResult):
        return value
    return OcrResult(lines=[OcrLine(text=line, confidence=1.0) for line in value.splitlines() if line])


def _parse_date(raw: str) -> str | None:
    cleaned = raw.strip()
    match = re.search(r"(?P<y>20\d{2})\s*[年./-]\s*(?P<m>\d{1,2})\s*[月./-]\s*(?P<d>\d{1,2})\s*日?", cleaned)
    if not match:
        return None
    try:
        return date(int(match["y"]), int(match["m"]), int(match["d"])).isoformat()
    except ValueError:
        return None


def _normalize_team(raw: str) -> str | None:
    cleaned = re.sub(r"\s+", "", raw)
    if any(word in cleaned for word in ("团队", "团体", "集体", "多人")):
        return "团队"
    if any(word in cleaned for word in ("个人", "单人", "个人项目")):
        return "个人"
    return None


def _normalize_field(name: str, raw: str) -> tuple[str | None, bool | None]:
    value = raw.strip()
    if not value:
        return None, None
    if name == "姓名":
        normalized_name = re.sub(r"(?:同学|同志)$", "", re.sub(r"\s+", "", value))
        return (normalized_name or None), None
    if name == "赛事名称":
        return value, None
    if name == "级别":
        normalized_tier = parse_tier(value)
        return normalized_tier, normalized_tier is not None
    if name == "奖项/名次":
        normalized_prize = parse_prize(value)
        return normalized_prize, normalized_prize is not None
    if name == "获奖日期":
        return _parse_date(value), None
    if name == "颁发单位":
        return value, None
    if name == "团队属性":
        normalized_team = _normalize_team(value)
        return normalized_team, normalized_team is not None
    raise ValueError(f"未知奖状字段：{name}")


def _heuristic_values(ocr: OcrResult) -> dict[str, str | None]:
    text = ocr.text
    event_match = re.search(r"在(.{2,100}?)(?:中|上)荣获", text.replace("\n", ""))
    issuer = next(
        (
            line.text.strip()
            for line in reversed(ocr.lines)
            if re.search(r"(?:委员会|学校|学院|协会|中心|政府|教育厅|组委会)$", line.text.strip())
        ),
        None,
    )
    date_match = re.search(r"20\d{2}\s*[年./-]\s*\d{1,2}\s*[月./-]\s*\d{1,2}\s*日?", text)
    tier = parse_tier(text)
    prize = parse_prize(text)
    team = _normalize_team(text)
    return {
        "赛事名称": event_match.group(1) if event_match else None,
        "级别": tier,
        "奖项/名次": prize,
        "获奖日期": _parse_date(date_match.group(0)) if date_match else None,
        "颁发单位": issuer,
        "团队属性": team,
        "姓名": None,
    }


def _location(ocr: OcrResult, draft: CertificateFieldDraft) -> FieldLocation:
    indexes = [
        index
        for index in dict.fromkeys(draft.line_indexes)
        if 0 <= index < len(ocr.lines)
        and (
            (draft.evidence_text and draft.evidence_text in ocr.lines[index].text)
            or (draft.raw_value and draft.raw_value in ocr.lines[index].text)
        )
    ]
    if not indexes:
        needle = draft.evidence_text or draft.raw_value or ""
        indexes = [index for index, line in enumerate(ocr.lines) if needle and needle in line.text]
    bboxes = [ocr.lines[index].bbox for index in indexes if ocr.lines[index].bbox is not None]
    evidence = draft.evidence_text or ""
    char_start = ocr.text.find(evidence) if evidence else -1
    return FieldLocation(
        line_indexes=indexes,
        bboxes=[bbox for bbox in bboxes if bbox is not None],
        char_start=char_start if char_start >= 0 else None,
        char_end=char_start + len(evidence) if char_start >= 0 else None,
    )


def _confidence(signals: FieldSignals) -> float:
    weighted: list[tuple[float, float]] = [
        (signals.ocr_line_confidence, 0.30),
        (float(signals.source_supported), 0.25),
        (float(signals.format_valid), 0.15),
    ]
    if signals.dictionary_match is not None:
        weighted.append((float(signals.dictionary_match), 0.15))
    if signals.multi_source_agreement is not None:
        weighted.append((float(signals.multi_source_agreement), 0.15))
    total_weight = sum(weight for _, weight in weighted)
    return round(sum(value * weight for value, weight in weighted) / total_weight, 4)


def _validate_field(
    name: str,
    draft: CertificateFieldDraft,
    ocr: OcrResult,
    heuristic: str | None,
) -> CertificateField:
    raw = draft.raw_value.strip() if draft.raw_value else None
    evidence = draft.evidence_text.strip() if draft.evidence_text else None
    errors: list[str] = []
    source_supported = bool(
        raw and evidence and raw in ocr.text and evidence in ocr.text and raw in evidence
    )
    if raw is None:
        errors.append("missing")
    elif not source_supported:
        errors.append("unsupported_by_ocr")
    normalized, dictionary_match = _normalize_field(name, raw or "")
    format_valid = normalized is not None
    if raw is not None and not format_valid:
        errors.append("invalid_format_or_enum")
    location = _location(ocr, draft) if source_supported else FieldLocation()
    ocr_confidence = (
        sum(ocr.lines[index].confidence for index in location.line_indexes)
        / len(location.line_indexes)
        if location.line_indexes
        else 0.0
    )
    agreement = None if heuristic is None or normalized is None else heuristic == normalized
    if agreement is False:
        errors.append("multi_source_disagreement")
    signals = FieldSignals(
        ocr_line_confidence=round(ocr_confidence, 4),
        source_supported=source_supported,
        format_valid=format_valid,
        dictionary_match=dictionary_match,
        multi_source_agreement=agreement,
    )
    confidence = _confidence(signals) if source_supported and format_valid else 0.0
    if agreement is False:
        confidence = min(confidence, 0.55)
    if not source_supported or not format_valid:
        raw = normalized = evidence = None
        location = FieldLocation()
    return CertificateField(
        raw_value=raw,
        normalized_value=normalized,
        evidence_text=evidence,
        location=location,
        confidence=confidence,
        signals=signals,
        validation_errors=errors,
    )


def _field_drafts(draft: CertificateDraft) -> dict[str, CertificateFieldDraft]:
    return {
        "姓名": draft.name,
        "赛事名称": draft.event_name,
        "级别": draft.level,
        "奖项/名次": draft.award,
        "获奖日期": draft.award_date,
        "颁发单位": draft.issuer,
        "团队属性": draft.team,
    }


def decide_vlm_fallback(
    fields: CertificateFieldSet,
    *,
    threshold: float = 0.8,
    provider: str | None = None,
    dashscope_configured: bool | None = None,
    zhipu_configured: bool | None = None,
) -> VlmDecision:
    """只选择低置信字段及其 OCR bbox；无裁剪定位时不得发送整图。"""
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold 必须位于 [0,1]")
    selected = [
        name
        for name, field in fields.by_name().items()
        if field.normalized_value is None or field.confidence < threshold
    ]
    if not selected:
        return VlmDecision()
    settings = get_settings()
    chosen = (provider if provider is not None else settings.vlm_provider) or None
    chosen = chosen.strip().lower() if chosen is not None else None
    chosen = None if chosen in {"", "none"} else chosen
    regions: list[VlmRegion] = []
    for name in selected:
        field = fields.by_name()[name]
        regions.extend(
            VlmRegion(field=name, bbox=bbox, reason=f"{name} 置信度 {field.confidence:.4f}")
            for bbox in field.location.bboxes
        )
    reasons = [f"{name}: 缺失或置信度低于 {threshold:g}" for name in selected]
    eligible = list(dict.fromkeys(region.field for region in regions))
    if chosen is None:
        reasons.append("未配置 SCOREPROOF_VLM_PROVIDER，转人工复核")
        return VlmDecision(
            requested=True,
            provider=None,
            status="provider_missing",
            low_confidence_fields=selected,
            eligible_fields=eligible,
            reasons=reasons,
            regions=regions,
        )
    if chosen not in {"qwen-vl-plus", "glm-4v"}:
        reasons.append(f"不支持的 VLM provider: {chosen}，转人工复核")
        return VlmDecision(
            requested=True,
            provider=chosen,
            status="provider_missing",
            low_confidence_fields=selected,
            eligible_fields=eligible,
            reasons=reasons,
            regions=regions,
        )
    dashscope = bool(os.environ.get("DASHSCOPE_API_KEY")) if dashscope_configured is None else dashscope_configured
    zhipu = bool(os.environ.get("ZHIPUAI_API_KEY")) if zhipu_configured is None else zhipu_configured
    has_key = dashscope if chosen == "qwen-vl-plus" else zhipu
    if not has_key:
        key_name = "DASHSCOPE_API_KEY" if chosen == "qwen-vl-plus" else "ZHIPUAI_API_KEY"
        reasons.append(f"缺少 {key_name}，转人工复核")
        return VlmDecision(
            requested=True,
            provider=chosen,
            status="key_missing",
            low_confidence_fields=selected,
            eligible_fields=eligible,
            reasons=reasons,
            regions=regions,
        )
    if not regions:
        reasons.append("低置信字段没有可安全发送的 OCR 裁剪区域，禁止发送整图，转人工复核")
        return VlmDecision(
            requested=True,
            provider=chosen,
            status="crop_missing",
            low_confidence_fields=selected,
            eligible_fields=[],
            reasons=reasons,
            regions=[],
        )
    return VlmDecision(
        requested=True,
        provider=chosen,
        status="ready",
        low_confidence_fields=selected,
        eligible_fields=eligible,
        reasons=reasons,
        regions=regions,
    )


def extract_certificate_fields(
    ocr_value: str | OcrResult,
    *,
    extractor: CertificateTextExtractor | None = None,
    confidence_threshold: float = 0.8,
    provider: str | None = None,
) -> CertificateExtraction:
    """OCR -> DeepSeek 草稿 -> 代码校验/归一化/置信度 -> VLM 决策。"""
    ocr = _as_ocr(ocr_value)
    if not ocr.lines:
        raise DataSourceError("OCR 文本为空，无法抽取奖状字段")
    selected = extractor or CertificateTextExtractor()
    draft = selected.extract(ocr)
    heuristics = _heuristic_values(ocr)
    validated = {
        name: _validate_field(name, item, ocr, heuristics.get(name))
        for name, item in _field_drafts(draft).items()
    }
    fields = CertificateFieldSet.model_validate(validated)
    vlm = decide_vlm_fallback(fields, threshold=confidence_threshold, provider=provider)
    review_reasons = list(vlm.reasons)
    return CertificateExtraction(
        fields=fields,
        model=selected.model,
        confidence_threshold=confidence_threshold,
        vlm=vlm,
        manual_review_required=vlm.requested,
        manual_review_status="pending_manual_review" if vlm.requested else "not_required",
        manual_review_reasons=review_reasons,
    )


def extract_with_vlm(
    path: str | Path,
    *,
    fields: Sequence[str],
    regions: Sequence[VlmRegion],
    client: Any | None = None,
    provider: str | None = None,
    out_dir: str | Path | None = None,
    cost_ledger: CostLedger | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """受限 VLM 注入口：只允许调用方传入低置信字段和明确 bbox。

    本函数故意不接受“整图兜底”开关。未注入已配置客户端时显式失败，避免
    把触发判断或缺 Key 状态伪装成已调用视觉模型。
    """
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        raise DataSourceError("VLM 图片文件不存在", detail={"path": str(candidate)})
    if not fields:
        raise ValueError("VLM 只能处理明确列出的低置信字段")
    region_fields = {region.field for region in regions}
    if not regions or set(fields) != region_fields:
        raise ValueError("VLM 必须只接收低置信字段对应的必要裁剪区域")
    chosen = provider or get_settings().vlm_provider
    if chosen not in {"qwen-vl-plus", "glm-4v"}:
        raise UnsupportedModality(
            "未配置可用的视觉模型，已转人工复核",
            detail={"required": ["SCOREPROOF_VLM_PROVIDER", "对应供应商 API Key"]},
        )
    key_name = "DASHSCOPE_API_KEY" if chosen == "qwen-vl-plus" else "ZHIPUAI_API_KEY"
    if not os.environ.get(key_name):
        raise UnsupportedModality(
            "视觉模型缺少对应 Key，已转人工复核",
            detail={"provider": chosen, "required": key_name},
        )
    if client is None:
        raise UnsupportedModality(
            "视觉模型客户端尚未注入，已转人工复核",
            detail={"provider": chosen, "fields": list(fields)},
        )
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - multimodal extra 缺失
        raise UnsupportedModality(
            "缺少 Pillow，无法生成 VLM 必要裁剪区域",
            detail={"dependency": "pillow"},
        ) from exc
    destination = (
        Path(out_dir)
        if out_dir is not None
        else get_settings().data_dir / "processed" / "vlm-crops"
    )
    destination.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, Any]] = []
    with Image.open(candidate) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        for index, region in enumerate(regions):
            left = max(0, min(width, int(region.bbox[0])))
            top = max(0, min(height, int(region.bbox[1])))
            right = max(0, min(width, int(region.bbox[2])))
            bottom = max(0, min(height, int(region.bbox[3])))
            if right <= left or bottom <= top:
                raise ValueError(f"VLM 裁剪区域无效：{region.bbox}")
            crop_path = destination / f"{candidate.stem}-crop-{index:02d}.png"
            image.crop((left, top, right, bottom)).save(crop_path, format="PNG")
            crops.append(
                {
                    "field": region.field,
                    "crop_path": str(crop_path),
                    "source_bbox": region.bbox,
                    "reason": region.reason,
                }
            )
    payload = {
        "fields": list(fields),
        "crops": crops,
    }
    try:
        response = client.invoke(payload)
    except Exception as exc:
        if cost_ledger is not None:
            cost_ledger.record_failure(
                provider=chosen,
                model=chosen,
                model_tier="vision",
                purpose="certificate_vlm_crop",
                error=exc,
                material_id=f"image:{_file_sha256(candidate)}",
                batch_id=batch_id,
            )
        raise
    if cost_ledger is not None:
        cost_ledger.record_response(
            response,
            provider=chosen,
            model=chosen,
            model_tier="vision",
            purpose="certificate_vlm_crop",
            material_id=f"image:{_file_sha256(candidate)}",
            batch_id=batch_id,
        )
    if not isinstance(response, Mapping):
        raise SchemaValidationError("VLM 返回必须是结构化对象")
    return dict(response)


def _quality_payload(quality: ImageQuality) -> QualityPayload:
    return QualityPayload(
        width=quality.width,
        height=quality.height,
        sharpness=quality.sharpness,
        is_blurry=quality.is_blurry,
        needs_retake=quality.needs_retake,
        rotation_applied=quality.rotation_applied,
        notes=quality.notes,
    )


def _ocr_payload(ocr: OcrResult) -> OcrPayload:
    return OcrPayload(
        engine=ocr.engine,
        elapsed_seconds=ocr.elapsed_seconds,
        mean_confidence=round(ocr.mean_confidence, 4),
        text=ocr.text,
        lines=[
            OcrLinePayload(text=line.text, bbox=line.bbox, confidence=line.confidence)
            for line in ocr.lines
        ],
    )


def extract_certificate(
    path: str | Path,
    *,
    run_preprocess: bool = True,
    processed_dir: str | Path | None = None,
    extractor: CertificateTextExtractor | None = None,
    confidence_threshold: float = 0.8,
    provider: str | None = None,
    ocr_result: OcrResult | None = None,
    cost_ledger: CostLedger | None = None,
    batch_id: str | None = None,
) -> CertificatePipelineResult:
    """真实图片主链路：质量检查、预处理、RapidOCR、文本 LLM 与复核状态。"""
    source = Path(path)
    quality = check_quality(source)
    target = preprocess(source, out_dir=processed_dir) if run_preprocess else source
    ocr = ocr_result or run_ocr(target)
    material_id = f"image:{_file_sha256(source)}"
    selected_extractor = extractor or CertificateTextExtractor(
        cost_ledger=cost_ledger,
        material_id=material_id,
        batch_id=batch_id,
    )
    extraction = extract_certificate_fields(
        ocr,
        extractor=selected_extractor,
        confidence_threshold=confidence_threshold,
        provider=provider,
    )
    if quality.needs_retake:
        extraction.manual_review_required = True
        extraction.manual_review_status = "pending_manual_review"
        extraction.manual_review_reasons.append("图片质量不足，建议重拍或人工复核")
    image_hash = phash(source)
    normalized = {
        name: field.normalized_value
        for name, field in extraction.fields.by_name().items()
        if field.normalized_value is not None
    }
    confidences = {
        name: field.confidence for name, field in extraction.fields.by_name().items()
    }
    evidence = Evidence(
        type="image",
        path=str(source),
        ocr_text=ocr.text,
        fields=normalized,
        field_confidence=confidences,
        phash=image_hash,
        extractor="ocr+llm",
        manual_corrected=False,
        extra={
            "field_details": extraction.fields.model_dump(mode="json", by_alias=True),
            "vlm": extraction.vlm.model_dump(mode="json"),
            "manual_review_reasons": extraction.manual_review_reasons,
        },
    )
    return CertificatePipelineResult(
        source=str(source),
        processed=str(target) if run_preprocess else None,
        quality=_quality_payload(quality),
        phash=image_hash,
        ocr=_ocr_payload(ocr),
        extraction=extraction,
        evidence=evidence,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CERTIFICATE_DRAFT_SCHEMA",
    "CERTIFICATE_FIELD_NAMES",
    "CertificateDraft",
    "CertificateExtraction",
    "CertificateField",
    "CertificateFieldDraft",
    "CertificateFieldSet",
    "CertificatePipelineResult",
    "CertificateTextExtractor",
    "FieldLocation",
    "FieldSignals",
    "VlmDecision",
    "VlmRegion",
    "decide_vlm_fallback",
    "extract_certificate",
    "extract_certificate_fields",
    "extract_with_vlm",
]
