"""证据联合查重：文件哈希、感知哈希与结构化事实三层判定。

本模块只输出拦截/复核建议，不删除证据。阈值必须通过成对评测集校准；默认值
用于工程烟雾验证，不代表正式业务验收。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..normalize import normalize_text_key, parse_prize, parse_tier
from ..schema import Evidence

FACT_FIELDS: tuple[str, ...] = (
    "姓名",
    "赛事名称",
    "级别",
    "奖项/名次",
    "获奖日期",
    "颁发单位",
    "团队属性",
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "姓名": ("姓名", "name", "student_name"),
    "赛事名称": ("赛事名称", "赛事", "event_name", "event"),
    "级别": ("级别", "等级", "tier", "level"),
    "奖项/名次": ("奖项/名次", "奖项", "名次", "award", "prize", "rank"),
    "获奖日期": ("获奖日期", "时间", "award_date", "date"),
    "颁发单位": ("颁发单位", "issuer", "organizer"),
    "团队属性": ("团队属性", "team_attribute", "team"),
}


class DuplicateThresholds(BaseModel):
    """待评测校准的联合查重阈值。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    phash_definite_max: int = Field(default=2, ge=0, le=64)
    phash_suspected_max: int = Field(default=10, ge=0, le=64)
    semantic_definite_min: float = Field(default=0.98, ge=0.0, le=1.0)
    semantic_suspected_min: float = Field(default=0.80, ge=0.0, le=1.0)
    semantic_min_fields: int = Field(default=4, ge=1, le=len(FACT_FIELDS))


class FieldSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field: str
    left_value: str
    right_value: str
    left_normalized: str
    right_normalized: str
    similarity: float = Field(ge=0.0, le=1.0)
    exact: bool
    conflict: bool


class DuplicateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    left_id: str
    right_id: str
    status: Literal["确定重复", "疑似重复", "非重复"]
    action: Literal["拦截并人工复核", "送人工复核", "放行"]
    reasons: list[str]
    left_sha256: str | None = None
    right_sha256: str | None = None
    sha256_match: bool | None = None
    phash_distance: int | None = Field(default=None, ge=0)
    fact_fingerprint_match: bool | None = None
    semantic_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    comparable_fields: int = Field(default=0, ge=0)
    field_similarities: list[FieldSimilarity] = Field(default_factory=list)
    thresholds: DuplicateThresholds

    @property
    def flagged(self) -> bool:
        return self.status != "非重复"


def file_sha256(path: str | Path | None) -> str | None:
    """计算真实文件 SHA-256；没有可读文件时返回 ``None``。"""
    if path is None:
        return None
    candidate = Path(path)
    try:
        if not candidate.exists() or not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def phash_distance(left: str | None, right: str | None) -> int | None:
    """返回两个十六进制感知哈希的汉明距离；非法输入不参与判定。"""
    if not left or not right or len(left) != len(right):
        return None
    if not re.fullmatch(r"[0-9a-fA-F]+", left) or not re.fullmatch(r"[0-9a-fA-F]+", right):
        return None
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _scalar(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("normalized_value", "raw_value", "value"):
            nested = value.get(key)
            if nested is not None:
                return _scalar(nested)
        return None
    if isinstance(value, bool):
        return "团队" if value else "个人"
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def evidence_facts(evidence: Evidence) -> dict[str, str]:
    """兼容当前中文字段与历史英文/简写字段，提取规范事实。"""
    facts: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            value = _scalar(evidence.fields.get(alias))
            if value is not None:
                facts[canonical] = value
                break
    return facts


def _normalize_fact(field: str, value: str) -> str:
    if field == "级别":
        return parse_tier(value) or normalize_text_key(value).lower()
    if field == "奖项/名次":
        return parse_prize(value) or normalize_text_key(value).lower()
    if field == "获奖日期":
        digits = re.findall(r"\d+", value)
        if len(digits) >= 3:
            year, month, day = int(digits[0]), int(digits[1]), int(digits[2])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
    if field == "团队属性":
        key = normalize_text_key(value).lower()
        if key in {"团队", "团体", "集体", "team", "true", "是"}:
            return "团队"
        if key in {"个人", "individual", "false", "否"}:
            return "个人"
    return normalize_text_key(value).lower()


def fact_fingerprint(evidence: Evidence) -> str | None:
    """生成带字段名的稳定事实指纹；空事实不产生指纹。"""
    facts = evidence_facts(evidence)
    parts = [f"{field}={_normalize_fact(field, facts[field])}" for field in FACT_FIELDS if field in facts]
    if not parts:
        return None
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _field_similarity(field: str, left: str, right: str) -> FieldSimilarity:
    left_norm = _normalize_fact(field, left)
    right_norm = _normalize_fact(field, right)
    exact = bool(left_norm) and left_norm == right_norm
    similarity = 1.0 if exact else SequenceMatcher(None, left_norm, right_norm).ratio()
    # 姓名、日期、级别、奖项是离散事实，不能用模糊相似掩盖冲突。
    discrete = {"姓名", "获奖日期", "级别", "奖项/名次", "团队属性"}
    conflict_cutoff = 1.0 if field in discrete else 0.85 if field == "赛事名称" else 0.55
    return FieldSimilarity(
        field=field,
        left_value=left,
        right_value=right,
        left_normalized=left_norm,
        right_normalized=right_norm,
        similarity=round(similarity, 4),
        exact=exact,
        conflict=similarity < conflict_cutoff,
    )


def compare_evidence(
    left: Evidence,
    right: Evidence,
    *,
    thresholds: DuplicateThresholds | None = None,
) -> DuplicateDecision:
    """联合判断两份证据，返回逐信号原因与保守处置建议。"""
    limits = thresholds or DuplicateThresholds()
    if limits.phash_definite_max > limits.phash_suspected_max:
        raise ValueError("phash_definite_max 不能大于 phash_suspected_max")
    if limits.semantic_definite_min < limits.semantic_suspected_min:
        raise ValueError("semantic_definite_min 不能小于 semantic_suspected_min")

    left_sha = file_sha256(left.path)
    right_sha = file_sha256(right.path)
    sha_match = left_sha == right_sha if left_sha is not None and right_sha is not None else None
    distance = phash_distance(left.phash, right.phash)
    left_fp = fact_fingerprint(left)
    right_fp = fact_fingerprint(right)
    fp_match = left_fp == right_fp if left_fp is not None and right_fp is not None else None

    left_facts = evidence_facts(left)
    right_facts = evidence_facts(right)
    comparisons = [
        _field_similarity(field, left_facts[field], right_facts[field])
        for field in FACT_FIELDS
        if field in left_facts and field in right_facts
    ]
    semantic = (
        sum(item.similarity for item in comparisons) / len(comparisons) if comparisons else None
    )
    semantic = round(semantic, 4) if semantic is not None else None
    conflicts = {item.field for item in comparisons if item.conflict}
    critical_conflicts = conflicts & {"姓名", "赛事名称", "获奖日期"}
    exact_fields = {item.field for item in comparisons if item.exact}
    identity_present = {"姓名", "赛事名称"}.issubset(exact_fields)

    reasons: list[str] = []
    status: Literal["确定重复", "疑似重复", "非重复"]
    if sha_match:
        status = "确定重复"
        reasons.append("文件 SHA-256 完全相同")
        if conflicts:
            reasons.append("同一文件的结构化字段存在冲突，需核对抽取或人工修改记录")
    elif critical_conflicts:
        status = "非重复"
        reasons.append(f"关键事实冲突：{'、'.join(sorted(critical_conflicts))}")
        if distance is not None and distance <= limits.phash_suspected_max:
            reasons.append(f"虽然 pHash 距离为 {distance}，但关键事实冲突，按困难负样本放行")
    elif (
        fp_match
        and len(comparisons) >= limits.semantic_min_fields
        and identity_present
    ):
        status = "确定重复"
        reasons.append(f"{len(comparisons)} 个可比事实完全一致，字段事实指纹相同")
    elif (
        distance is not None
        and distance <= limits.phash_definite_max
        and (not comparisons or semantic is not None and semantic >= limits.semantic_suspected_min)
    ):
        status = "确定重复"
        reasons.append(f"pHash 距离 {distance} ≤ 确定阈值 {limits.phash_definite_max}")
        if comparisons:
            reasons.append(f"结构化事实相似度 {semantic:.4f}，未发现关键冲突")
    elif (
        semantic is not None
        and len(comparisons) >= limits.semantic_min_fields
        and identity_present
        and semantic >= limits.semantic_definite_min
    ):
        status = "确定重复"
        reasons.append(f"结构化事实相似度 {semantic:.4f} 达到确定阈值")
    elif distance is not None and distance <= limits.phash_suspected_max:
        status = "疑似重复"
        reasons.append(f"pHash 距离 {distance} ≤ 疑似阈值 {limits.phash_suspected_max}")
        if comparisons:
            reasons.append(f"结构化事实相似度 {semantic:.4f}，交由人工确认")
        else:
            reasons.append("缺少可比结构化事实，不能自动确认")
    elif (
        semantic is not None
        and len(comparisons) >= limits.semantic_min_fields
        and identity_present
        and semantic >= limits.semantic_suspected_min
    ):
        status = "疑似重复"
        reasons.append(f"结构化事实相似度 {semantic:.4f} 达到疑似阈值")
    else:
        status = "非重复"
        if conflicts:
            reasons.append(f"事实存在差异：{'、'.join(sorted(conflicts))}")
        else:
            reasons.append("文件、图像与结构化事实信号均未达到查重阈值")
        if not comparisons:
            reasons.append("缺少可比字段，本结果仅表示未命中，不证明材料真实或有效")

    if status == "确定重复":
        action: Literal["拦截并人工复核", "送人工复核", "放行"] = "拦截并人工复核"
    elif status == "疑似重复":
        action = "送人工复核"
    else:
        action = "放行"
    return DuplicateDecision(
        left_id=left.id,
        right_id=right.id,
        status=status,
        action=action,
        reasons=reasons,
        left_sha256=left_sha,
        right_sha256=right_sha,
        sha256_match=sha_match,
        phash_distance=distance,
        fact_fingerprint_match=fp_match,
        semantic_similarity=semantic,
        comparable_fields=len(comparisons),
        field_similarities=comparisons,
        thresholds=limits,
    )


__all__ = [
    "DuplicateDecision",
    "DuplicateThresholds",
    "FACT_FIELDS",
    "FieldSimilarity",
    "compare_evidence",
    "evidence_facts",
    "fact_fingerprint",
    "file_sha256",
    "phash_distance",
]
