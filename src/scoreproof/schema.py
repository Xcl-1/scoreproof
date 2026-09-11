"""核心数据模型：规则 rule / 申报 claim / 证据 evidence / 计算账本。

设计约束（来自项目总结第 5 节）：
1. ``synonyms`` 是必须做的脏活，也是壁垒；
2. ``source`` 是引用溯源的来源，任何分值都必须能回溯到原文出处；
3. 计算只依赖本模块的纯数据，不依赖任何 LLM 输出格式。
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import SchemaValidationError

# "2025-2026" 或 "2024" 都合法；不做"必须连续"的强校验，避免误伤真实脏数据
ACADEMIC_YEAR_RE = re.compile(r"^\d{4}(-\d{4})?$")

# ======================================================================
# 1) 规则库
# ======================================================================


class SourceRef(BaseModel):
    """原文出处：引用面板与人工复核的唯一依据。"""

    model_config = ConfigDict(extra="forbid")

    doc: str = Field(description="原始文件名，如 2025综测细则.pdf")
    page: int | None = Field(default=None, ge=1, description="页码（1-based）")
    table: str | None = Field(default=None, description="表格名，如 加分标准表")
    clause: str | None = Field(default=None, description="条款，如 第三章第7条")
    row: int | None = Field(default=None, description="表格行号，便于人工回查")
    text: str | None = Field(default=None, description="命中的原文片段")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="PDF 坐标 (x0,y0,x1,y1)，用于高亮定位"
    )

    def short(self) -> str:
        bits = [self.doc]
        if self.page is not None:
            bits.append(f"p{self.page}")
        if self.clause:
            bits.append(self.clause)
        if self.table:
            bits.append(self.table)
        return " / ".join(bits)


class ConstraintSpec(BaseModel):
    """规则的约束语义。全部由代码解释，绝不交给模型心算。"""

    model_config = ConfigDict(extra="forbid")

    dedup_group: str = Field(
        default="默认组",
        description="同类互斥组：同组内只取最高分，不累加",
    )
    cap: float | None = Field(
        default=None,
        ge=0,
        description="该规则单条最大计入分值；细则写'不设上限'则填 None",
    )
    team_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="团队奖折算系数，个人奖为 1.0",
    )
    require_catalog: str | None = Field(
        default=None,
        description="需要出现在该认可目录中才计分（资格认定），如 认可竞赛目录",
    )
    valid_years: list[str] | None = Field(
        default=None,
        description="限定学年；None 表示跟随 rule.academic_year",
    )
    exclusive_with: list[str] = Field(
        default_factory=list,
        description="与之互斥的 dedup_group 列表（如 竞赛与奖学金不可同时计）",
    )
    note: str | None = None


class Rule(BaseModel):
    """结构化加分规则：系统的确定性核心。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"r_{uuid.uuid4().hex[:10]}")
    academic_year: str = Field(description="学年，如 2025-2026")
    college: str | None = Field(default=None, description="学院；None 表示校级通用")
    category: str = Field(min_length=1, description="类别，如 学科竞赛")
    level: str = Field(min_length=1, description="规范化后的等级/名次，如 省级二等奖")
    score: float = Field(ge=0, description="该等级对应分值")
    synonyms: list[str] = Field(default_factory=list, description="原始表述与别名，用于归一化匹配")
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    source: SourceRef
    priority: int = Field(default=0, description="数值越大优先级越高，用于多份文件冲突裁决")
    enabled: bool = Field(default=True)
    raw_text: str | None = Field(default=None, description="抽取前的原文，便于人工校对")
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("academic_year")
    @classmethod
    def _check_year(cls, v: str) -> str:
        v = v.strip()
        if not ACADEMIC_YEAR_RE.match(v):
            raise ValueError(f"academic_year 需形如 '2025-2026' 或 '2024'，收到 {v!r}")
        return v

    @field_validator("level", "category")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("synonyms")
    @classmethod
    def _clean_synonyms(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for item in v:
            s = item.strip()
            if s:
                seen.setdefault(s, None)
        return list(seen)

    @property
    def effective_cap(self) -> float | None:
        """单条计分上限：None 表示不设额度。"""
        return self.constraints.cap

    @property
    def match_keys(self) -> list[str]:
        """可用于结构化查表的全部键：正式等级 + 同义词。"""
        return [self.level, *self.synonyms]

    def sort_key(self) -> tuple:
        """同分规则裁决顺序：优先级降序 -> 分值降序 -> id 升序（保证可复现）。"""
        return (-self.priority, -self.score, self.id)


class Ruleset(BaseModel):
    """一次性加载的规则库，带学年/学院过滤能力。"""

    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(default_factory=list)
    version: str = "0.1.0"
    meta: dict[str, Any] = Field(default_factory=dict)

    # ---------- 构造与 IO ----------

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Ruleset:
        ids = [r.id for r in self.rules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"规则 id 重复：{sorted(dupes)}")
        return self

    def add(self, rule: Rule) -> Ruleset:
        self.rules.append(rule)
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> Ruleset:
        p = Path(path)
        if not p.exists():
            raise SchemaValidationError(f"规则文件不存在：{p}", detail={"path": str(p)})
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - 防御性
            raise SchemaValidationError(f"规则文件不是合法 JSON：{exc}", detail={"path": str(p)}) from exc
        if isinstance(payload, list):
            payload = {"rules": payload}
        return cls.model_validate(payload)

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p

    # ---------- 查询 ----------

    def filter(
        self,
        *,
        academic_year: str | None = None,
        college: str | None = None,
        category: str | None = None,
        enabled_only: bool = True,
    ) -> Ruleset:
        """按学年/学院/类别过滤。

        学院语义：``college=None`` 的规则对任何学院都生效（校级通用）。
        """
        out: list[Rule] = []
        for r in self.rules:
            if enabled_only and not r.enabled:
                continue
            if academic_year and r.academic_year != academic_year:
                continue
            if college and r.college not in (None, college):
                continue
            if category and r.category != category:
                continue
            out.append(r)
        return Ruleset(rules=out, version=self.version, meta=dict(self.meta))

    def by_id(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    def groups(self) -> dict[str, list[Rule]]:
        acc: dict[str, list[Rule]] = {}
        for r in self.rules:
            acc.setdefault(r.constraints.dedup_group, []).append(r)
        return acc

    def build_index(
        self,
        *,
        academic_year: str | None = None,
        college: str | None = None,
    ) -> dict[str, list[Rule]]:
        """构建 ``匹配键 -> 规则列表`` 的精确查表索引（主通道）。"""
        index: dict[str, list[Rule]] = {}
        for r in self.filter(academic_year=academic_year, college=college).rules:
            for key in r.match_keys:
                index.setdefault(key, []).append(r)
        return index

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterable[Rule]:  # type: ignore[override]
        return iter(self.rules)


# ======================================================================
# 2) 申报条目
# ======================================================================

ClaimStatus = Literal["待核对", "已核对", "已驳回", "低置信", "未命中规则"]


class Claim(BaseModel):
    """学生申报条目：raw_text 是原始表述，level 是归一化结果。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"c_{uuid.uuid4().hex[:10]}")
    student_id: str = Field(min_length=1)
    student_name: str | None = None
    academic_year: str | None = None
    college: str | None = None
    category: str = Field(default="未分类")
    raw_text: str = Field(default="", description="学生填写的原始描述")
    level: str | None = Field(default=None, description="归一化后的等级/名次")
    team: bool = Field(default=False, description="是否为团队获奖")
    evidence_ids: list[str] = Field(default_factory=list)
    catalog_listed: bool = Field(default=True, description="是否在认可竞赛目录内")
    status: ClaimStatus = Field(default="待核对")
    source_ref: SourceRef | None = Field(default=None, description="申报条目在综测表中的位置")
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("academic_year")
    @classmethod
    def _check_year(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not ACADEMIC_YEAR_RE.match(v):
            raise ValueError(f"academic_year 需形如 '2025-2026'，收到 {v!r}")
        return v


# ======================================================================
# 3) 证据（多模态核心）
# ======================================================================

EvidenceType = Literal["image", "pdf", "docx", "excel", "text"]
ExtractorKind = Literal["ocr+llm", "vlm", "manual", "regex", "table"]


class Evidence(BaseModel):
    """证据：字段 + OCR 原文 + 图片路径 + 感知哈希 + 出处。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:10]}")
    type: EvidenceType
    path: str | None = Field(default=None, description="原始文件路径（不入公开仓库）")
    ocr_text: str | None = Field(default=None, description="OCR 原文，可溯源")
    fields: dict[str, Any] = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    phash: str | None = Field(default=None, description="感知哈希，用于重复申报检测")
    extractor: ExtractorKind = "manual"
    manual_corrected: bool = False
    source_locator: str | None = Field(default=None, description="如 cert#region_1")
    captured_at: date | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_confidence_range(self) -> Evidence:
        for field, conf in self.field_confidence.items():
            if not 0.0 <= conf <= 1.0:
                raise ValueError(f"字段 {field} 的置信度需在 [0,1]，收到 {conf}")
        return self

    @property
    def min_confidence(self) -> float:
        """整体置信度取各字段最小值：任一项不确定就值得复核。"""
        return min(self.field_confidence.values(), default=1.0)

    def requires_review(self, threshold: float = 0.8) -> bool:
        return self.min_confidence < threshold or not self.manual_corrected

    def fingerprint(self, keys: tuple[str, ...] = ("赛事", "等级", "时间", "姓名")) -> str:
        """字段指纹：与 pHash 互补，用于拦截"一证多报"的截图变体。"""
        import hashlib

        parts = [str(self.fields.get(k, "")).strip() for k in keys]
        if not any(parts) and self.ocr_text:
            parts = [re.sub(r"\s+", "", self.ocr_text)]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


# ======================================================================
# 4) 计算结果账本（可解释性）
# ======================================================================


class RuleMatch(BaseModel):
    """一条申报 -> 规则的命中记录。"""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    rule_id: str | None
    matched_key: str | None = Field(default=None, description="命中的键（正式等级或同义词）")
    score: float = Field(default=0.0, description="计入分值（已折算/封顶，未去重）")
    raw_score: float = Field(default=0.0, description="规则分值（未折算）")
    team_factor: float = 1.0
    capped: bool = Field(default=False, description="是否被本规则 cap 截断")
    dedup_group: str | None = None
    counted: bool = Field(default=True, description="去重后是否真正计入")
    channel: Literal["structured", "vector", "none"] = "structured"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_review: bool = False
    reason: str | None = Field(default=None, description="未命中或不计入的原因")
    source: SourceRef | None = None


class GroupBreakdown(BaseModel):
    """一个互斥组的账目。"""

    model_config = ConfigDict(extra="forbid")

    group: str
    cap: float | None = None
    candidates: list[RuleMatch] = Field(default_factory=list)
    winner: RuleMatch | None = None
    subtotal: float = 0.0
    after_cap: float = 0.0
    capped: bool = False


class ScoreBreakdown(BaseModel):
    """最终结果：可逐条回溯到规则与原文。"""

    model_config = ConfigDict(extra="forbid")

    student_id: str
    academic_year: str | None = None
    college: str | None = None
    total: float = 0.0
    groups: list[GroupBreakdown] = Field(default_factory=list)
    matches: list[RuleMatch] = Field(default_factory=list)
    unmatched_claims: list[str] = Field(default_factory=list)
    review_claims: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


__all__ = [
    "ACADEMIC_YEAR_RE",
    "Claim",
    "ClaimStatus",
    "ConstraintSpec",
    "Evidence",
    "EvidenceType",
    "ExtractorKind",
    "GroupBreakdown",
    "Rule",
    "RuleMatch",
    "Ruleset",
    "ScoreBreakdown",
    "SourceRef",
]
