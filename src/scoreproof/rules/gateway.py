"""LLM 规则抽取的代码级验证网关。

模型输出在本模块中始终被视为不可信数据。只有五道校验全部通过，且
未触发独立的发布前冲突门禁，``GatewayItem`` 才能转换为正式 ``Rule``。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..normalize import normalize_academic_year, normalize_level
from ..schema import ACADEMIC_YEAR_RE, ConstraintSpec, Rule, SourceRef

GatewayStatus = Literal["accepted", "review", "rejected"]
GatewayStage = Literal["validation", "publication"]
GatewayCode = Literal[
    "schema_error",
    "score_not_in_quote",
    "level_not_in_quote",
    "quote_not_found",
    "score_out_of_range",
    "invalid_level",
    "invalid_academic_year",
    "invalid_effective_date",
    "effective_date_out_of_year",
    "invalid_cap",
    "invalid_team_factor",
    "secondary_missing",
    "secondary_schema_error",
    "secondary_mismatch",
    "rule_conflict",
]


class RuleDraftInput(BaseModel):
    """LLM 可提交的唯一输入契约；禁止额外字段和隐式字符串转数值。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: str = Field(min_length=1, description="加分类别，如 学科竞赛")
    level: str = Field(min_length=1, description="等级/名次原文，如 省级二等奖")
    score: float = Field(description="原文明确出现的分值")
    synonyms: list[str] = Field(default_factory=list, description="原文同义表述")
    cap: float | None = Field(default=None, description="单项封顶；原文没有则为 null")
    team_factor: float | None = Field(default=None, description="团队折算系数")
    clause: str | None = Field(default=None, description="条款编号，如 第三章第7条")
    evidence_quote: str = Field(description="支撑规则的逐字原文片段")
    rank: str | None = Field(default=None, description="独立名次，无法拆分则为 null")
    item_name: str | None = Field(default=None, description="赛事或项目名称")
    effective_date: str | None = Field(default=None, description="ISO 日期；原文没有则为 null")

    @field_validator(
        "category", "level", "evidence_quote", "clause", "rank", "item_name", "effective_date"
    )
    @classmethod
    def _strip_strings(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("synonyms")
    @classmethod
    def _clean_synonyms(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class GatewayContext(BaseModel):
    """一次抽取批次锁定的来源与适用范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    academic_year: str
    doc: str = Field(min_length=1)
    college: str | None = None
    page: int | None = Field(default=None, ge=1)
    table: str | None = None
    priority: int = 0
    allowed_levels: frozenset[str] = Field(default_factory=frozenset)


class GatewayIssue(BaseModel):
    """稳定、可统计的单条拦截原因。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: GatewayStage = "validation"
    layer: Literal[1, 2, 3, 4, 5] | None
    code: GatewayCode
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_stage_and_layer(self) -> GatewayIssue:
        if self.stage == "validation" and self.layer is None:
            raise ValueError("五道验证问题必须标明 layer")
        if self.stage == "publication" and self.layer is not None:
            raise ValueError("发布前门禁不是第六层，layer 必须为 null")
        if self.code == "rule_conflict" and self.stage != "publication":
            raise ValueError("rule_conflict 必须属于独立发布前门禁")
        if self.code != "rule_conflict" and self.stage == "publication":
            raise ValueError("发布前门禁目前只接受 rule_conflict")
        return self


class GatewayItem(BaseModel):
    """单条草稿的验证结果。"""

    model_config = ConfigDict(extra="forbid")

    index: int
    raw_payload: dict[str, Any]
    draft: RuleDraftInput | None = None
    status: GatewayStatus = "accepted"
    issues: list[GatewayIssue] = Field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    source_chunk_hash: str | None = None
    extract_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity_flag: bool = False

    @property
    def publishable(self) -> bool:
        return self.status == "accepted" and self.draft is not None and not self.issues

    def add_issue(self, issue: GatewayIssue, *, status: GatewayStatus = "review") -> None:
        if issue.code not in {current.code for current in self.issues}:
            self.issues.append(issue)
        if self.status != "rejected":
            self.status = status
        if issue.layer == 2:
            self.extract_confidence = 0.0
        elif self.extract_confidence > 0.5:
            self.extract_confidence = 0.5
        if issue.layer == 5:
            self.ambiguity_flag = True

    def to_rule(self, context: GatewayContext) -> Rule:
        """将已通过网关的条目转换为正式规则。"""
        if not self.publishable or self.draft is None:
            raise ValueError("只有通过五道验证及发布前冲突门禁的条目才能转换为 Rule")
        draft = self.draft
        canonical = normalize_level(draft.level).canonical
        return Rule(
            academic_year=normalize_academic_year(context.academic_year),
            college=context.college,
            category=draft.category,
            level=canonical,
            rank=draft.rank,
            item_name=draft.item_name,
            score=draft.score,
            synonyms=sorted(
                {item for item in [draft.level, *draft.synonyms] if item and item != canonical}
            ),
            constraints=ConstraintSpec(
                dedup_group=draft.category,
                cap=draft.cap,
                team_factor=draft.team_factor if draft.team_factor is not None else 1.0,
            ),
            source=SourceRef(
                doc=context.doc,
                page=context.page,
                table=context.table,
                clause=draft.clause,
                text=draft.evidence_quote[:200],
                chunk_hash=self.source_chunk_hash,
                char_start=self.char_start,
                char_end=self.char_end,
            ),
            priority=context.priority,
            raw_text=draft.evidence_quote[:500],
        )


class GatewayReport(BaseModel):
    """一次网关运行的可审计结果。"""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(default_factory=lambda: f"xb_{uuid.uuid4().hex[:12]}")
    chunk_hash: str
    context: GatewayContext
    items: list[GatewayItem]
    secondary_used: bool = False

    @property
    def publishable(self) -> bool:
        return bool(self.items) and all(item.publishable for item in self.items)

    def rules(self) -> list[Rule]:
        if not self.publishable:
            raise ValueError("批次包含被拦截或待复核条目，不能发布")
        return [item.to_rule(self.context) for item in self.items]

    def metrics(self) -> dict[str, Any]:
        reason_counts = Counter(issue.code for item in self.items for issue in item.issues)
        validation_reasons = Counter(
            issue.code
            for item in self.items
            for issue in item.issues
            if issue.stage == "validation"
        )
        publication_reasons = Counter(
            issue.code
            for item in self.items
            for issue in item.issues
            if issue.stage == "publication"
        )
        status_counts = Counter(item.status for item in self.items)
        total = len(self.items)
        blocked = total - status_counts.get("accepted", 0)
        return {
            "batch_id": self.batch_id,
            "chunk_hash": self.chunk_hash,
            "sample_size": total,
            "accepted": status_counts.get("accepted", 0),
            "review": status_counts.get("review", 0),
            "rejected": status_counts.get("rejected", 0),
            "intercepted": blocked,
            "interception_rate": blocked / total if total else 0.0,
            "secondary_used": self.secondary_used,
            "reasons": dict(sorted(reason_counts.items())),
            "validation_reasons": dict(sorted(validation_reasons.items())),
            "publication_gate_reasons": dict(sorted(publication_reasons.items())),
            "publishable": self.publishable,
        }


def chunk_hash(text: str) -> str:
    """对规范化逻辑块生成稳定 SHA-256，供抽取与模型调用缓存复用。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = "\n".join(line.strip() for line in normalized.splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


_ARABIC_SCORE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])\s*分")
_CHINESE_SCORE_RE = re.compile(r"([零〇一二两三四五六七八九十百]+)\s*分")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _parse_chinese_number(raw: str) -> int | None:
    if raw == "百" or raw == "一百":
        return 100
    if "百" in raw:
        return None
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(raw) == 1:
        return _CN_DIGITS.get(raw)
    return None


def _score_occurs(score: float, quote: str) -> bool:
    values = {float(match.group(1)) for match in _ARABIC_SCORE_RE.finditer(quote)}
    for match in _CHINESE_SCORE_RE.finditer(quote):
        parsed = _parse_chinese_number(match.group(1))
        if parsed is not None:
            values.add(float(parsed))
    return float(score) in values


def _level_occurs(level: str, quote: str) -> bool:
    compact_level = re.sub(r"\s+", "", unicodedata.normalize("NFKC", level))
    compact_quote = re.sub(r"\s+", "", unicodedata.normalize("NFKC", quote))
    if compact_level and compact_level in compact_quote:
        return True
    wanted = normalize_level(level).canonical
    found = normalize_level(quote).canonical
    return bool(wanted and found and (wanted in found or found in wanted))


def _semantic_signature(draft: RuleDraftInput) -> str:
    payload = {
        "category": draft.category,
        "level": normalize_level(draft.level).canonical,
        "rank": draft.rank,
        "item_name": draft.item_name,
        "score": draft.score,
        "cap": draft.cap,
        "team_factor": draft.team_factor,
        "clause": draft.clause,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _identity_key(draft: RuleDraftInput, context: GatewayContext) -> tuple[str, ...]:
    return (
        normalize_academic_year(context.academic_year),
        context.college or "",
        draft.category,
        normalize_level(draft.level).canonical,
        draft.rank or "",
        draft.item_name or "",
    )


def _rule_identity_key(rule: Rule) -> tuple[str, ...]:
    return (
        normalize_academic_year(rule.academic_year),
        rule.college or "",
        rule.category,
        normalize_level(rule.level).canonical,
        rule.rank or "",
        rule.item_name or "",
    )


class ExtractionGateway:
    """实现计划书 5.2 的五道抽取验证及独立发布前冲突门禁。"""

    def validate_batch(
        self,
        payloads: Sequence[Mapping[str, Any]],
        *,
        source_text: str,
        context: GatewayContext,
        secondary_payloads: Sequence[Mapping[str, Any]] | None = None,
        existing_rules: Iterable[Rule] = (),
    ) -> GatewayReport:
        digest = chunk_hash(source_text)
        items = [
            self._validate_primary(index, dict(payload), source_text=source_text, context=context)
            for index, payload in enumerate(payloads)
        ]
        for item in items:
            item.source_chunk_hash = digest
        if secondary_payloads is not None:
            self._cross_validate(items, secondary_payloads, context=context)
        self._detect_conflicts(items, context=context, existing_rules=existing_rules)
        return GatewayReport(
            chunk_hash=digest,
            context=context,
            items=items,
            secondary_used=secondary_payloads is not None,
        )

    def _validate_primary(
        self,
        index: int,
        payload: dict[str, Any],
        *,
        source_text: str,
        context: GatewayContext,
    ) -> GatewayItem:
        item = GatewayItem(index=index, raw_payload=payload)
        try:
            draft = RuleDraftInput.model_validate(payload)
        except ValidationError as exc:
            item.add_issue(
                GatewayIssue(
                    layer=1,
                    code="schema_error",
                    message="模型输出不符合严格 RuleDraft schema",
                    detail={"errors": exc.errors(include_url=False, include_input=False)},
                ),
                status="rejected",
            )
            return item
        item.draft = draft

        if not _score_occurs(draft.score, draft.evidence_quote):
            item.add_issue(
                GatewayIssue(
                    layer=2,
                    code="score_not_in_quote",
                    message="分值未在 evidence_quote 中出现",
                    detail={"score": draft.score},
                ),
                status="rejected",
            )
        if not _level_occurs(draft.level, draft.evidence_quote):
            item.add_issue(
                GatewayIssue(
                    layer=2,
                    code="level_not_in_quote",
                    message="等级未在 evidence_quote 中出现",
                    detail={"level": draft.level},
                ),
                status="rejected",
            )

        quote = draft.evidence_quote
        start = source_text.find(quote) if quote else -1
        if start < 0:
            item.add_issue(
                GatewayIssue(
                    layer=3,
                    code="quote_not_found",
                    message="evidence_quote 无法在原文块中定位",
                )
            )
        else:
            item.char_start = start
            item.char_end = start + len(quote)

        self._validate_boundaries(item, context=context)
        return item

    @staticmethod
    def _validate_boundaries(item: GatewayItem, *, context: GatewayContext) -> None:
        assert item.draft is not None
        draft = item.draft
        if not 0 < draft.score <= 100:
            item.add_issue(
                GatewayIssue(layer=4, code="score_out_of_range", message="score 必须满足 0 < score ≤ 100")
            )
        normalized_level = normalize_level(draft.level)
        allowed = {normalize_level(value).canonical for value in context.allowed_levels}
        if not normalized_level.matched and normalized_level.canonical not in allowed:
            item.add_issue(
                GatewayIssue(
                    layer=4,
                    code="invalid_level",
                    message="level 不在已知等级枚举或批次允许列表中",
                    detail={"level": draft.level},
                )
            )
        academic_year = normalize_academic_year(context.academic_year)
        if not ACADEMIC_YEAR_RE.fullmatch(academic_year):
            item.add_issue(
                GatewayIssue(
                    layer=4,
                    code="invalid_academic_year",
                    message="academic_year 格式非法",
                    detail={"academic_year": context.academic_year},
                )
            )
        if draft.cap is not None and not 0 <= draft.cap <= 100:
            item.add_issue(
                GatewayIssue(layer=4, code="invalid_cap", message="cap 必须位于 0～100")
            )
        if draft.team_factor is not None and not 0 <= draft.team_factor <= 1:
            item.add_issue(
                GatewayIssue(layer=4, code="invalid_team_factor", message="team_factor 必须位于 0～1")
            )
        if draft.effective_date:
            try:
                parsed = date.fromisoformat(draft.effective_date)
            except ValueError:
                item.add_issue(
                    GatewayIssue(
                        layer=4,
                        code="invalid_effective_date",
                        message="effective_date 必须是 ISO 日期",
                        detail={"effective_date": draft.effective_date},
                    )
                )
            else:
                match = re.fullmatch(r"(\d{4})-(\d{4})", academic_year)
                if match:
                    start_year, end_year = int(match.group(1)), int(match.group(2))
                    start_date = date(start_year, 9, 1)
                    end_date = date(end_year, 8, 31)
                    if not start_date <= parsed <= end_date:
                        item.add_issue(
                            GatewayIssue(
                                layer=4,
                                code="effective_date_out_of_year",
                                message="effective_date 不在批次学年范围内",
                                detail={"effective_date": draft.effective_date},
                            )
                        )

    @staticmethod
    def _cross_validate(
        items: list[GatewayItem],
        secondary_payloads: Sequence[Mapping[str, Any]],
        *,
        context: GatewayContext,
    ) -> None:
        secondary: dict[tuple[str, ...], list[RuleDraftInput]] = {}
        secondary_schema_failed = False
        for payload in secondary_payloads:
            try:
                draft = RuleDraftInput.model_validate(dict(payload))
            except ValidationError:
                secondary_schema_failed = True
                continue
            secondary.setdefault(_identity_key(draft, context), []).append(draft)

        for item in items:
            if item.draft is None:
                continue
            matches = secondary.get(_identity_key(item.draft, context), [])
            if not matches:
                code: GatewayCode = "secondary_schema_error" if secondary_schema_failed else "secondary_missing"
                item.add_issue(
                    GatewayIssue(layer=5, code=code, message="第二次独立抽取缺少可比较条目")
                )
                continue
            signatures = {_semantic_signature(value) for value in matches}
            if len(matches) != 1 or _semantic_signature(item.draft) not in signatures:
                item.add_issue(
                    GatewayIssue(
                        layer=5,
                        code="secondary_mismatch",
                        message="两次独立抽取结果不一致",
                    )
                )

    @staticmethod
    def _detect_conflicts(
        items: list[GatewayItem],
        *,
        context: GatewayContext,
        existing_rules: Iterable[Rule],
    ) -> None:
        scores_by_key: dict[tuple[str, ...], set[float]] = {}
        item_keys: dict[int, tuple[str, ...]] = {}
        for rule in existing_rules:
            scores_by_key.setdefault(_rule_identity_key(rule), set()).add(rule.score)
        for item in items:
            if item.draft is None:
                continue
            key = _identity_key(item.draft, context)
            item_keys[item.index] = key
            scores_by_key.setdefault(key, set()).add(item.draft.score)
        conflicting = {key for key, scores in scores_by_key.items() if len(scores) > 1}
        for item in items:
            item_key = item_keys.get(item.index)
            if item_key in conflicting:
                assert item_key is not None
                item.add_issue(
                    GatewayIssue(
                        stage="publication",
                        layer=None,
                        code="rule_conflict",
                        message="同一适用范围与规则键出现多个分值，批次不可发布",
                        detail={"key": list(item_key), "scores": sorted(scores_by_key[item_key])},
                    )
                )


__all__ = [
    "ExtractionGateway",
    "GatewayCode",
    "GatewayContext",
    "GatewayIssue",
    "GatewayItem",
    "GatewayReport",
    "GatewayStage",
    "GatewayStatus",
    "RuleDraftInput",
    "chunk_hash",
]
