"""申报与证据的确定性逐字段一致性比对。"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..normalize import academic_year_of, normalize_text_key, parse_prize, parse_tier
from ..schema import Claim, Evidence
from .dedup import evidence_facts

MatchStatus = Literal["完全一致", "归一化一致", "不一致", "信息不足"]


class ConsistencyPolicy(BaseModel):
    """学院可注入的赛事别名、目录与颁发单位约束。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_aliases: dict[str, str] = Field(default_factory=dict)
    category_aliases: dict[str, str] = Field(default_factory=dict)
    allowed_issuers: list[str] = Field(default_factory=list)
    catalog_events: list[str] = Field(default_factory=list)


class ConsistencyField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field: str
    claim_value: str | None = None
    evidence_value: str | None = None
    claim_normalized: str | None = None
    evidence_normalized: str | None = None
    status: MatchStatus
    reason: str


class EvidenceConsistencyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: str
    evidence_id: str
    status: Literal["通过", "需人工复核", "不一致"]
    requires_review: bool
    fields: list[ConsistencyField]
    mismatch_fields: list[str]
    insufficient_fields: list[str]
    notes: list[str] = Field(default_factory=list)


def _text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _alias_normalize(value: str, aliases: dict[str, str]) -> str:
    key = normalize_text_key(value).lower()
    for alias, canonical in aliases.items():
        if key == normalize_text_key(alias).lower():
            return normalize_text_key(canonical).lower()
    return key


def _compare(
    field: str,
    claim_value: str | None,
    evidence_value: str | None,
    *,
    normalizer=lambda value: normalize_text_key(value).lower(),
) -> ConsistencyField:
    if claim_value is None or evidence_value is None:
        missing = "申报" if claim_value is None else "证据"
        return ConsistencyField(
            field=field,
            claim_value=claim_value,
            evidence_value=evidence_value,
            status="信息不足",
            reason=f"{missing}侧缺少可比值",
        )
    claim_norm = normalizer(claim_value)
    evidence_norm = normalizer(evidence_value)
    if claim_value.strip() == evidence_value.strip():
        status: MatchStatus = "完全一致"
        reason = "原始值完全一致"
    elif claim_norm and claim_norm == evidence_norm:
        status = "归一化一致"
        reason = "经确定性归一化后一致"
    else:
        status = "不一致"
        reason = "归一化后仍不一致"
    return ConsistencyField(
        field=field,
        claim_value=claim_value,
        evidence_value=evidence_value,
        claim_normalized=claim_norm,
        evidence_normalized=evidence_norm,
        status=status,
        reason=reason,
    )


def _date_academic_year(value: str) -> str:
    cleaned = value.strip().replace("年", "-").replace("月", "-").replace("日", "")
    parts = [part for part in cleaned.replace("/", "-").replace(".", "-").split("-") if part]
    if len(parts) != 3:
        return ""
    try:
        parsed = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return ""
    return academic_year_of(parsed)


def _team_value(value: str) -> str:
    key = normalize_text_key(value).lower()
    if key in {"团队", "团体", "集体", "team", "true", "是"}:
        return "团队"
    if key in {"个人", "individual", "false", "否"}:
        return "个人"
    return key


def compare_claim_evidence(
    claim: Claim,
    evidence: Evidence,
    *,
    policy: ConsistencyPolicy | None = None,
) -> EvidenceConsistencyReport:
    """按姓名、赛事、等级奖项、学年、团队、目录/单位和类别逐项比对。"""
    rules = policy or ConsistencyPolicy()
    facts = evidence_facts(evidence)
    extra = claim.extra
    event_claim = _text(extra.get("event_name") or extra.get("赛事名称"))
    tier_claim = _text(extra.get("tier") or extra.get("级别")) or parse_tier(
        claim.level or claim.raw_text
    )
    award_claim = _text(extra.get("award") or extra.get("奖项/名次")) or parse_prize(
        claim.level or claim.raw_text
    )
    issuer_claim = _text(extra.get("issuer") or extra.get("颁发单位"))
    category_evidence = _text(evidence.fields.get("申报类别") or evidence.fields.get("category"))
    def event_normalizer(value: str) -> str:
        return _alias_normalize(value, rules.event_aliases)

    def category_normalizer(value: str) -> str:
        return _alias_normalize(value, rules.category_aliases)
    fields = [
        _compare("姓名", claim.student_name, facts.get("姓名")),
        _compare("赛事名称", event_claim, facts.get("赛事名称"), normalizer=event_normalizer),
        _compare("级别", tier_claim, facts.get("级别"), normalizer=lambda value: parse_tier(value) or ""),
        _compare(
            "奖项/名次",
            award_claim,
            facts.get("奖项/名次"),
            normalizer=lambda value: parse_prize(value) or "",
        ),
        _compare(
            "获奖日期/学年",
            claim.academic_year,
            facts.get("获奖日期"),
            normalizer=lambda value: (
                value if "-" in value and len(value) == 9 else _date_academic_year(value)
            ),
        ),
        _compare(
            "团队属性",
            "团队" if claim.team else "个人",
            facts.get("团队属性"),
            normalizer=_team_value,
        ),
        _compare("申报类别", claim.category, category_evidence, normalizer=category_normalizer),
    ]

    evidence_issuer = facts.get("颁发单位")
    if issuer_claim is not None:
        fields.append(_compare("颁发单位", issuer_claim, evidence_issuer))
    elif rules.allowed_issuers:
        normalized_allowed = {normalize_text_key(item).lower() for item in rules.allowed_issuers}
        issuer_norm = normalize_text_key(evidence_issuer or "").lower()
        fields.append(
            ConsistencyField(
                field="颁发单位",
                claim_value=" / ".join(rules.allowed_issuers),
                evidence_value=evidence_issuer,
                claim_normalized="|".join(sorted(normalized_allowed)),
                evidence_normalized=issuer_norm or None,
                status=(
                    "归一化一致"
                    if issuer_norm in normalized_allowed
                    else "信息不足" if not issuer_norm else "不一致"
                ),
                reason=(
                    "颁发单位命中允许清单"
                    if issuer_norm in normalized_allowed
                    else "证据缺少颁发单位" if not issuer_norm else "颁发单位未命中允许清单"
                ),
            )
        )
    else:
        fields.append(_compare("颁发单位", None, evidence_issuer))

    if rules.catalog_events:
        event_norm = event_normalizer(facts.get("赛事名称", ""))
        catalog = {event_normalizer(item) for item in rules.catalog_events}
        fields.append(
            ConsistencyField(
                field="目录赛事",
                claim_value="在目录内" if claim.catalog_listed else "不在目录内",
                evidence_value=facts.get("赛事名称"),
                claim_normalized=str(claim.catalog_listed).lower(),
                evidence_normalized=event_norm or None,
                status=(
                    "归一化一致"
                    if bool(event_norm in catalog) == claim.catalog_listed
                    else "信息不足" if not event_norm else "不一致"
                ),
                reason="按注入的赛事目录确定性核对",
            )
        )
    else:
        fields.append(
            ConsistencyField(
                field="目录赛事",
                claim_value="在目录内" if claim.catalog_listed else "不在目录内",
                evidence_value=facts.get("赛事名称"),
                status="信息不足",
                reason="未注入适用学院/学年的赛事目录",
            )
        )

    mismatches = [item.field for item in fields if item.status == "不一致"]
    insufficient = [item.field for item in fields if item.status == "信息不足"]
    if mismatches:
        overall: Literal["通过", "需人工复核", "不一致"] = "不一致"
    elif insufficient:
        overall = "需人工复核"
    else:
        overall = "通过"
    notes = ["LLM 不参与最终一致性裁决；结果来自字段归一化与显式策略。"]
    if insufficient:
        notes.append("信息不足字段不能按一致处理，必须补录或人工复核。")
    return EvidenceConsistencyReport(
        claim_id=claim.id,
        evidence_id=evidence.id,
        status=overall,
        requires_review=overall != "通过",
        fields=fields,
        mismatch_fields=mismatches,
        insufficient_fields=insufficient,
        notes=notes,
    )


__all__ = [
    "ConsistencyField",
    "ConsistencyPolicy",
    "EvidenceConsistencyReport",
    "MatchStatus",
    "compare_claim_evidence",
]
