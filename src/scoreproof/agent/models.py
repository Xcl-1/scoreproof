"""工具编排层的公开数据契约。"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from ..retrieval.citation import CitationCheck
from ..schema import Claim, ScoreBreakdown


class LookupRuleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    academic_year: StrictStr
    college: StrictStr | None = None
    category: StrictStr
    level: StrictStr
    rank: StrictStr | None = None
    item_name: StrictStr | None = None


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    academic_year: StrictStr | None = None
    college: StrictStr | None = None
    doc_id: StrictStr | None = None


class SearchClausesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: StrictStr = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: StrictInt = Field(default=5, ge=1, le=20)


class CalcClaimInput(BaseModel):
    """确定性核算入参；刻意不接受一整段自然语言或模型提供的分值。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: StrictStr
    academic_year: StrictStr | None = None
    college: StrictStr | None = None
    category: StrictStr
    level: StrictStr
    team: StrictBool = False
    catalog_listed: StrictBool = True

    def to_claim(self, *, student_id: str) -> Claim:
        return Claim(
            id=self.claim_id,
            student_id=student_id,
            academic_year=self.academic_year,
            college=self.college,
            category=self.category,
            level=self.level,
            raw_text=self.level,
            team=self.team,
            catalog_listed=self.catalog_listed,
        )

    @classmethod
    def from_claim(cls, claim: Claim) -> CalcClaimInput:
        level = claim.level or claim.raw_text.strip()
        return cls(
            claim_id=claim.id,
            academic_year=claim.academic_year,
            college=claim.college,
            category=claim.category,
            level=level,
            team=claim.team,
            catalog_listed=claim.catalog_listed,
        )


class CalcScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[CalcClaimInput] = Field(min_length=1)
    ruleset_version: StrictStr
    student_id: StrictStr = Field(min_length=1)


class CheckEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim: dict[str, Any]
    evidence_fields: dict[str, Any]


class AskClarificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    missing_fields: list[StrictStr] = Field(min_length=1)
    question: StrictStr = Field(min_length=1)


class FieldDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    claim_value: Any = None
    evidence_value: Any = None
    matches: bool
    reason: str


class OrchestrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    session_id: str = Field(default_factory=lambda: f"s_{uuid.uuid4().hex}")
    claims: list[Claim] = Field(default_factory=list)
    academic_year: str | None = None
    college: str | None = None


class NumberValidation(BaseModel):
    valid: bool
    seen: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["answer", "clarification", "refusal"]
    answer: str
    session_id: str
    ruleset_version: str
    ledger: ScoreBreakdown | None = None
    degraded: bool = False
    model_version: str
    state_trace: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    number_validation: NumberValidation | None = None
    citation_check: CitationCheck | None = None
    blocked_answer_count: int = 0


__all__ = [
    "AskClarificationInput",
    "CalcClaimInput",
    "CalcScoreInput",
    "CheckEvidenceInput",
    "FieldDiff",
    "LookupRuleInput",
    "NumberValidation",
    "OrchestrationRequest",
    "OrchestrationResult",
    "SearchClausesInput",
    "SearchFilters",
]
