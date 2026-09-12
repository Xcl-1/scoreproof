"""抽取验证网关的可复跑负例评测。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..rules.gateway import ExtractionGateway, GatewayCode, GatewayContext, GatewayIssue

GatewayTarget = Literal[
    "layer_1",
    "layer_2",
    "layer_3",
    "layer_4",
    "layer_5",
    "publication_conflict",
]
_TARGET_CODES: dict[str, frozenset[str]] = {
    "layer_1": frozenset({"schema_error"}),
    "layer_2": frozenset({"score_not_in_quote", "level_not_in_quote"}),
    "layer_3": frozenset({"quote_not_found"}),
    "layer_4": frozenset(
        {
            "score_out_of_range",
            "invalid_level",
            "invalid_academic_year",
            "invalid_effective_date",
            "effective_date_out_of_year",
            "invalid_cap",
            "invalid_team_factor",
        }
    ),
    "layer_5": frozenset(
        {"secondary_missing", "secondary_schema_error", "secondary_mismatch"}
    ),
    "publication_conflict": frozenset({"rule_conflict"}),
}


class GatewayNegativeCase(BaseModel):
    """人工注入的不可信抽取场景及其预期拦截点。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    target: GatewayTarget
    expected_code: GatewayCode
    source_text: str
    payloads: list[dict[str, Any]] = Field(min_length=1)
    secondary_payloads: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _validate_expected_target(self) -> GatewayNegativeCase:
        if self.expected_code not in _TARGET_CODES[self.target]:
            raise ValueError(
                f"expected_code={self.expected_code!r} 不属于 target={self.target!r}"
            )
        return self


class GatewayTargetResult(BaseModel):
    """某一道校验或发布门禁的独立检出结果。"""

    model_config = ConfigDict(extra="forbid")

    sample_size: int
    detected: int
    detection_rate: float
    wilson_lower: float
    wilson_upper: float


class GatewayEvaluationReport(BaseModel):
    """带样本量、Wilson 区间和版本哈希的网关评测报告。"""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    config_hash: str
    sample_size: int
    detected: int
    missed_case_ids: list[str]
    detection_rate: float
    confidence_level: float
    wilson_lower: float
    wilson_upper: float
    reasons: dict[str, int]
    target_results: dict[str, GatewayTargetResult]


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """计算二项比例的 Wilson score interval，不依赖 SciPy。"""
    if total <= 0:
        raise ValueError("total 必须大于 0")
    if not 0 <= successes <= total:
        raise ValueError("successes 必须位于 0 与 total 之间")
    if not 0 < confidence < 1:
        raise ValueError("confidence 必须位于 0 与 1 之间")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


def evaluate_gateway_negatives(
    cases: Sequence[GatewayNegativeCase],
    *,
    context: GatewayContext,
    dataset_version: str,
    gateway: ExtractionGateway | None = None,
    confidence: float = 0.95,
) -> GatewayEvaluationReport:
    """按计划书 13.2 口径评估五道校验和独立发布冲突门禁。"""
    if not cases:
        raise ValueError("负例集不能为空")
    checker = gateway or ExtractionGateway()
    detected = 0
    missed: list[str] = []
    reasons: Counter[str] = Counter()
    target_totals: Counter[str] = Counter()
    target_detected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            raise ValueError(f"负例 id 重复：{case.id}")
        seen_ids.add(case.id)
        report = checker.validate_batch(
            case.payloads,
            source_text=case.source_text,
            context=context,
            secondary_payloads=case.secondary_payloads,
        )
        issues = [issue for item in report.items for issue in item.issues]
        reasons.update(issue.code for issue in issues)
        target_totals[case.target] += 1
        if any(_matches_target(issue, case) for issue in issues):
            detected += 1
            target_detected[case.target] += 1
        else:
            missed.append(case.id)
    lower, upper = wilson_interval(detected, len(cases), confidence=confidence)
    target_results: dict[str, GatewayTargetResult] = {}
    for target, total in sorted(target_totals.items()):
        successes = target_detected[target]
        target_lower, target_upper = wilson_interval(successes, total, confidence=confidence)
        target_results[target] = GatewayTargetResult(
            sample_size=total,
            detected=successes,
            detection_rate=successes / total,
            wilson_lower=target_lower,
            wilson_upper=target_upper,
        )
    config_payload = {
        "context": context.model_dump(mode="json"),
        "confidence": confidence,
        "gateway": "five-layer-plus-publication-conflict-v2",
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return GatewayEvaluationReport(
        dataset_version=dataset_version,
        config_hash=config_hash,
        sample_size=len(cases),
        detected=detected,
        missed_case_ids=missed,
        detection_rate=detected / len(cases),
        confidence_level=confidence,
        wilson_lower=lower,
        wilson_upper=upper,
        reasons=dict(sorted(reasons.items())),
        target_results=target_results,
    )


def _matches_target(issue: GatewayIssue, case: GatewayNegativeCase) -> bool:
    if issue.code != case.expected_code:
        return False
    if case.target == "publication_conflict":
        return issue.stage == "publication" and issue.layer is None
    expected_layer = int(case.target.removeprefix("layer_"))
    return issue.stage == "validation" and issue.layer == expected_layer


__all__ = [
    "GatewayEvaluationReport",
    "GatewayNegativeCase",
    "GatewayTarget",
    "GatewayTargetResult",
    "evaluate_gateway_negatives",
    "wilson_interval",
]
