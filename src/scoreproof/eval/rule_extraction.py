"""规则结构化抽取评测：以人工金标检查通过网关后的完整规则字段。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..normalize import normalize_level
from ..rules.gateway import ExtractionGateway, GatewayContext, GatewayReport, RuleDraftInput
from ..schema import Rule
from .gateway import wilson_interval

RULE_EXTRACTION_FIELDS: tuple[str, ...] = (
    "category",
    "level",
    "score",
    "synonyms",
    "cap",
    "team_factor",
    "clause",
    "evidence_quote",
    "rank",
    "item_name",
    "effective_date",
)


class ValidatedRuleExtractor(Protocol):
    model: str

    def extract_validated(
        self,
        text: str,
        *,
        context: GatewayContext,
        risk_level: Literal["normal", "high"] = "normal",
        existing_rules: Iterable[Rule] = (),
        **kwargs: Any,
    ) -> GatewayReport: ...


class RuleExtractionCase(BaseModel):
    """一个可独立复核的原文块及其人工金标。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1, description="脱敏文档/逻辑块标识，不得放原始路径")
    source_text: str = Field(min_length=1, max_length=50_000)
    academic_year: str
    college: str | None = None
    page: int | None = Field(default=None, ge=1)
    table: str | None = None
    category_hint: str | None = None
    allowed_levels: list[str] = Field(default_factory=list)
    risk_level: Literal["normal", "high"] = "normal"
    real_source: bool
    synthetic: bool = False
    expected_rules: list[RuleDraftInput] = Field(default_factory=list)

    @field_validator("case_id", "source_id", "source_text")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不得为空白")
        return value

    @model_validator(mode="after")
    def _gold_must_pass_the_same_gateway(self) -> Self:
        signatures = [
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for item in self.expected_rules
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("同一原文块中的人工金标规则不得重复")
        context = self.gateway_context()
        report = ExtractionGateway().validate_batch(
            [item.model_dump(mode="python") for item in self.expected_rules],
            source_text=self.source_text,
            context=context,
        )
        if any(not item.publishable for item in report.items):
            codes = sorted({issue.code for item in report.items for issue in item.issues})
            raise ValueError(f"人工金标未通过同一代码级网关：{codes}")
        return self

    def gateway_context(self) -> GatewayContext:
        return GatewayContext(
            academic_year=self.academic_year,
            college=self.college,
            doc=self.source_id,
            page=self.page,
            table=self.table,
            allowed_levels=frozenset(self.allowed_levels),
        )


class RuleExtractionDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_version: str = Field(min_length=1)
    authorization_reference: str | None = None
    independent_real_samples: bool = False
    cases: list[RuleExtractionCase] = Field(default_factory=list, max_length=200)

    @field_validator("authorization_reference")
    @classmethod
    def _strip_authorization(cls, value: str | None) -> str | None:
        stripped = value.strip() if value is not None else None
        return stripped or None

    @model_validator(mode="after")
    def _case_ids_and_sources_are_unique(self) -> Self:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id 不得重复")
        source_ids = [case.source_id for case in self.cases]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id 不得重复")
        source_hashes = [
            hashlib.sha256(case.source_text.encode("utf-8")).hexdigest()
            for case in self.cases
        ]
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError("相同原文块不得通过更换 source_id 重复计数")
        return self


class AccuracyMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    correct: int = Field(ge=0)
    total: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0, le=1)
    ci95: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.correct > self.total:
            raise ValueError("correct 不得超过 total")
        if self.total == 0 and (self.value is not None or self.ci95 is not None):
            raise ValueError("空指标不得填写 value 或 ci95")
        if self.total > 0 and (self.value is None or self.ci95 is None):
            raise ValueError("非空指标必须填写 value 与 ci95")
        if self.total > 0 and self.value != round(self.correct / self.total, 4):
            raise ValueError("value 必须由 correct / total 计算")
        return self


class FieldPRFMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)
    precision_ci95: tuple[float, float] | None = None
    recall_ci95: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _values_are_consistent(self) -> Self:
        precision_total = self.true_positive + self.false_positive
        recall_total = self.true_positive + self.false_negative
        expected_precision = (
            round(self.true_positive / precision_total, 4) if precision_total else None
        )
        expected_recall = round(self.true_positive / recall_total, 4) if recall_total else None
        expected_f1 = None
        if expected_precision is not None and expected_recall is not None:
            expected_f1 = (
                round(2 * expected_precision * expected_recall / (expected_precision + expected_recall), 4)
                if expected_precision + expected_recall
                else 0.0
            )
        if (self.precision, self.recall, self.f1) != (
            expected_precision,
            expected_recall,
            expected_f1,
        ):
            raise ValueError("precision / recall / f1 必须由混淆计数计算")
        if (self.precision_ci95 is None) != (precision_total == 0):
            raise ValueError("precision_ci95 与 precision 分母不一致")
        if (self.recall_ci95 is None) != (recall_total == 0):
            raise ValueError("recall_ci95 与 recall 分母不一致")
        return self


class RuleExtractionCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_rules: int = Field(ge=0)
    predicted_rules: int = Field(ge=0)
    scored_rule_units: int = Field(ge=0)
    exact_rules_correct: int = Field(ge=0)
    field_correct: dict[str, int]
    field_total: dict[str, int]
    field_true_positive: dict[str, int]
    field_false_positive: dict[str, int]
    field_false_negative: dict[str, int]
    complete_correct: bool
    gateway_rejected: int = Field(ge=0)
    secondary_used: bool
    extraction_error: str | None = None

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> Self:
        if self.scored_rule_units != max(self.expected_rules, self.predicted_rules):
            raise ValueError("scored_rule_units 必须覆盖漏抽与多抽")
        if self.exact_rules_correct > min(self.expected_rules, self.predicted_rules):
            raise ValueError("exact_rules_correct 超出可匹配规则数")
        if set(self.field_correct) != set(RULE_EXTRACTION_FIELDS) or set(
            self.field_total
        ) != set(RULE_EXTRACTION_FIELDS):
            raise ValueError("逐例字段计数必须完整覆盖规则字段")
        for counts in (
            self.field_true_positive,
            self.field_false_positive,
            self.field_false_negative,
        ):
            if set(counts) != set(RULE_EXTRACTION_FIELDS) or any(value < 0 for value in counts.values()):
                raise ValueError("逐例字段 P/R/F1 计数必须完整且非负")
        if any(
            self.field_correct[field] > self.field_total[field]
            or self.field_total[field] != self.scored_rule_units
            for field in RULE_EXTRACTION_FIELDS
        ):
            raise ValueError("逐字段计数不一致")
        return self


class RuleExtractionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    dataset_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str
    model: str
    sample_size: int = Field(ge=0, description="人工金标规则数")
    scored_rule_units: int = Field(ge=0, description="含额外模型输出的惩罚后计分单元")
    case_count: int = Field(ge=0)
    real_sources: bool
    authorization_verified: bool
    independent_real_samples: bool
    real_external_service: bool
    exact_rule_accuracy: AccuracyMetric
    field_micro_accuracy: AccuracyMetric
    field_metrics: dict[str, AccuracyMetric]
    field_micro_prf: FieldPRFMetric
    field_prf: dict[str, FieldPRFMetric]
    complete_case_accuracy: AccuracyMetric
    formal_gate_eligible: bool
    passed: bool
    smoke_test_only: bool
    results: list[RuleExtractionCaseResult]
    limitations: list[str]

    @model_validator(mode="after")
    def _flags_and_counts_are_consistent(self) -> Self:
        if self.smoke_test_only == self.formal_gate_eligible:
            raise ValueError("smoke_test_only 必须与 formal_gate_eligible 相反")
        if self.passed and not self.formal_gate_eligible:
            raise ValueError("非正式报告不得标记 passed")
        if self.case_count != len(self.results):
            raise ValueError("case_count 与结果数量不一致")
        if set(self.field_metrics) != set(RULE_EXTRACTION_FIELDS):
            raise ValueError("field_metrics 必须完整覆盖规则字段")
        if set(self.field_prf) != set(RULE_EXTRACTION_FIELDS):
            raise ValueError("field_prf 必须完整覆盖规则字段")
        expected_rules = sum(item.expected_rules for item in self.results)
        scored_units = sum(item.scored_rule_units for item in self.results)
        exact_correct = sum(item.exact_rules_correct for item in self.results)
        if self.sample_size != expected_rules or self.scored_rule_units != scored_units:
            raise ValueError("报告样本量与逐例结果不一致")
        if (
            self.exact_rule_accuracy.correct != exact_correct
            or self.exact_rule_accuracy.total != scored_units
        ):
            raise ValueError("完整规则准确率计数与逐例结果不一致")
        for field in RULE_EXTRACTION_FIELDS:
            if self.field_metrics[field].correct != sum(
                item.field_correct[field] for item in self.results
            ) or self.field_metrics[field].total != sum(
                item.field_total[field] for item in self.results
            ):
                raise ValueError(f"字段 {field} 指标与逐例结果不一致")
        if self.field_micro_accuracy.correct != sum(
            metric.correct for metric in self.field_metrics.values()
        ) or self.field_micro_accuracy.total != sum(
            metric.total for metric in self.field_metrics.values()
        ):
            raise ValueError("字段 micro 指标与各字段指标不一致")
        for field in RULE_EXTRACTION_FIELDS:
            expected_counts = (
                sum(item.field_true_positive[field] for item in self.results),
                sum(item.field_false_positive[field] for item in self.results),
                sum(item.field_false_negative[field] for item in self.results),
            )
            metric = self.field_prf[field]
            if expected_counts != (
                metric.true_positive,
                metric.false_positive,
                metric.false_negative,
            ):
                raise ValueError(f"字段 {field} P/R/F1 与逐例结果不一致")
        if (
            self.field_micro_prf.true_positive,
            self.field_micro_prf.false_positive,
            self.field_micro_prf.false_negative,
        ) != (
            sum(metric.true_positive for metric in self.field_prf.values()),
            sum(metric.false_positive for metric in self.field_prf.values()),
            sum(metric.false_negative for metric in self.field_prf.values()),
        ):
            raise ValueError("字段 micro P/R/F1 与各字段计数不一致")
        if (
            self.complete_case_accuracy.correct != sum(item.complete_correct for item in self.results)
            or self.complete_case_accuracy.total != self.case_count
        ):
            raise ValueError("整块完全正确率与逐例结果不一致")
        expected_formal = bool(
            self.sample_size >= 50
            and self.real_sources
            and self.authorization_verified
            and self.independent_real_samples
            and self.real_external_service
            and all(item.extraction_error is None for item in self.results)
        )
        if self.formal_gate_eligible != expected_formal:
            raise ValueError("formal_gate_eligible 与正式样本条件不一致")
        expected_passed = bool(
            expected_formal
            and self.exact_rule_accuracy.value is not None
            and self.exact_rule_accuracy.value >= 0.96
        )
        if self.passed != expected_passed:
            raise ValueError("passed 与完整规则正确率门槛不一致")
        return self


def load_rule_extraction_dataset(path: str | Path) -> RuleExtractionDataset:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取规则抽取评测集：{exc}") from exc
    return RuleExtractionDataset.model_validate(payload)


def _compact_text(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).strip()


def _normalized_rule(rule: RuleDraftInput) -> dict[str, Any]:
    return {
        "category": _compact_text(rule.category),
        "level": normalize_level(rule.level).canonical,
        "score": float(rule.score),
        "synonyms": tuple(sorted(filter(None, (_compact_text(item) for item in rule.synonyms)))),
        "cap": float(rule.cap) if rule.cap is not None else None,
        "team_factor": float(rule.team_factor) if rule.team_factor is not None else None,
        "clause": _compact_text(rule.clause),
        "evidence_quote": _compact_text(rule.evidence_quote),
        "rank": _compact_text(rule.rank),
        "item_name": _compact_text(rule.item_name),
        "effective_date": _compact_text(rule.effective_date),
    }


def _field_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return sum(left[field] == right[field] for field in RULE_EXTRACTION_FIELDS)


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != ()


def _score_case(
    case: RuleExtractionCase,
    report: GatewayReport | None,
    *,
    extraction_error: str | None,
) -> RuleExtractionCaseResult:
    expected = [_normalized_rule(item) for item in case.expected_rules]
    accepted = (
        [_normalized_rule(item.draft) for item in report.items if item.publishable and item.draft]
        if report is not None
        else []
    )
    unmatched_expected = set(range(len(expected)))
    unmatched_predicted = set(range(len(accepted)))
    pairs: list[tuple[int, int]] = []

    # 先固定完全正确项，再对剩余项按字段相似度做稳定的一对一匹配。
    for expected_index, gold in enumerate(expected):
        for predicted_index in sorted(unmatched_predicted):
            if gold == accepted[predicted_index]:
                pairs.append((expected_index, predicted_index))
                unmatched_expected.discard(expected_index)
                unmatched_predicted.discard(predicted_index)
                break
    candidates = sorted(
        (
            _field_score(expected[e], accepted[p]),
            e,
            p,
        )
        for e in unmatched_expected
        for p in unmatched_predicted
    )
    for score, expected_index, predicted_index in reversed(candidates):
        if score <= 0:
            break
        if expected_index in unmatched_expected and predicted_index in unmatched_predicted:
            pairs.append((expected_index, predicted_index))
            unmatched_expected.remove(expected_index)
            unmatched_predicted.remove(predicted_index)

    field_correct = {field: 0 for field in RULE_EXTRACTION_FIELDS}
    field_tp = {field: 0 for field in RULE_EXTRACTION_FIELDS}
    field_fp = {field: 0 for field in RULE_EXTRACTION_FIELDS}
    field_fn = {field: 0 for field in RULE_EXTRACTION_FIELDS}
    for expected_index, predicted_index in pairs:
        for field in RULE_EXTRACTION_FIELDS:
            gold = expected[expected_index][field]
            predicted = accepted[predicted_index][field]
            field_correct[field] += int(gold == predicted)
            if gold == predicted and _has_value(gold):
                field_tp[field] += 1
            elif gold != predicted:
                field_fn[field] += int(_has_value(gold))
                field_fp[field] += int(_has_value(predicted))
    for expected_index in unmatched_expected:
        for field in RULE_EXTRACTION_FIELDS:
            field_fn[field] += int(_has_value(expected[expected_index][field]))
    for predicted_index in unmatched_predicted:
        for field in RULE_EXTRACTION_FIELDS:
            field_fp[field] += int(_has_value(accepted[predicted_index][field]))
    scored_units = max(len(expected), len(accepted))
    exact_correct = sum(expected[e] == accepted[p] for e, p in pairs)
    return RuleExtractionCaseResult(
        case_id=case.case_id,
        source_id=case.source_id,
        source_sha256=hashlib.sha256(case.source_text.encode("utf-8")).hexdigest(),
        expected_rules=len(expected),
        predicted_rules=len(accepted),
        scored_rule_units=scored_units,
        exact_rules_correct=exact_correct,
        field_correct=field_correct,
        field_total={field: scored_units for field in RULE_EXTRACTION_FIELDS},
        field_true_positive=field_tp,
        field_false_positive=field_fp,
        field_false_negative=field_fn,
        complete_correct=(
            extraction_error is None
            and len(expected) == len(accepted)
            and exact_correct == len(expected)
        ),
        gateway_rejected=(
            sum(not item.publishable for item in report.items) if report is not None else 0
        ),
        secondary_used=bool(report and report.secondary_used),
        extraction_error=extraction_error,
    )


def _metric(correct: int, total: int) -> AccuracyMetric:
    return AccuracyMetric(
        correct=correct,
        total=total,
        value=round(correct / total, 4) if total else None,
        ci95=wilson_interval(correct, total) if total else None,
    )


def _prf_metric(true_positive: int, false_positive: int, false_negative: int) -> FieldPRFMetric:
    precision_total = true_positive + false_positive
    recall_total = true_positive + false_negative
    precision = round(true_positive / precision_total, 4) if precision_total else None
    recall = round(true_positive / recall_total, 4) if recall_total else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision + recall
            else 0.0
        )
    return FieldPRFMetric(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        precision_ci95=(
            wilson_interval(true_positive, precision_total) if precision_total else None
        ),
        recall_ci95=wilson_interval(true_positive, recall_total) if recall_total else None,
    )


def evaluate_rule_extraction(
    dataset: RuleExtractionDataset,
    *,
    extractor: ValidatedRuleExtractor,
    provider: str,
    real_external_service: bool,
    force_double_check: bool = False,
) -> RuleExtractionReport:
    """真实调用抽取器并评测；报告不保存原文、金标或模型原始输出。"""

    results: list[RuleExtractionCaseResult] = []
    for case in dataset.cases:
        report: GatewayReport | None = None
        error: str | None = None
        try:
            report = extractor.extract_validated(
                case.source_text,
                context=case.gateway_context(),
                risk_level="high" if force_double_check else case.risk_level,
                existing_rules=(),
                **({"category": case.category_hint} if case.category_hint else {}),
            )
        except Exception as exc:  # 错误必须进入报告，不能让整批静默消失
            error = f"{type(exc).__name__}: {exc}"
        results.append(_score_case(case, report, extraction_error=error))

    sample_size = sum(item.expected_rules for item in results)
    scored_units = sum(item.scored_rule_units for item in results)
    exact_correct = sum(item.exact_rules_correct for item in results)
    field_correct = {
        field: sum(item.field_correct[field] for item in results)
        for field in RULE_EXTRACTION_FIELDS
    }
    field_total = {
        field: sum(item.field_total[field] for item in results)
        for field in RULE_EXTRACTION_FIELDS
    }
    field_tp = {
        field: sum(item.field_true_positive[field] for item in results)
        for field in RULE_EXTRACTION_FIELDS
    }
    field_fp = {
        field: sum(item.field_false_positive[field] for item in results)
        for field in RULE_EXTRACTION_FIELDS
    }
    field_fn = {
        field: sum(item.field_false_negative[field] for item in results)
        for field in RULE_EXTRACTION_FIELDS
    }
    all_external_calls_succeeded = bool(results) and all(
        item.extraction_error is None for item in results
    )
    real_sources = bool(dataset.cases) and all(
        case.real_source and not case.synthetic for case in dataset.cases
    )
    authorization_verified = bool(dataset.authorization_reference)
    formal = bool(
        sample_size >= 50
        and real_sources
        and authorization_verified
        and dataset.independent_real_samples
        and real_external_service
        and all_external_calls_succeeded
    )
    exact_metric = _metric(exact_correct, scored_units)
    micro_correct = sum(field_correct.values())
    micro_total = sum(field_total.values())
    micro_metric = _metric(micro_correct, micro_total)
    complete_cases = sum(item.complete_correct for item in results)
    passed = bool(
        formal
        and exact_metric.value is not None
        and exact_metric.value >= 0.96
    )
    canonical = {
        "dataset": dataset.model_dump(mode="json"),
        "provider": provider,
        "model": extractor.model,
    }
    limitations: list[str] = []
    if sample_size < 50:
        limitations.append(f"人工金标规则只有 {sample_size} 条；正式验收要求 n≥50。")
    if not real_sources:
        limitations.append("包含合成或未声明为真实的原文，只能作为烟雾测试。")
    if not authorization_verified:
        limitations.append("缺少公开来源或数据授权引用。")
    if not dataset.independent_real_samples:
        limitations.append("未声明为独立真实评测样本。")
    if not real_external_service:
        limitations.append("未通过真实外部文本模型执行评测。")
    if not all_external_calls_succeeded:
        limitations.append("至少一个抽取调用失败，错误已保留在逐例结果中。")
    return RuleExtractionReport(
        dataset_version=dataset.dataset_version,
        source_sha256=hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        provider=provider,
        model=extractor.model,
        sample_size=sample_size,
        scored_rule_units=scored_units,
        case_count=len(results),
        real_sources=real_sources,
        authorization_verified=authorization_verified,
        independent_real_samples=dataset.independent_real_samples,
        real_external_service=real_external_service and all_external_calls_succeeded,
        exact_rule_accuracy=exact_metric,
        field_micro_accuracy=micro_metric,
        field_metrics={
            field: _metric(field_correct[field], field_total[field])
            for field in RULE_EXTRACTION_FIELDS
        },
        field_micro_prf=_prf_metric(
            sum(field_tp.values()),
            sum(field_fp.values()),
            sum(field_fn.values()),
        ),
        field_prf={
            field: _prf_metric(field_tp[field], field_fp[field], field_fn[field])
            for field in RULE_EXTRACTION_FIELDS
        },
        complete_case_accuracy=_metric(complete_cases, len(results)),
        formal_gate_eligible=formal,
        passed=passed,
        smoke_test_only=not formal,
        results=results,
        limitations=limitations,
    )


__all__ = [
    "AccuracyMetric",
    "FieldPRFMetric",
    "RULE_EXTRACTION_FIELDS",
    "RuleExtractionCase",
    "RuleExtractionCaseResult",
    "RuleExtractionDataset",
    "RuleExtractionReport",
    "ValidatedRuleExtractor",
    "evaluate_rule_extraction",
    "load_rule_extraction_dataset",
]
