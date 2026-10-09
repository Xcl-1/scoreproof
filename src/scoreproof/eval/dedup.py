"""证据查重成对评测与正式样本量门禁。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..evidence.dedup import DuplicateDecision, DuplicateThresholds, compare_evidence
from ..schema import Evidence
from .gateway import wilson_interval


class DedupPairCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    duplicate: bool
    left: Evidence
    right: Evidence
    transformation: str | None = None
    hard_negative: str | None = None


class DedupCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    expected_duplicate: bool
    predicted_duplicate: bool
    correct: bool
    transformation: str | None = None
    hard_negative: str | None = None
    decision: DuplicateDecision


class DedupEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_head: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dataset_kind: Literal["synthetic", "real_redacted"] = "synthetic"
    authorization_verified: bool = False
    redaction_verified: bool = False
    unique_pair_hashes: int = Field(default=0, ge=0)
    real_cli_entry: bool = False
    dataset_version: str = Field(pattern=r"^[A-Za-z0-9._-]{1,80}$")
    config_hash: str
    sample_size: int
    positive_pairs: int
    negative_pairs: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    recall: float
    recall_ci95: tuple[float, float]
    precision: float
    precision_ci95: tuple[float, float]
    f1: float
    thresholds: DuplicateThresholds
    independent_real_pairs: bool
    smoke_test_only: bool
    formal_gate_eligible: bool
    target_recall: float = 0.96
    target_precision: float = 0.95
    target_passed: bool
    results: list[DedupCaseResult]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _formal_provenance_and_counts(self) -> Self:
        if self.sample_size != len(self.results):
            raise ValueError("查重报告样本量与逐对结果不一致")
        if self.positive_pairs != self.true_positive + self.false_negative:
            raise ValueError("查重正例计数与混淆矩阵不一致")
        if self.negative_pairs != self.true_negative + self.false_positive:
            raise ValueError("查重负例计数与混淆矩阵不一致")
        if self.sample_size != self.positive_pairs + self.negative_pairs:
            raise ValueError("查重样本量与正负例计数不一致")
        if self.formal_gate_eligible and (
            self.smoke_test_only
            or self.sample_size < 50
            or not self.independent_real_pairs
            or self.dataset_kind != "real_redacted"
            or not self.authorization_verified
            or not self.redaction_verified
            or self.unique_pair_hashes != self.sample_size
            or not self.real_cli_entry
            or self.source_sha256 is None
            or self.git_head is None
        ):
            raise ValueError("查重正式报告缺少独立真实脱敏成对数据或可追溯主链路")
        return self


def load_dedup_dataset(path: str | Path) -> tuple[str, bool, list[DedupPairCase]]:
    """读取 JSON 数据集；顶层必须声明版本、真实独立性和 pairs。"""
    candidate = Path(path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("查重评测集顶层必须是 JSON 对象")
    version = payload.get("dataset_version")
    independent = payload.get("independent_real_pairs", False)
    rows = payload.get("pairs")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("查重评测集缺少 dataset_version")
    if not isinstance(independent, bool):
        raise ValueError("independent_real_pairs 必须是布尔值")
    if not isinstance(rows, list):
        raise ValueError("查重评测集 pairs 必须是数组")
    return version, independent, [DedupPairCase.model_validate(row) for row in rows]


def evaluate_dedup_pairs(
    cases: list[DedupPairCase],
    *,
    dataset_version: str,
    independent_real_pairs: bool,
    thresholds: DuplicateThresholds | None = None,
    formal_data_verified: bool = False,
    source_sha256: str | None = None,
    git_head: str | None = None,
    unique_pair_hashes: int = 0,
    real_cli_entry: bool = False,
) -> DedupEvaluationReport:
    limits = thresholds or DuplicateThresholds()
    results: list[DedupCaseResult] = []
    tp = fp = tn = fn = 0
    safe_transformations = {"exact", "compressed", "rotated", "cropped", "screenshot", "same_fact"}
    safe_hard_negatives = {
        "same_event_different_person",
        "same_person_different_year",
        "same_level_different_event",
        "other",
    }
    for index, case in enumerate(cases, 1):
        decision = compare_evidence(case.left, case.right, thresholds=limits)
        predicted = decision.flagged
        decision.left_id = f"left-{index}"
        decision.right_id = f"right-{index}"
        # 指标只需相似度和冲突布尔值，正式报告不应保存姓名等原始字段。
        decision.field_similarities = [
            item.model_copy(
                update={
                    "left_value": "[REDACTED]",
                    "right_value": "[REDACTED]",
                    "left_normalized": "[REDACTED]",
                    "right_normalized": "[REDACTED]",
                }
            )
            for item in decision.field_similarities
        ]
        if case.duplicate and predicted:
            tp += 1
        elif case.duplicate:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
        results.append(
            DedupCaseResult(
                id=f"pair-{index}",
                expected_duplicate=case.duplicate,
                predicted_duplicate=predicted,
                correct=case.duplicate == predicted,
                transformation=(case.transformation if case.transformation in safe_transformations else None),
                hard_negative=(case.hard_negative if case.hard_negative in safe_hard_negatives else None),
                decision=decision,
            )
        )

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    has_both_classes = (tp + fn) > 0 and (tn + fp) > 0
    eligible = (
        len(cases) >= 50
        and independent_real_pairs
        and has_both_classes
        and formal_data_verified
        and unique_pair_hashes == len(cases)
        and real_cli_entry
    )
    config_json = json.dumps(limits.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    notes = [
        "预测为‘确定重复’或‘疑似重复’均计入查重召回；系统只拦截/复核，不自动删除。",
    ]
    if len(cases) < 50:
        notes.append(f"样本量 n={len(cases)} < 50，仅可作为烟雾测试。")
    if not independent_real_pairs:
        notes.append("数据集未声明为独立真实脱敏成对样本，不具备正式验收资格。")
    if not formal_data_verified:
        notes.append("未通过真实文件、授权、脱敏与成对去重预检，不具备正式资格。")
    if not has_both_classes:
        notes.append("数据集必须同时包含重复正例与困难负例。")
    return DedupEvaluationReport(
        git_head=git_head,
        source_sha256=source_sha256,
        dataset_kind="real_redacted" if formal_data_verified else "synthetic",
        authorization_verified=formal_data_verified,
        redaction_verified=formal_data_verified,
        unique_pair_hashes=unique_pair_hashes,
        real_cli_entry=real_cli_entry,
        dataset_version=dataset_version,
        config_hash=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        sample_size=len(cases),
        positive_pairs=tp + fn,
        negative_pairs=tn + fp,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        recall=round(recall, 4),
        recall_ci95=wilson_interval(tp, tp + fn) if tp + fn else (0.0, 0.0),
        precision=round(precision, 4),
        precision_ci95=wilson_interval(tp, tp + fp) if tp + fp else (0.0, 0.0),
        f1=round(f1, 4),
        thresholds=limits,
        independent_real_pairs=independent_real_pairs,
        smoke_test_only=not eligible,
        formal_gate_eligible=eligible,
        target_passed=eligible and recall >= 0.96 and precision >= 0.95,
        results=results,
        notes=notes,
    )


__all__ = [
    "DedupCaseResult",
    "DedupEvaluationReport",
    "DedupPairCase",
    "evaluate_dedup_pairs",
    "load_dedup_dataset",
]
