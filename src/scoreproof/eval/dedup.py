"""证据查重成对评测与正式样本量门禁。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
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
) -> DedupEvaluationReport:
    limits = thresholds or DuplicateThresholds()
    results: list[DedupCaseResult] = []
    tp = fp = tn = fn = 0
    for case in cases:
        decision = compare_evidence(case.left, case.right, thresholds=limits)
        predicted = decision.flagged
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
                id=case.id,
                expected_duplicate=case.duplicate,
                predicted_duplicate=predicted,
                correct=case.duplicate == predicted,
                transformation=case.transformation,
                hard_negative=case.hard_negative,
                decision=decision,
            )
        )

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    has_both_classes = (tp + fn) > 0 and (tn + fp) > 0
    eligible = len(cases) >= 50 and independent_real_pairs and has_both_classes
    config_json = json.dumps(limits.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    notes = [
        "预测为‘确定重复’或‘疑似重复’均计入查重召回；系统只拦截/复核，不自动删除。",
    ]
    if len(cases) < 50:
        notes.append(f"样本量 n={len(cases)} < 50，仅可作为烟雾测试。")
    if not independent_real_pairs:
        notes.append("数据集未声明为独立真实脱敏成对样本，不具备正式验收资格。")
    if not has_both_classes:
        notes.append("数据集必须同时包含重复正例与困难负例。")
    return DedupEvaluationReport(
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
        recall_ci95=wilson_interval(tp, tp + fn),
        precision=round(precision, 4),
        precision_ci95=wilson_interval(tp, tp + fp),
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
