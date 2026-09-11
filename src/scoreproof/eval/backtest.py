"""评测层：用往年综测表做回测，输出可写进简历的真实数字。

最大优势（项目总结第 7 节）：**往年综测表本身就是天然 ground truth**。
本模块只负责"逐人逐项比对 + 汇总指标"，不做任何取巧。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..calc.engine import EngineConfig, compute_all
from ..errors import DataSourceError
from ..schema import Claim, Ruleset, ScoreBreakdown


def _round(x: float, digits: int = 2) -> float:
    return round(float(x), digits)


def _num(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and v != v):  # None / NaN -> 视为缺失
        return None
    try:
        if isinstance(v, str):
            v = v.strip().replace("分", "")
            if not v:
                return None
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class ItemDiff:
    """单条差异：用于定位"系统算错在哪"。"""

    student_id: str
    claim_id: str | None
    level: str | None
    expected: float | None
    actual: float
    kind: str  # missing / extra / value_mismatch
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "claim_id": self.claim_id,
            "level": self.level,
            "expected": self.expected,
            "actual": self.actual,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass
class StudentResult:
    student_id: str
    expected: float | None
    actual: float
    delta: float | None
    matched: bool
    review_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "expected": self.expected,
            "actual": self.actual,
            "delta": self.delta,
            "matched": self.matched,
        }


@dataclass
class BacktestReport:
    """回测报告：准确率、误差分布、可解释差异清单。"""

    total_students: int = 0
    matched_students: int = 0
    exact: int = 0
    within_tolerance: int = 0
    tolerance: float = 0.01
    results: list[StudentResult] = field(default_factory=list)
    diffs: list[ItemDiff] = field(default_factory=list)
    unmatched_claims: int = 0
    review_claims: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """完全一致比例（简历口径：逐人比对，含小数一致）。"""
        return _round(self.exact / self.total_students, 4) if self.total_students else 0.0

    @property
    def accuracy_within_tolerance(self) -> float:
        return (
            _round(self.within_tolerance / self.total_students, 4) if self.total_students else 0.0
        )

    def summary(self) -> dict:
        return {
            "total_students": self.total_students,
            "exact": self.exact,
            "accuracy": self.accuracy,
            "within_tolerance": self.within_tolerance,
            "accuracy_within_tolerance": self.accuracy_within_tolerance,
            "tolerance": self.tolerance,
            "unmatched_claims": self.unmatched_claims,
            "review_claims": self.review_claims,
            "example_diffs": [d.to_dict() for d in self.diffs[:10]],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])


def compare_students(
    predicted: dict[str, ScoreBreakdown] | dict[str, float],
    expected: dict[str, float],
    *,
    tolerance: float = 0.01,
    diffs: Sequence[ItemDiff] = (),
) -> BacktestReport:
    """逐人比对（纯函数，便于测试）。"""
    report = BacktestReport(tolerance=tolerance, diffs=list(diffs))
    for sid, exp in expected.items():
        report.total_students += 1
        pred = predicted.get(sid)
        actual = pred.total if isinstance(pred, ScoreBreakdown) else (pred or 0.0)
        delta = _round(actual - exp, 4)
        ok = abs(delta) <= tolerance
        if ok:
            report.exact += 1
            report.within_tolerance += 1
        elif abs(delta) <= max(tolerance, 0.5):
            report.within_tolerance += 1
        report.results.append(
            StudentResult(
                student_id=sid,
                expected=_round(exp, 4),
                actual=_round(actual, 4),
                delta=delta,
                matched=ok,
                review_claims=list(pred.review_claims) if isinstance(pred, ScoreBreakdown) else [],
            )
        )
        if isinstance(pred, ScoreBreakdown):
            report.unmatched_claims += len(pred.unmatched_claims)
            report.review_claims += len(pred.review_claims)
    report.meta["expected_students"] = len(expected)
    report.meta["predicted_students"] = len(predicted)
    missing = set(expected) - set(predicted)
    if missing:
        report.meta["missing_students"] = sorted(missing)[:20]
    return report


def run_backtest(
    claims: Iterable[Claim],
    ruleset: Ruleset,
    ground_truth: dict[str, float],
    *,
    academic_year: str | None = None,
    college: str | None = None,
    config: EngineConfig | None = None,
    tolerance: float = 0.01,
) -> BacktestReport:
    """完整回测：核算全部学生 -> 与往年手算结果比对。"""
    claims = list(claims)
    predicted = compute_all(
        claims, ruleset, academic_year=academic_year, college=college, config=config
    )
    report = compare_students(predicted, ground_truth, tolerance=tolerance)
    report.meta["claims"] = len(claims)
    report.meta["rules"] = len(ruleset)
    report.diffs = _explain_diffs(predicted, ground_truth, tolerance=tolerance)
    return report


def _explain_diffs(
    predicted: dict[str, ScoreBreakdown],
    expected: dict[str, float],
    *,
    tolerance: float,
) -> list[ItemDiff]:
    """给出差异的初步归因（未命中规则 / 需复核 / 分值不符）。"""
    out: list[ItemDiff] = []
    for sid, exp in expected.items():
        pred = predicted.get(sid)
        if pred is None:
            out.append(ItemDiff(sid, None, None, _round(exp, 4), 0.0, "missing",
                                "该学号没有任何申报条目"))
            continue
        delta = pred.total - exp
        if abs(delta) <= tolerance:
            continue
        if pred.unmatched_claims:
            out.append(ItemDiff(sid, None, None, _round(exp, 4), pred.total, "missing",
                                f"{len(pred.unmatched_claims)} 条申报未命中规则"))
        elif pred.review_claims:
            out.append(ItemDiff(sid, None, None, _round(exp, 4), pred.total, "value_mismatch",
                                f"{len(pred.review_claims)} 条申报需人工复核"))
        else:
            out.append(ItemDiff(sid, None, None, _round(exp, 4), pred.total, "value_mismatch",
                                "分值不符，检查规则分值或互斥/封顶语义"))
    return out


def load_ground_truth(
    path: str | Path,
    *,
    sheet: str | int = 0,
    student_col: str = "学号",
    total_col: str = "总分",
) -> dict[str, float]:
    """从往年综测表读取 ground truth：``学号 -> 总分``。"""
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"评测文件不存在：{p}", detail={"path": str(p)})
    df = pd.read_excel(p, sheet_name=sheet)
    cols = {str(c).strip(): c for c in df.columns}
    if student_col not in cols or total_col not in cols:
        raise DataSourceError(
            f"评测表缺少列：{student_col} / {total_col}",
            detail={"found_columns": list(df.columns)},
        )
    truth: dict[str, float] = {}
    for _, row in df.iterrows():
        sid, total = row[cols[student_col]], _num(row[cols[total_col]])
        if sid is None or total is None:
            continue
        truth[str(sid).strip()] = total
    return truth


__all__ = [
    "BacktestReport",
    "ItemDiff",
    "StudentResult",
    "compare_students",
    "load_ground_truth",
    "run_backtest",
]
