"""52 人历史回测：逐人总分、逐项得分、差异归因与数据完整性门禁。

历史人工结果不天然等于真值。只有经过业务确认并冻结的数据，才允许使用
``adjudicated_truth``（准确率）口径；否则必须使用 ``historical_reference``
（一致率/差异分析）口径。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..calc.engine import EngineConfig, compute_all
from ..errors import DataSourceError
from ..schema import Claim, RuleMatch, Ruleset, ScoreBreakdown

BacktestMode = Literal["historical_reference", "adjudicated_truth"]


def _round(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def _num(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        if isinstance(v, str):
            v = v.strip().replace("分", "")
            if not v:
                return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> bool:
    if v is None or (isinstance(v, float) and v != v):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "是", "已裁决", "√"}


def claim_item_key(claim: Claim) -> str:
    """返回跨多次导入稳定的申报项标识，优先使用表内显式标识。"""
    explicit = str(claim.extra.get("item_key") or "").strip()
    if explicit:
        return explicit
    if claim.source_ref and claim.source_ref.row is not None:
        return f"row:{claim.source_ref.row}"
    return claim.id


@dataclass(frozen=True)
class ItemExpectation:
    """一条历史参照或已裁决真值。"""

    student_id: str
    item_key: str
    expected: float
    adjudicated: bool = False
    attribution: str | None = None
    suspected_manual_error: bool = False
    note: str | None = None


@dataclass
class ItemDiff:
    """单条差异，保留系统值、参照值、规则状态与双向归因。"""

    student_id: str
    claim_id: str | None
    level: str | None
    expected: float | None
    actual: float
    kind: str
    reason: str | None = None
    item_key: str | None = None
    category: str | None = None
    raw_text: str | None = None
    attribution: str | None = None
    suspected_manual_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "item_key": self.item_key,
            "claim_id": self.claim_id,
            "category": self.category,
            "raw_text": self.raw_text,
            "level": self.level,
            "expected": self.expected,
            "actual": self.actual,
            "kind": self.kind,
            "reason": self.reason,
            "attribution": self.attribution,
            "suspected_manual_error": self.suspected_manual_error,
        }


@dataclass
class StudentResult:
    student_id: str
    expected: float | None
    actual: float
    delta: float | None
    matched: bool
    review_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "expected": self.expected,
            "actual": self.actual,
            "delta": self.delta,
            "matched": self.matched,
            "review_claims": self.review_claims,
        }


@dataclass
class BacktestReport:
    """可复跑回测报告；完整差异不会被摘要截断。"""

    mode: BacktestMode = "historical_reference"
    total_students: int = 0
    matched_students: int = 0
    exact: int = 0
    within_tolerance: int = 0
    tolerance: float = 0.01
    results: list[StudentResult] = field(default_factory=list)
    diffs: list[ItemDiff] = field(default_factory=list)
    total_items: int = 0
    matched_items: int = 0
    unmatched_claims: int = 0
    review_claims: int = 0
    required_students: int | None = None
    data_complete: bool = False
    gate_passed: bool | None = None
    data_issues: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """兼容旧调用；历史参照模式对外必须称为逐人一致率。"""
        return _round(self.exact / self.total_students) if self.total_students else 0.0

    @property
    def accuracy_within_tolerance(self) -> float:
        return _round(self.within_tolerance / self.total_students) if self.total_students else 0.0

    @property
    def item_agreement(self) -> float | None:
        return _round(self.matched_items / self.total_items) if self.total_items else None

    @property
    def mean_absolute_error(self) -> float | None:
        deltas = [abs(r.delta) for r in self.results if r.delta is not None]
        return _round(sum(deltas) / len(deltas)) if deltas else None

    @property
    def max_absolute_error(self) -> float | None:
        deltas = [abs(r.delta) for r in self.results if r.delta is not None]
        return _round(max(deltas)) if deltas else None

    @property
    def error_distribution(self) -> dict[str, int]:
        return dict(sorted(Counter(diff.kind for diff in self.diffs).items()))

    @property
    def attribution_distribution(self) -> dict[str, int]:
        labels = [diff.attribution or "未归因" for diff in self.diffs if diff.expected is not None]
        return dict(sorted(Counter(labels).items()))

    @property
    def person_metric_name(self) -> str:
        return "逐人总分准确率" if self.mode == "adjudicated_truth" else "逐人总分一致率"

    @property
    def item_metric_name(self) -> str:
        return "逐项准确率" if self.mode == "adjudicated_truth" else "逐项一致率"

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metric_semantics": "业务裁决真值" if self.mode == "adjudicated_truth" else "历史人工结果参照",
            "total_students": self.total_students,
            "matched_students": self.matched_students,
            "person_metric_name": self.person_metric_name,
            "person_exact": self.exact,
            "person_agreement": self.accuracy,
            "within_tolerance": self.within_tolerance,
            "agreement_within_tolerance": self.accuracy_within_tolerance,
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "total_items": self.total_items,
            "item_metric_name": self.item_metric_name,
            "matched_items": self.matched_items,
            "item_agreement": self.item_agreement,
            "tolerance": self.tolerance,
            "unmatched_claims": self.unmatched_claims,
            "review_claims": self.review_claims,
            "error_distribution": self.error_distribution,
            "attribution_distribution": self.attribution_distribution,
            "required_students": self.required_students,
            "data_complete": self.data_complete,
            "gate_passed": self.gate_passed,
            "data_issues": self.data_issues,
            "example_diffs": [d.to_dict() for d in self.diffs[:10]],
            "meta": self.meta,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "results": [result.to_dict() for result in self.results],
            "diffs": [diff.to_dict() for diff in self.diffs],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [r.to_dict() for r in self.results],
            columns=["student_id", "expected", "actual", "delta", "matched", "review_claims"],
        )

    def diffs_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [d.to_dict() for d in self.diffs],
            columns=[
                "student_id",
                "item_key",
                "claim_id",
                "category",
                "raw_text",
                "level",
                "expected",
                "actual",
                "kind",
                "reason",
                "attribution",
                "suspected_manual_error",
            ],
        )


def compare_students(
    predicted: dict[str, ScoreBreakdown] | dict[str, float],
    expected: dict[str, float],
    *,
    tolerance: float = 0.01,
    diffs: Sequence[ItemDiff] = (),
    mode: BacktestMode = "historical_reference",
) -> BacktestReport:
    """逐人比较总分；历史模式只表达一致关系。"""
    report = BacktestReport(mode=mode, tolerance=tolerance, diffs=list(diffs))
    for sid, exp in expected.items():
        report.total_students += 1
        pred = predicted.get(sid)
        if pred is not None:
            report.matched_students += 1
        actual = pred.total if isinstance(pred, ScoreBreakdown) else (pred or 0.0)
        delta = _round(actual - exp)
        ok = abs(delta) <= tolerance
        if ok:
            report.exact += 1
            report.within_tolerance += 1
        elif abs(delta) <= max(tolerance, 0.5):
            report.within_tolerance += 1
        report.results.append(
            StudentResult(
                student_id=sid,
                expected=_round(exp),
                actual=_round(actual),
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
    extra = set(predicted) - set(expected)
    if missing:
        report.meta["missing_students"] = sorted(missing)[:20]
    if extra:
        report.meta["extra_students"] = sorted(extra)[:20]
    return report


def _match_kind(match: RuleMatch | None, claim: Claim | None) -> tuple[str, str]:
    if claim is None:
        return "missing_claim", "历史参照中的申报项在本次明细表中不存在"
    if match is None or match.rule_id is None:
        return "unmatched_rule", "申报项未命中规则"
    if match.needs_review:
        return "manual_review", "申报项因低置信度未自动计入，需人工复核"
    if not match.counted:
        return "dedup_or_exclusive", match.reason or "申报项因去重、互斥或资格约束未计入"
    return "score_mismatch", match.reason or "系统计分与参照分值不一致"


def _contributions(predicted: dict[str, ScoreBreakdown]) -> dict[str, float]:
    """把分组封顶后的实际计入分归到获胜申报项。"""
    out: dict[str, float] = {}
    for breakdown in predicted.values():
        for match in breakdown.matches:
            out[match.claim_id] = 0.0
        for group in breakdown.groups:
            if group.winner is not None and group.winner.counted:
                out[group.winner.claim_id] = group.after_cap
    return out


def compare_items(
    claims: Sequence[Claim],
    predicted: dict[str, ScoreBreakdown],
    expectations: Sequence[ItemExpectation],
    *,
    tolerance: float,
) -> tuple[int, list[ItemDiff], list[str]]:
    """逐项比较并返回一致数、完整差异、数据问题。"""
    claim_by_key: dict[tuple[str, str], Claim] = {}
    issues: list[str] = []
    for source_claim in claims:
        key = (source_claim.student_id, claim_item_key(source_claim))
        if key in claim_by_key:
            issues.append(f"申报项标识重复：{key[0]} / {key[1]}")
        claim_by_key[key] = source_claim

    expectation_by_key: dict[tuple[str, str], ItemExpectation] = {}
    for item in expectations:
        key = (item.student_id, item.item_key)
        if key in expectation_by_key:
            issues.append(f"逐项参照标识重复：{key[0]} / {key[1]}")
        expectation_by_key[key] = item

    matches: dict[str, RuleMatch] = {
        match.claim_id: match for breakdown in predicted.values() for match in breakdown.matches
    }
    contributions = _contributions(predicted)
    exact = 0
    diffs: list[ItemDiff] = []
    for key, item in expectation_by_key.items():
        claim = claim_by_key.get(key)
        match = matches.get(claim.id) if claim else None
        actual = contributions.get(claim.id, 0.0) if claim else 0.0
        if claim is not None and abs(actual - item.expected) <= tolerance:
            exact += 1
            continue
        kind, reason = _match_kind(match, claim)
        diffs.append(
            ItemDiff(
                student_id=item.student_id,
                claim_id=claim.id if claim else None,
                level=claim.level if claim else None,
                expected=_round(item.expected),
                actual=_round(actual),
                kind=kind,
                reason=reason,
                item_key=item.item_key,
                category=claim.category if claim else None,
                raw_text=claim.raw_text if claim else None,
                attribution=item.attribution,
                suspected_manual_error=item.suspected_manual_error,
            )
        )

    missing_reference = set(claim_by_key) - set(expectation_by_key)
    if missing_reference:
        issues.append(f"{len(missing_reference)} 条申报缺少逐项参照分值")
        for sid, item_key in sorted(missing_reference):
            claim = claim_by_key[(sid, item_key)]
            actual = contributions.get(claim.id, 0.0)
            diffs.append(
                ItemDiff(
                    student_id=sid,
                    claim_id=claim.id,
                    level=claim.level,
                    expected=None,
                    actual=_round(actual),
                    kind="missing_reference_item",
                    reason="本次申报明细存在该项，但逐项参照表未提供分值",
                    item_key=item_key,
                    category=claim.category,
                    raw_text=claim.raw_text,
                )
            )
    missing_claims = set(expectation_by_key) - set(claim_by_key)
    if missing_claims:
        issues.append(f"{len(missing_claims)} 条逐项参照在申报明细中不存在")
    unattributed = [diff for diff in diffs if diff.expected is not None and not diff.attribution]
    if unattributed:
        issues.append(f"{len(unattributed)} 条逐项差异尚未填写差异归因")
    return exact, diffs, issues


def run_backtest(
    claims: Iterable[Claim],
    ruleset: Ruleset,
    ground_truth: dict[str, float],
    *,
    item_expectations: Sequence[ItemExpectation] | None = None,
    mode: BacktestMode = "historical_reference",
    required_students: int | None = None,
    academic_year: str | None = None,
    college: str | None = None,
    config: EngineConfig | None = None,
    tolerance: float = 0.01,
) -> BacktestReport:
    """核算全部学生，执行逐人/逐项比较并计算验收门禁。"""
    claim_list = list(claims)
    predicted = compute_all(
        claim_list, ruleset, academic_year=academic_year, college=college, config=config
    )
    report = compare_students(predicted, ground_truth, tolerance=tolerance, mode=mode)
    report.required_students = required_students
    report.meta.update({"claims": len(claim_list), "rules": len(ruleset)})

    if item_expectations is not None:
        report.total_items = len(item_expectations)
        report.matched_items, report.diffs, issues = compare_items(
            claim_list, predicted, item_expectations, tolerance=tolerance
        )
        report.data_issues.extend(issues)
        if not item_expectations:
            report.data_issues.append("逐项参照表没有可用的已标注分值")
        item_totals: dict[str, float] = {}
        for item in item_expectations:
            item_totals[item.student_id] = item_totals.get(item.student_id, 0.0) + item.expected
        inconsistent_totals = [
            sid
            for sid, expected_total in ground_truth.items()
            if abs(item_totals.get(sid, 0.0) - expected_total) > tolerance
        ]
        if inconsistent_totals:
            report.data_issues.append(
                f"{len(inconsistent_totals)} 人的逐项参照合计与汇总参照不一致"
            )
        if mode == "adjudicated_truth" and any(not item.adjudicated for item in item_expectations):
            report.data_issues.append("准确率口径要求全部逐项结果均已业务裁决")
    else:
        report.diffs = _explain_total_diffs(predicted, ground_truth, tolerance=tolerance)
        report.data_issues.append("缺少逐项参照表，不能计算逐项指标或输出完整逐项差异")

    if report.matched_students != report.total_students:
        report.data_issues.append("部分汇总表学生在申报明细中不存在")
    if set(predicted) - set(ground_truth):
        report.data_issues.append("部分申报学生在汇总参照表中不存在")
    if not ground_truth:
        report.data_issues.append("汇总参照表没有可用学生记录")
    if required_students is not None and report.total_students != required_students:
        report.data_issues.append(
            f"学生数不满足门禁：要求 {required_students}，实际 {report.total_students}"
        )

    report.data_issues = list(dict.fromkeys(report.data_issues))
    report.data_complete = not report.data_issues
    report.gate_passed = report.data_complete if required_students is not None else None
    return report


def _explain_total_diffs(
    predicted: dict[str, ScoreBreakdown],
    expected: dict[str, float],
    *,
    tolerance: float,
) -> list[ItemDiff]:
    """无逐项参照时仅提供总分级初步归因，不冒充逐项差异。"""
    out: list[ItemDiff] = []
    for sid, exp in expected.items():
        pred = predicted.get(sid)
        if pred is None:
            out.append(ItemDiff(sid, None, None, _round(exp), 0.0, "missing_student", "该学号没有申报条目"))
            continue
        if abs(pred.total - exp) <= tolerance:
            continue
        if pred.unmatched_claims:
            kind = "unmatched_rule"
            reason = f"{len(pred.unmatched_claims)} 条申报未命中规则"
        elif pred.review_claims:
            kind = "manual_review"
            reason = f"{len(pred.review_claims)} 条申报需人工复核"
        else:
            kind = "total_mismatch"
            reason = "总分不符；缺少逐项参照，无法继续定位"
        out.append(ItemDiff(sid, None, None, _round(exp), _round(pred.total), kind, reason))
    return out


def _read_excel(path: str | Path, *, sheet: str | int) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"评测文件不存在：{p}", detail={"path": str(p)})
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, dtype=str)
        return pd.read_excel(p, sheet_name=sheet)
    except (ImportError, ValueError, OSError) as exc:
        raise DataSourceError(f"无法读取评测文件：{p}（{exc}）", detail={"path": str(p)}) from exc


def _column(df: pd.DataFrame, requested: str, aliases: Sequence[str]) -> Any | None:
    cols = {str(c).strip(): c for c in df.columns}
    for name in (requested, *aliases):
        if name in cols:
            return cols[name]
    return None


def load_ground_truth(
    path: str | Path,
    *,
    sheet: str | int = 0,
    student_col: str = "学号",
    total_col: str = "总分",
) -> dict[str, float]:
    """从历史汇总表读取 ``学号 -> 总分``。"""
    df = _read_excel(path, sheet=sheet)
    sid_col = _column(df, student_col, ("学生学号",))
    score_col = _column(df, total_col, ("历史总分", "裁决总分"))
    if sid_col is None or score_col is None:
        raise DataSourceError(
            f"评测表缺少列：{student_col} / {total_col}",
            detail={"found_columns": list(df.columns)},
        )
    truth: dict[str, float] = {}
    for _, row in df.iterrows():
        sid, total = row[sid_col], _num(row[score_col])
        if sid is None or total is None:
            continue
        key = str(sid).strip()
        if key in truth:
            raise DataSourceError(f"汇总参照表学号重复：{key}")
        truth[key] = total
    return truth


def load_item_expectations(
    path: str | Path,
    *,
    sheet: str | int = 0,
    student_col: str = "学号",
    item_col: str = "申报项标识",
    score_col: str = "历史/裁决得分",
) -> list[ItemExpectation]:
    """读取逐项参照。空分值行视为未标注并跳过，完整性门禁会据此失败。"""
    df = _read_excel(path, sheet=sheet)
    sid_key = _column(df, student_col, ("学生学号",))
    item_key = _column(df, item_col, ("申报项ID", "申报项编号", "明细行号"))
    value_key = _column(df, score_col, ("历史得分", "裁决得分", "预期得分"))
    if sid_key is None or item_key is None or value_key is None:
        raise DataSourceError(
            f"逐项参照表缺少列：{student_col} / {item_col} / {score_col}",
            detail={"found_columns": list(df.columns)},
        )
    adjudicated_key = _column(df, "已裁决", ("裁决状态",))
    attribution_key = _column(df, "差异归因", ("差异原因", "错误来源"))
    suspected_key = _column(df, "疑似人工错误", ("人工结果疑似错误",))
    note_key = _column(df, "备注", ("说明",))
    out: list[ItemExpectation] = []
    seen: set[tuple[str, str]] = set()
    line_number_only = item_key == _column(df, "明细行号", ())
    for _, row in df.iterrows():
        score = _num(row[value_key])
        if score is None:
            continue
        sid = str(row[sid_key]).strip()
        raw_item_key = str(row[item_key]).strip()
        if line_number_only and raw_item_key.replace(".0", "").isdigit():
            raw_item_key = f"row:{raw_item_key.replace('.0', '')}"
        if not sid or sid.lower() == "nan" or not raw_item_key or raw_item_key.lower() == "nan":
            continue
        identity = (sid, raw_item_key)
        if identity in seen:
            raise DataSourceError(f"逐项参照标识重复：{sid} / {raw_item_key}")
        seen.add(identity)
        out.append(
            ItemExpectation(
                student_id=sid,
                item_key=raw_item_key,
                expected=score,
                adjudicated=_bool(row[adjudicated_key]) if adjudicated_key is not None else False,
                attribution=str(row[attribution_key]).strip()
                if attribution_key is not None and pd.notna(row[attribution_key])
                else None,
                suspected_manual_error=_bool(row[suspected_key]) if suspected_key is not None else False,
                note=str(row[note_key]).strip()
                if note_key is not None and pd.notna(row[note_key])
                else None,
            )
        )
    return out


def item_reference_template(claims: Iterable[Claim]) -> pd.DataFrame:
    """生成逐项参照模板；输出行号标识可跨多次回测稳定复用。"""
    rows: list[dict[str, Any]] = []
    for claim in claims:
        rows.append(
            {
                "学号": claim.student_id,
                "姓名": claim.student_name,
                "申报项标识": claim_item_key(claim),
                "明细行号": claim.source_ref.row if claim.source_ref else None,
                "类别": claim.category,
                "申报内容": claim.raw_text,
                "等级": claim.level,
                "历史/裁决得分": None,
                "已裁决": "否",
                "差异归因": None,
                "疑似人工错误": "否",
                "备注": None,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BacktestMode",
    "BacktestReport",
    "ItemDiff",
    "ItemExpectation",
    "StudentResult",
    "claim_item_key",
    "compare_items",
    "compare_students",
    "item_reference_template",
    "load_ground_truth",
    "load_item_expectations",
    "run_backtest",
]
