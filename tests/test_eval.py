"""回测层测试：准确率指标、差异归因、ground truth 读取。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scoreproof.calc.engine import compute_all
from scoreproof.eval.backtest import (
    compare_students,
    load_ground_truth,
    run_backtest,
)
from scoreproof.rules.store import RuleStore
from scoreproof.schema import Ruleset, ScoreBreakdown

from .conftest import make_claim


class TestCompareStudents:
    def test_perfect_match(self) -> None:
        predicted = {"1": 8.0, "2": 14.0}
        report = compare_students(predicted, {"1": 8.0, "2": 14.0})
        assert report.total_students == 2
        assert report.exact == 2
        assert report.accuracy == 1.0

    def test_mismatch_counted(self) -> None:
        report = compare_students({"1": 8.0, "2": 10.0}, {"1": 8.0, "2": 14.0})
        assert report.exact == 1
        assert report.accuracy == 0.5
        bad = next(r for r in report.results if r.student_id == "2")
        assert bad.delta == -4.0 and bad.matched is False

    def test_tolerance(self) -> None:
        report = compare_students({"1": 8.005}, {"1": 8.0}, tolerance=0.01)
        assert report.exact == 1  # 落在容差内

    def test_missing_prediction_counts_as_zero(self) -> None:
        report = compare_students({"1": 8.0}, {"1": 8.0, "2": 6.0})
        assert report.total_students == 2
        assert report.meta["missing_students"] == ["2"]
        assert report.accuracy == 0.5

    def test_accepts_score_breakdown(self) -> None:
        bd = ScoreBreakdown(student_id="1", total=8.0, review_claims=["c9"])
        report = compare_students({"1": bd}, {"1": 8.0})
        assert report.exact == 1
        assert report.review_claims == 1

    def test_empty_truth(self) -> None:
        report = compare_students({}, {})
        assert report.total_students == 0 and report.accuracy == 0.0

    def test_summary_shape(self) -> None:
        summary = compare_students({"1": 8.0}, {"1": 10.0}).summary()
        for key in ("accuracy", "exact", "total_students", "example_diffs"):
            assert key in summary


class TestRunBacktest:
    def test_end_to_end(self, base_ruleset: Ruleset) -> None:
        """张三分 15（同类取最高），李四 4（团队折算）+ 10 = 14。"""
        claims = [
            make_claim("国赛一等奖", level="国家级一等奖", student_id="2023001", claim_id="c1"),
            make_claim("省赛一等奖", level="省级一等奖", student_id="2023001", claim_id="c2"),
            make_claim("省二等奖", level="省级二等奖", student_id="2023002", team=True, claim_id="c3"),
            make_claim("国奖", level="国家奖学金", category="荣誉称号",
                       student_id="2023002", claim_id="c4"),
        ]
        truth = {"2023001": 15.0, "2023002": 14.0}
        report = run_backtest(claims, base_ruleset, truth, academic_year="2025-2026")
        assert report.accuracy == 1.0
        assert report.exact == 2
        assert report.meta["claims"] == 4
        assert report.meta["rules"] == len(base_ruleset)

    def test_diff_attribution_unmatched(self, base_ruleset: Ruleset) -> None:
        claims = [
            make_claim("宿舍卫生", level=None, category="宿舍管理", student_id="1", claim_id="c1"),
        ]
        report = run_backtest(claims, base_ruleset, {"1": 5.0}, academic_year="2025-2026")
        assert report.accuracy == 0.0
        assert report.diffs and "未命中" in (report.diffs[0].reason or "")

    def test_diff_attribution_review(self, base_ruleset: Ruleset) -> None:
        claims = [make_claim("参加了比赛", level=None, student_id="1", claim_id="c1")]
        report = run_backtest(claims, base_ruleset, {"1": 99.0}, academic_year="2025-2026")
        assert report.diffs and "复核" in (report.diffs[0].reason or "")

    def test_missing_student_in_claims(self, base_ruleset: Ruleset) -> None:
        report = run_backtest([], base_ruleset, {"nobody": 5.0}, academic_year="2025-2026")
        assert report.diffs[0].kind == "missing"

    def test_deterministic(self, base_ruleset: Ruleset) -> None:
        claims = [make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")]
        accuracies = {
            run_backtest(claims, base_ruleset, {"1": 8.0}, academic_year="2025-2026").accuracy
            for _ in range(5)
        }
        assert accuracies == {1.0}

    def test_to_frame(self, base_ruleset: Ruleset) -> None:
        claims = [make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")]
        frame = run_backtest(claims, base_ruleset, {"1": 8.0}, academic_year="2025-2026").to_frame()
        assert list(frame.columns) == ["student_id", "expected", "actual", "delta", "matched"]
        assert len(frame) == 1


class TestGroundTruth:
    def test_load(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "汇总"
        ws.append(["学号", "总分"])
        ws.append(["2023001", 15])
        ws.append(["2023002", "14分"])
        ws.append(["2023003", None])  # 空值应跳过
        path = tmp_path / "truth.xlsx"
        wb.save(path)
        truth = load_ground_truth(path, sheet="汇总")
        assert truth == {"2023001": 15.0, "2023002": 14.0}

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        from scoreproof.errors import DataSourceError

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "备注"])
        path = tmp_path / "bad.xlsx"
        wb.save(path)
        with pytest.raises(DataSourceError):
            load_ground_truth(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from scoreproof.errors import DataSourceError

        with pytest.raises(DataSourceError):
            load_ground_truth(tmp_path / "nope.xlsx")


class TestComputeAll:
    def test_groups_by_student(self, base_ruleset: Ruleset) -> None:
        claims = [
            make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1"),
            make_claim("国赛一等奖", level="国家级一等奖", student_id="2", claim_id="c2"),
        ]
        results = compute_all(claims, base_ruleset, academic_year="2025-2026")
        assert set(results) == {"1", "2"}
        assert results["1"].total == 8
        assert results["2"].total == 15

    def test_empty(self, base_ruleset: Ruleset) -> None:
        assert compute_all([], base_ruleset, academic_year="2025-2026") == {}

    def test_rules_from_store_match_in_memory(self, base_ruleset: Ruleset, tmp_path: Path) -> None:
        """入库 -> 读回 -> 算分结果必须完全一致（防止序列化丢语义）。"""
        with RuleStore(tmp_path / "r.sqlite") as store:
            store.upsert_rules(base_ruleset.rules)
            loaded = store.load_ruleset()
        claims = [
            make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1"),
            make_claim("省二等奖", level="省级二等奖", student_id="1", team=True, claim_id="c2"),
        ]
        a = compute_all(claims, base_ruleset, academic_year="2025-2026")
        b = compute_all(claims, loaded, academic_year="2025-2026")
        assert a["1"].total == b["1"].total == 8
