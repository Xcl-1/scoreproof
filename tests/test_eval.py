"""回测层测试：准确率指标、差异归因、ground truth 读取。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scoreproof.calc.engine import compute_all
from scoreproof.cli import app
from scoreproof.eval.backtest import (
    ItemExpectation,
    claim_item_key,
    compare_students,
    item_reference_template,
    load_ground_truth,
    load_item_expectations,
    run_backtest,
)
from scoreproof.rules.store import RuleStore
from scoreproof.schema import Claim, Ruleset, ScoreBreakdown, SourceRef

from .conftest import make_claim, make_rule


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
        for key in ("person_agreement", "person_exact", "total_students", "example_diffs"):
            assert key in summary
        assert "准确率" not in summary["person_metric_name"]


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
        assert report.diffs[0].kind == "missing_student"

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
        assert list(frame.columns) == [
            "student_id", "expected", "actual", "delta", "matched", "review_claims"
        ]
        assert len(frame) == 1


class TestItemBacktest:
    def test_complete_historical_reference_passes_data_gate(self, base_ruleset: Ruleset) -> None:
        claims = [
            make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1"),
            make_claim("省三等奖", level="省级三等奖", student_id="1", claim_id="c2"),
        ]
        items = [
            ItemExpectation("1", "c1", 8.0),
            ItemExpectation("1", "c2", 0.0),
        ]
        report = run_backtest(
            claims, base_ruleset, {"1": 8.0}, item_expectations=items, required_students=1,
            academic_year="2025-2026"
        )
        assert report.item_agreement == 1.0
        assert report.data_complete is True and report.gate_passed is True
        assert report.item_metric_name == "逐项一致率"

    def test_adjudicated_mode_uses_accuracy_label(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")
        report = run_backtest(
            [claim], base_ruleset, {"1": 8.0},
            item_expectations=[ItemExpectation("1", "c1", 8.0, adjudicated=True)],
            mode="adjudicated_truth", required_students=1, academic_year="2025-2026"
        )
        assert report.person_metric_name == "逐人总分准确率"
        assert report.item_metric_name == "逐项准确率"
        assert report.gate_passed is True

    def test_unadjudicated_item_fails_accuracy_gate(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")
        report = run_backtest(
            [claim], base_ruleset, {"1": 8.0},
            item_expectations=[ItemExpectation("1", "c1", 8.0)],
            mode="adjudicated_truth", required_students=1, academic_year="2025-2026"
        )
        assert report.gate_passed is False
        assert any("业务裁决" in issue for issue in report.data_issues)

    def test_difference_requires_attribution(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")
        report = run_backtest(
            [claim], base_ruleset, {"1": 7.0},
            item_expectations=[ItemExpectation("1", "c1", 7.0)],
            required_students=1, academic_year="2025-2026"
        )
        assert report.item_agreement == 0.0
        assert report.gate_passed is False
        assert any("差异归因" in issue for issue in report.data_issues)

    def test_attributed_difference_is_complete(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")
        item = ItemExpectation(
            "1", "c1", 7.0, attribution="历史口径按旧细则计分", suspected_manual_error=True
        )
        report = run_backtest(
            [claim], base_ruleset, {"1": 7.0}, item_expectations=[item],
            required_students=1, academic_year="2025-2026"
        )
        assert report.gate_passed is True
        assert report.diffs[0].suspected_manual_error is True
        assert report.error_distribution == {"score_mismatch": 1}
        assert report.attribution_distribution == {"历史口径按旧细则计分": 1}

    def test_missing_item_reference_fails_gate(self, base_ruleset: Ruleset) -> None:
        claims = [
            make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1"),
            make_claim("省三等奖", level="省级三等奖", student_id="1", claim_id="c2"),
        ]
        report = run_backtest(
            claims, base_ruleset, {"1": 8.0},
            item_expectations=[ItemExpectation("1", "c1", 8.0)],
            required_students=1, academic_year="2025-2026"
        )
        assert report.gate_passed is False
        assert any(diff.kind == "missing_reference_item" for diff in report.diffs)

    def test_reference_for_missing_claim_fails_gate(self, base_ruleset: Ruleset) -> None:
        report = run_backtest(
            [], base_ruleset, {"1": 3.0},
            item_expectations=[ItemExpectation("1", "missing", 3.0, attribution="明细缺失")],
            required_students=1, academic_year="2025-2026"
        )
        assert report.gate_passed is False
        assert report.diffs[0].kind == "missing_claim"

    def test_zero_reference_for_missing_claim_is_still_a_diff(self, base_ruleset: Ruleset) -> None:
        report = run_backtest(
            [], base_ruleset, {"1": 0.0},
            item_expectations=[ItemExpectation("1", "missing", 0.0, attribution="明细缺失")],
            required_students=1, academic_year="2025-2026"
        )
        assert report.item_agreement == 0.0
        assert report.diffs[0].kind == "missing_claim"

    def test_item_sum_must_equal_total_reference(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")
        report = run_backtest(
            [claim], base_ruleset, {"1": 9.0},
            item_expectations=[ItemExpectation("1", "c1", 8.0)],
            required_students=1, academic_year="2025-2026"
        )
        assert report.gate_passed is False
        assert any("逐项参照合计" in issue for issue in report.data_issues)

    def test_group_cap_is_reflected_in_item_contribution(self) -> None:
        ruleset = Ruleset(rules=[make_rule("国家级一等奖", 15, group="竞赛", cap=10)])
        claim = make_claim("国一", level="国家级一等奖", student_id="1", claim_id="c1")
        report = run_backtest(
            [claim], ruleset, {"1": 10.0},
            item_expectations=[ItemExpectation("1", "c1", 10.0)],
            required_students=1, academic_year="2025-2026"
        )
        assert report.item_agreement == 1.0 and report.gate_passed is True

    def test_required_student_count_is_enforced(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("省二等奖", level="省级二等奖", student_id="1", claim_id="c1")
        report = run_backtest(
            [claim], base_ruleset, {"1": 8.0},
            item_expectations=[ItemExpectation("1", "c1", 8.0)],
            required_students=52, academic_year="2025-2026"
        )
        assert report.gate_passed is False
        assert any("要求 52" in issue for issue in report.data_issues)

    def test_full_report_keeps_all_differences(self, base_ruleset: Ruleset) -> None:
        claims = [
            make_claim("省二等奖", level="省级二等奖", student_id=str(i), claim_id=f"c{i}")
            for i in range(12)
        ]
        items = [
            ItemExpectation(str(i), f"c{i}", 7.0, attribution="测试归因") for i in range(12)
        ]
        report = run_backtest(
            claims, base_ruleset, {str(i): 7.0 for i in range(12)},
            item_expectations=items, academic_year="2025-2026"
        )
        assert len(report.summary()["example_diffs"]) == 10
        assert len(report.to_dict()["diffs"]) == 12

    def test_template_uses_source_row_as_stable_key(self) -> None:
        claim = Claim(
            student_id="1", raw_text="省二等奖", level="省级二等奖",
            source_ref=SourceRef(doc="claims.xlsx", table="明细", row=7)
        )
        assert claim_item_key(claim) == "row:7"
        frame = item_reference_template([claim])
        assert frame.loc[0, "申报项标识"] == "row:7"
        assert frame.loc[0, "历史/裁决得分"] is None

    def test_empty_diff_export_keeps_columns(self) -> None:
        frame = compare_students({}, {}).diffs_frame()
        assert "item_key" in frame.columns and "attribution" in frame.columns


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

    def test_load_item_expectations(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "逐项参照"
        ws.append(["学号", "申报项标识", "历史/裁决得分", "已裁决", "差异归因", "疑似人工错误"])
        ws.append(["2023001", "row:2", "8分", "是", "", "否"])
        ws.append(["2023001", "row:3", 0, "否", "旧表重复计分", "是"])
        ws.append(["2023002", "row:4", None, "否", "待标注", "否"])
        path = tmp_path / "items.xlsx"
        wb.save(path)
        items = load_item_expectations(path, sheet="逐项参照")
        assert len(items) == 2
        assert items[0].expected == 8.0 and items[0].adjudicated is True
        assert items[1].expected == 0.0 and items[1].suspected_manual_error is True

    def test_item_expectations_support_line_number_column(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "明细行号", "历史得分"])
        ws.append(["1", 12, 5])
        path = tmp_path / "line-items.xlsx"
        wb.save(path)
        items = load_item_expectations(path, item_col="明细行号", score_col="历史得分")
        assert items[0].item_key == "row:12"

    def test_duplicate_item_expectation_raises(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        from scoreproof.errors import DataSourceError

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "申报项标识", "历史/裁决得分"])
        ws.append(["1", "x", 5])
        ws.append(["1", "x", 5])
        path = tmp_path / "duplicates.xlsx"
        wb.save(path)
        with pytest.raises(DataSourceError, match="重复"):
            load_item_expectations(path)

    def test_duplicate_total_student_raises(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        from scoreproof.errors import DataSourceError

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "总分"])
        ws.append(["1", 5])
        ws.append(["1", 6])
        path = tmp_path / "duplicates-total.xlsx"
        wb.save(path)
        with pytest.raises(DataSourceError, match="重复"):
            load_ground_truth(path)


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


class TestBacktestCli:
    @staticmethod
    def _files(tmp_path: Path, base_ruleset: Ruleset) -> tuple[Path, Path, Path, Path]:
        openpyxl = pytest.importorskip("openpyxl")
        claims = tmp_path / "claims.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "类别", "申报内容", "等级"])
        ws.append(["1", "学科竞赛", "省二等奖", "省级二等奖"])
        wb.save(claims)

        totals = tmp_path / "totals.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "总分"])
        ws.append(["1", 8])
        wb.save(totals)

        items = tmp_path / "items.csv"
        items.write_text(
            "学号,申报项标识,历史/裁决得分,已裁决\n1,row:2,8,否\n", encoding="utf-8-sig"
        )
        database = tmp_path / "rules.sqlite"
        with RuleStore(database) as store:
            store.upsert_rules(base_ruleset.rules)
        return claims, totals, items, database

    def test_real_cli_writes_complete_report(self, tmp_path: Path, base_ruleset: Ruleset) -> None:
        claims, totals, items, database = self._files(tmp_path, base_ruleset)
        out = tmp_path / "report.json"
        result = CliRunner().invoke(
            app,
            [
                "backtest", str(claims), "--truth", str(totals), "--item-reference", str(items),
                "--year", "2025-2026", "--required-students", "1", "--db", str(database),
                "--out", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["gate_passed"] is True
        assert payload["item_agreement"] == 1.0
        assert payload["meta"]["input_sha256"]["claims"]

    def test_cli_fails_required_52_gate(self, tmp_path: Path, base_ruleset: Ruleset) -> None:
        claims, totals, items, database = self._files(tmp_path, base_ruleset)
        result = CliRunner().invoke(
            app,
            [
                "backtest", str(claims), "--truth", str(totals), "--item-reference", str(items),
                "--year", "2025-2026", "--required-students", "52", "--db", str(database),
            ],
        )
        assert result.exit_code == 2
        assert "要求 52，实际 1" in result.output

    def test_export_template_cli(self, tmp_path: Path, base_ruleset: Ruleset) -> None:
        claims, _, _, _ = self._files(tmp_path, base_ruleset)
        out = tmp_path / "template.csv"
        result = CliRunner().invoke(
            app, ["export-backtest-template", str(claims), "--out", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "row:2" in out.read_text(encoding="utf-8-sig")
