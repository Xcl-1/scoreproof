"""接入层测试：合并单元格 fill-down、表头探测、列名模糊匹配、出处定位。

"Excel 合并单元格必须 fill-down" 是本项目最经典的翻车点，这里必须覆盖。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scoreproof.ingest.excel_loader import (
    detect_header_row,
    fill_merged_cells,
    group_claims_by_student,
    load_claims,
    load_rules,
    load_sheet,
    load_workbook_grid,
    normalize_header,
    to_number,
)


class TestHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("学号", "学号"),
            (" 学 号 ", "学号"),
            ("加分值（分）", "加分值"),
            ("获奖等级\n(必填)", "获奖等级"),
            (None, ""),
            ("项目 类别", "项目类别"),
        ],
    )
    def test_normalize_header(self, raw, expected: str) -> None:
        assert normalize_header(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("8", 8.0), ("8分", 8.0), ("8.5", 8.5), ("", None), (None, None), ("无", None)],
    )
    def test_to_number(self, raw, expected: float | None) -> None:
        assert to_number(raw) == expected

    def test_to_number_default(self) -> None:
        assert to_number("无", default=1.0) == 1.0

    def test_detect_header_row(self) -> None:
        df = pd.DataFrame([["某某学院综测加分细则", None, None], ["类别", "等级", "分值"], [1, 2, 3]])
        assert detect_header_row(df) == 1


class TestFillMergedCells:
    def test_fill_down_vertical_merge(self) -> None:
        """纵向合并：下面几行的类别必须补齐，否则整表分组错位。"""
        df = pd.DataFrame([["学科竞赛", "省级一等奖"], [None, "省级二等奖"], [None, "省级三等奖"]])
        out = fill_merged_cells(df, [(1, 1, 3, 1)])
        assert list(out[0]) == ["学科竞赛"] * 3

    def test_fill_right_horizontal_merge(self) -> None:
        df = pd.DataFrame([["加分标准", None, None], ["类别", "等级", "分值"]])
        out = fill_merged_cells(df, [(1, 1, 1, 3)])
        assert list(out.iloc[0]) == ["加分标准"] * 3

    def test_no_fill_outside_merged_range(self) -> None:
        """未合并的空单元格必须保持为空（防止列值污染整行）。"""
        df = pd.DataFrame([["学科竞赛", "省级一等奖"], [None, None]])
        out = fill_merged_cells(df, [(1, 1, 2, 1)])
        assert pd.isna(out.iloc[1, 1])

    def test_unknown_range_is_ignored(self) -> None:
        df = pd.DataFrame([["a"]])
        out = fill_merged_cells(df, [(99, 99, 100, 100)])
        assert out.equals(df)

    def test_empty_merged_range_untouched(self) -> None:
        df = pd.DataFrame([[None, "x"], [None, "y"]])
        out = fill_merged_cells(df, [(1, 1, 2, 1)])
        assert pd.isna(out.iloc[0, 0])


class TestLoadSheet:
    def test_rules_sheet_has_categories_filled(self, rules_excel: Path) -> None:
        grid = load_sheet(rules_excel, "加分标准表")
        assert list(grid.df.columns) == ["类别", "等级", "分值", "封顶"]
        assert len(grid) == 5
        # A2:A3 合并 -> 第二行类别必须也是"学科竞赛"
        assert list(grid.df["类别"]) == ["学科竞赛", "学科竞赛", "文体活动", "文体活动", "荣誉称号"]

    def test_column_fuzzy_match(self, rules_excel: Path) -> None:
        grid = load_sheet(rules_excel, "加分标准表")
        assert grid.column("分值") == "分值"
        assert grid.column("加分数值") == "分值"
        assert grid.column("不存在") is None

    def test_missing_sheet_raises(self, rules_excel: Path) -> None:
        from scoreproof.errors import DataSourceError

        with pytest.raises(DataSourceError):
            load_sheet(rules_excel, "不存在的表")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from scoreproof.errors import DataSourceError

        with pytest.raises(DataSourceError):
            load_workbook_grid(tmp_path / "nope.xlsx")


class TestLoadRules:
    def test_basic(self, rules_excel: Path) -> None:
        rules = load_rules(rules_excel, academic_year="2025")
        assert len(rules) == 5
        levels = {r.level for r in rules}
        assert "省级一等奖" in levels and "国家奖学金" in levels
        by_level = {r.level: r for r in rules}
        assert by_level["国家级一等奖"].score == 15
        assert by_level["省级一等奖"].constraints.cap == 20
        assert by_level["省级一等奖"].academic_year == "2025-2026"  # 单年 -> 学年

    def test_synonyms_autofilled(self, rules_excel: Path) -> None:
        rules = {r.level: r for r in load_rules(rules_excel, academic_year="2025")}
        assert "省一等奖" in rules["省级一等奖"].synonyms

    def test_source_is_traceable(self, rules_excel: Path) -> None:
        rule = next(r for r in load_rules(rules_excel, academic_year="2025") if r.level == "省级一等奖")
        assert rule.source.doc == rules_excel.name
        assert rule.source.table == "加分标准表"
        assert rule.source.row == 3  # 表头占第 1 行，数据第二行 = 工作表第 3 行
        assert rule.source.text == "省级一等奖"

    def test_college_and_priority(self, rules_excel: Path) -> None:
        rules = load_rules(rules_excel, academic_year="2025-2026", college="计算机学院", priority=3)
        assert all(r.college == "计算机学院" and r.priority == 3 for r in rules)

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        from scoreproof.errors import DataSourceError

        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "坏表"
        ws.append(["类别", "备注"])  # 没有等级与分值
        ws.append(["学科竞赛", "x"])
        path = tmp_path / "bad.xlsx"
        wb.save(path)
        with pytest.raises(DataSourceError):
            load_rules(path, academic_year="2025-2026", sheet="坏表")


class TestLoadClaims:
    def test_basic(self, claims_excel: Path) -> None:
        claims = load_claims(claims_excel, academic_year="2024-2025")
        assert len(claims) == 5
        assert {c.student_id for c in claims} == {"2023001", "2023002"}

    def test_merged_student_id_filled_down(self, claims_excel: Path) -> None:
        """A2:A4 合并：后两行的学号必须补齐，否则条目会挂到错误的人身上。"""
        claims = load_claims(claims_excel)
        first_three = claims[:3]
        assert all(c.student_id == "2023001" for c in first_three)
        assert all(c.student_name == "张三" for c in first_three)

    def test_level_normalized(self, claims_excel: Path) -> None:
        claims = {c.raw_text: c for c in load_claims(claims_excel)}
        assert claims["省赛三等奖"].level == "省级三等奖"
        assert claims["国赛二等奖"].level == "国家级二等奖"

    def test_team_flag(self, claims_excel: Path) -> None:
        claims = {c.raw_text: c for c in load_claims(claims_excel)}
        assert claims["省赛三等奖"].team is True
        assert claims["国赛二等奖"].team is False

    def test_category_normalized(self, claims_excel: Path) -> None:
        claims = {c.raw_text: c for c in load_claims(claims_excel)}
        assert claims["校运会一等奖"].category == "文体活动"

    def test_source_row_points_to_spreadsheet(self, claims_excel: Path) -> None:
        claim = next(c for c in load_claims(claims_excel) if c.raw_text == "省二等奖")
        assert claim.source_ref is not None
        assert claim.source_ref.row == 2  # 表头第 1 行，第一条数据在第 2 行
        assert claim.source_ref.table == "综测汇总"

    def test_claim_ids_are_stable_across_reloads(self, claims_excel: Path) -> None:
        first = [claim.id for claim in load_claims(claims_excel)]
        second = [claim.id for claim in load_claims(claims_excel)]
        assert first == second
        assert len(first) == len(set(first))

    def test_group_by_student(self, claims_excel: Path) -> None:
        grouped = group_claims_by_student(load_claims(claims_excel))
        assert len(grouped["2023001"]) == 3 and len(grouped["2023002"]) == 2

    def test_unknown_level_is_not_invented(self, tmp_path: Path) -> None:
        """原始表述认不出来时 level 必须为 None，而不是硬编一个等级。"""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "表"
        ws.append(["学号", "姓名", "类别", "申报内容", "等级"])
        ws.append(["1", "甲", "学科竞赛", "参加了比赛", "别的什么东西"])
        path = tmp_path / "odd.xlsx"
        wb.save(path)
        claim = load_claims(path, sheet="表")[0]
        assert claim.level is None
        assert claim.extra["level_matched"] is False

    def test_explicit_item_key_is_preserved(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["学号", "申报项ID", "申报内容", "等级"])
        ws.append(["1", "award-001", "省二等奖", "省级二等奖"])
        path = tmp_path / "item-key.xlsx"
        wb.save(path)
        claim = load_claims(path)[0]
        assert claim.extra["item_key"] == "award-001"
