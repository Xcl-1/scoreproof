"""归一化单元测试：等级、学年、类别。

这是"脏活"，也是最容易算错分的地方，测试必须覆盖各种野生写法。
"""

from __future__ import annotations

from datetime import date

import pytest

from scoreproof.normalize import (
    academic_year_of,
    canonical_level,
    level_aliases,
    match_academic_year,
    normalize_academic_year,
    normalize_category,
    normalize_level,
    parse_prize,
    parse_tier,
)


class TestLevelNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("省二等奖", "省级二等奖"),
            ("省级二等", "省级二等奖"),
            ("省赛第二名", "省级二等奖"),
            ("省级二等奖", "省级二等奖"),
            ("省 二 等 奖", "省级二等奖"),
            (" 省级  二等奖 ", "省级二等奖"),
            ("全国一等奖", "国家级一等奖"),
            ("国家级第二名", "国家级二等奖"),
            ("省部级一等奖", "省级一等奖"),  # 项目总结明确要求：省部级→省级
            ("校级一等", "校级一等奖"),
            ("院级第三名", "院级三等奖"),
            ("市赛冠军", "市级一等奖"),
            ("亚军", "二等奖"),  # 只给奖项时保留奖项、不臆造级别
            ("一等奖", "一等奖"),
            ("特等奖", "特等奖"),
            ("优秀奖", "优秀奖"),
        ],
    )
    def test_canonical(self, raw: str, expected: str) -> None:
        assert canonical_level(raw) == expected

    def test_fullwidth_and_punctuation(self) -> None:
        assert canonical_level("省级（二等奖）") == "省级二等奖"
        assert canonical_level("省级-二等奖") == "省级二等奖"

    def test_special_awards(self) -> None:
        assert canonical_level("国家奖学金") == "国家奖学金"
        assert canonical_level("校级三好学生") == "三好学生"

    def test_unknown_returns_input_not_guess(self) -> None:
        """认不出来时原样返回 —— 绝不臆造等级。"""
        res = normalize_level("参加志愿服务10小时")
        assert res.matched is False
        assert res.canonical == "参加志愿服务10小时"

    def test_extra_aliases_override(self) -> None:
        res = normalize_level("院级A类", extra_aliases={"院级A类": "省级一等奖"})
        assert res.canonical == "省级一等奖"
        assert res.rule == "alias"

    def test_parse_components(self) -> None:
        assert parse_tier("省二等奖") == "省级"
        assert parse_prize("省二等奖") == "二等奖"
        assert parse_tier("奇奇怪怪") is None
        assert parse_prize("没有名次") is None

    def test_level_aliases_roundtrip(self) -> None:
        aliases = level_aliases("省级二等奖")
        assert "省级二等奖" in aliases
        assert "省二等奖" in aliases
        assert "省级二等" in aliases
        # 每个别名都应归一化回同一规范等级
        for a in aliases:
            assert canonical_level(a) == "省级二等奖"


class TestAcademicYear:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2025-2026", "2025-2026"),
            ("2025", "2025-2026"),
            ("2025~2026", "2025-2026"),
            ("2025至2026", "2025-2026"),
            ("2024-2025", "2024-2025"),
            ("乱写", "乱写"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_academic_year(raw) == expected

    def test_rollover(self) -> None:
        assert academic_year_of(date(2025, 9, 1)) == "2025-2026"
        assert academic_year_of(date(2025, 8, 31)) == "2024-2025"

    def test_match(self) -> None:
        assert match_academic_year("2025", "2025-2026")
        assert not match_academic_year("2024-2025", "2025-2026")


class TestCategoryNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("竞赛", "学科竞赛"),
            ("科技竞赛", "学科竞赛"),
            ("论文", "科研学术"),
            ("志愿", "志愿服务"),
            ("学生工作", "社会工作"),
            ("", "未分类"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_category(raw) == expected

    def test_unknown_keeps_raw(self) -> None:
        assert normalize_category("神秘类别") == "神秘类别"
