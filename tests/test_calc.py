"""算分核心测试：取最高 / 封顶 / 团队折算 / 互斥 / 拒答 / 引用可追溯。

这些用例就是"计算准确率"的地基 —— 引擎行为一旦变化，这里必须红。
"""

from __future__ import annotations

import pytest

from scoreproof.calc.engine import (
    EngineConfig,
    RuleIndex,
    apply_cap,
    apply_team_factor,
    compute_claims,
    dedup_take_max,
    match_claim,
    resolve_exclusive_groups,
)
from scoreproof.schema import RuleMatch, Ruleset

from .conftest import make_claim, make_rule

# ======================================================================
# 纯函数
# ======================================================================


class TestPureFunctions:
    @pytest.mark.parametrize(
        ("score", "team", "factor", "expected"),
        [
            (8, True, 0.5, 4.0),
            (8, False, 0.5, 8.0),  # 个人奖不折算
            (8, True, 1.0, 8.0),
            (8, True, 0.0, 0.0),
        ],
    )
    def test_team_factor(self, score: float, team: bool, factor: float, expected: float) -> None:
        assert apply_team_factor(score, team, factor) == pytest.approx(expected)

    def test_team_factor_disabled(self) -> None:
        assert apply_team_factor(8, True, 0.5, enabled=False) == 8.0

    @pytest.mark.parametrize(
        ("score", "cap", "expected", "capped"),
        [
            (8, None, 8.0, False),  # 不设额度
            (8, 10, 8.0, False),
            (8, 8, 8.0, False),  # 正好等于上限不算截断
            (25, 20, 20.0, True),
            (0, 20, 0.0, False),
        ],
    )
    def test_cap(self, score: float, cap: float | None, expected: float, capped: bool) -> None:
        assert apply_cap(score, cap) == (pytest.approx(expected), capped)

    def test_cap_disabled(self) -> None:
        assert apply_cap(25, 20, enabled=False) == (25.0, False)

    def test_dedup_take_max(self) -> None:
        a = RuleMatch(claim_id="c1", rule_id="r1", score=8.0)
        b = RuleMatch(claim_id="c2", rule_id="r2", score=12.0)
        c = RuleMatch(claim_id="c3", rule_id="r3", score=5.0)
        assert dedup_take_max([a, b, c]).rule_id == "r2"
        assert dedup_take_max([]) is None

    def test_dedup_tie_is_deterministic(self) -> None:
        """同分时的裁决必须稳定可复现（否则回测数字会飘）。"""
        a = RuleMatch(claim_id="c1", rule_id="r_b", score=8.0)
        b = RuleMatch(claim_id="c2", rule_id="r_a", score=8.0)
        assert dedup_take_max([a, b]).rule_id == "r_a"
        assert dedup_take_max([b, a]).rule_id == "r_a"


# ======================================================================
# 匹配
# ======================================================================


class TestMatching:
    def test_exact_level(self, base_ruleset: Ruleset) -> None:
        idx = RuleIndex(base_ruleset)
        outcome = match_claim(make_claim(level="省级二等奖"), idx)
        assert outcome.matched
        assert outcome.rule is not None and outcome.rule.score == 8
        assert outcome.strategy == "exact_level"
        assert outcome.confidence == 1.0

    def test_synonym(self, base_ruleset: Ruleset) -> None:
        """等级写成同义词，也应精确命中（不算降级）。"""
        idx = RuleIndex(base_ruleset)
        outcome = match_claim(make_claim(level="省二等奖"), idx)
        assert outcome.matched and outcome.strategy == "synonym"
        assert outcome.rule is not None and outcome.rule.level == "省级二等奖"

    def test_raw_text_normalized(self, base_ruleset: Ruleset) -> None:
        """level 为空但有原始表述 -> 归一化后命中，置信度略降。"""
        idx = RuleIndex(base_ruleset)
        outcome = match_claim(make_claim(raw_text="省赛第二名", level=None), idx)
        assert outcome.matched
        assert outcome.rule is not None and outcome.rule.level == "省级二等奖"
        assert outcome.confidence < 1.0

    def test_category_fallback_is_low_confidence(self, base_ruleset: Ruleset) -> None:
        """等级识别不出时按类别降级，必须标低置信 + 待复核。"""
        idx = RuleIndex(base_ruleset)
        outcome = match_claim(make_claim(raw_text="参加了比赛", level=None), idx)
        assert outcome.matched
        assert outcome.strategy == "category_fallback"
        assert outcome.confidence < EngineConfig().review_threshold

    def test_fallback_disabled(self, base_ruleset: Ruleset) -> None:
        idx = RuleIndex(base_ruleset)
        cfg = EngineConfig(fuzzy_fallback=False)
        outcome = match_claim(make_claim(raw_text="参加了比赛", level=None), idx, config=cfg)
        assert not outcome.matched

    def test_no_match_refuses(self, base_ruleset: Ruleset) -> None:
        idx = RuleIndex(base_ruleset)
        outcome = match_claim(
            make_claim(raw_text="宿舍卫生优秀", level=None, category="宿舍管理"), idx
        )
        assert not outcome.matched
        assert outcome.reason and "未找到" in outcome.reason

    def test_college_scoping(self) -> None:
        """学院规则不影响其他学院；college=None 视为校级通用。"""
        rs = Ruleset(
            rules=[
                make_rule("省级一等奖", 10, college=None, rule_id="general"),
                make_rule("省级一等奖", 12, college="计算机学院", rule_id="cs"),
                make_rule("省级一等奖", 9, college="外国语学院", rule_id="fl"),
            ]
        )
        cs = RuleIndex(rs, college="计算机学院")
        assert cs.lookup_level("省级一等奖")[0].id == "cs"
        other = RuleIndex(rs, college="数学学院")
        assert other.lookup_level("省级一等奖")[0].id == "general"

    def test_year_filtering(self) -> None:
        rs = Ruleset(
            rules=[
                make_rule("省级一等奖", 10, academic_year="2024-2025", rule_id="old"),
                make_rule("省级一等奖", 12, academic_year="2025-2026", rule_id="new"),
            ]
        )
        idx = RuleIndex(rs, academic_year="2025-2026")
        assert idx.lookup_level("省级一等奖")[0].id == "new"
        assert len(idx) == 1

    def test_priority_breaks_same_score_ties(self) -> None:
        """同等级多条规则：优先级高的胜出，保证可复现。"""
        rs = Ruleset(
            rules=[
                make_rule("省级一等奖", 10, priority=0, rule_id="low"),
                make_rule("省级一等奖", 10, priority=5, rule_id="high"),
            ]
        )
        idx = RuleIndex(rs)
        assert idx.lookup_level("省级一等奖")[0].id == "high"


# ======================================================================
# 互斥组
# ======================================================================


class TestExclusiveGroups:
    def test_two_groups_keep_higher(self) -> None:
        winners = {
            "a": RuleMatch(claim_id="c1", rule_id="r1", score=10.0),
            "b": RuleMatch(claim_id="c2", rule_id="r2", score=6.0),
        }
        rules_by_group = {
            "a": [make_rule("省级一等奖", 10, group="a", exclusive_with=["b"], rule_id="r1")],
            "b": [make_rule("省级二等奖", 6, group="b", rule_id="r2")],
        }
        keep, dropped = resolve_exclusive_groups(winners, rules_by_group)
        assert set(keep) == {"a"}
        assert dropped == ["b"]

    def test_non_conflicting_groups_both_kept(self) -> None:
        winners = {
            "a": RuleMatch(claim_id="c1", rule_id="r1", score=10.0),
            "b": RuleMatch(claim_id="c2", rule_id="r2", score=6.0),
        }
        rules_by_group = {
            "a": [make_rule("省级一等奖", 10, group="a", rule_id="r1")],
            "b": [make_rule("省级二等奖", 6, group="b", rule_id="r2")],
        }
        keep, dropped = resolve_exclusive_groups(winners, rules_by_group)
        assert set(keep) == {"a", "b"}
        assert dropped == []

    def test_three_way_conflict_picks_best_combo(self) -> None:
        """a-b 互斥、b-c 互斥：a+c(10+4) 优于单独 b(9)。"""
        winners = {
            "a": RuleMatch(claim_id="c1", rule_id="r1", score=10.0),
            "b": RuleMatch(claim_id="c2", rule_id="r2", score=9.0),
            "c": RuleMatch(claim_id="c3", rule_id="r3", score=4.0),
        }
        rules_by_group = {
            "a": [make_rule("省级一等奖", 10, group="a", exclusive_with=["b"], rule_id="r1")],
            "b": [make_rule("省级二等奖", 9, group="b", exclusive_with=["a", "c"], rule_id="r2")],
            "c": [make_rule("校级一等奖", 4, group="c", rule_id="r3")],
        }
        keep, dropped = resolve_exclusive_groups(winners, rules_by_group)
        assert set(keep) == {"a", "c"}
        assert dropped == ["b"]

    def test_disabled_keeps_all(self) -> None:
        winners = {
            "a": RuleMatch(claim_id="c1", rule_id="r1", score=10.0),
            "b": RuleMatch(claim_id="c2", rule_id="r2", score=6.0),
        }
        rules_by_group = {
            "a": [make_rule("省级一等奖", 10, group="a", exclusive_with=["b"], rule_id="r1")],
            "b": [make_rule("省级二等奖", 6, group="b", rule_id="r2")],
        }
        keep, dropped = resolve_exclusive_groups(winners, rules_by_group, enabled=False)
        assert set(keep) == {"a", "b"} and dropped == []


# ======================================================================
# 端到端（单个学生）
# ======================================================================


class TestComputeClaims:
    def test_same_group_take_max_not_sum(self, base_ruleset: Ruleset) -> None:
        """同类取最高、不累加：8 分与 5 分同组 -> 只算 8。"""
        claims = [
            make_claim("省二等奖", level="省级二等奖", claim_id="c1"),
            make_claim("省三等奖", level="省级三等奖", claim_id="c2"),
        ]
        bd = compute_claims(claims, base_ruleset, academic_year="2025-2026")
        assert bd.total == 8
        counted = [m for m in bd.matches if m.counted]
        assert len(counted) == 1 and counted[0].score == 8

    def test_different_groups_accumulate(self, base_ruleset: Ruleset) -> None:
        """不同组累加：学科竞赛 8 + 文体 6 = 14。"""
        claims = [
            make_claim("省二等奖", level="省级二等奖", category="学科竞赛", claim_id="c1"),
            make_claim("校运会一等奖", level="一等奖", category="文体活动", claim_id="c2"),
        ]
        bd = compute_claims(claims, base_ruleset, academic_year="2025-2026")
        assert bd.total == 14

    def test_group_cap_applies_after_dedup(self, base_ruleset: Ruleset) -> None:
        """封顶作用在"取最高"之后：20 分上限、国一 15 与省一 10 同组 -> 15。"""
        claims = [
            make_claim("国赛一等奖", level="国家级一等奖", claim_id="c1"),
            make_claim("省赛一等奖", level="省级一等奖", claim_id="c2"),
        ]
        bd = compute_claims(claims, base_ruleset, academic_year="2025-2026")
        assert bd.total == 15
        group = next(g for g in bd.groups if g.group == "学科竞赛")
        assert group.cap == 30 and group.capped is False  # 15 < 30

    def test_item_cap(self) -> None:
        rs = Ruleset(rules=[make_rule("省级一等奖", 50, group="学科竞赛", cap=20)])
        bd = compute_claims(
            [make_claim("省一等奖", level="省级一等奖")], rs, academic_year="2025-2026"
        )
        assert bd.total == 20
        assert any(m.capped for m in bd.matches)

    def test_team_factor_applied(self, base_ruleset: Ruleset) -> None:
        """团队奖按 0.5 折算：省级二等奖 8 -> 4。"""
        claims = [make_claim("省二等奖", level="省级二等奖", team=True, claim_id="c1")]
        bd = compute_claims(claims, base_ruleset, academic_year="2025-2026")
        assert bd.total == 4
        assert any("团队奖" in (m.reason or "") for m in bd.matches)

    def test_team_factor_then_cap(self) -> None:
        """先折算后封顶：8*0.5=4，封顶 3 -> 3。"""
        rs = Ruleset(rules=[make_rule("省级二等奖", 8, group="g", cap=3, team_factor=0.5)])
        bd = compute_claims(
            [make_claim("省二等奖", level="省级二等奖", team=True, claim_id="c1")],
            rs, academic_year="2025-2026",
        )
        assert bd.total == 3

    def test_year_mismatch_rejected(self, base_ruleset: Ruleset) -> None:
        claims = [make_claim("省二等奖", level="省级二等奖", academic_year="2020-2021",
                             claim_id="c1")]
        bd = compute_claims(claims, base_ruleset, academic_year="2020-2021")
        assert bd.total == 0
        assert bd.unmatched_claims

    def test_catalog_required(self) -> None:
        rs = Ruleset(
            rules=[make_rule("省级一等奖", 10, group="学科竞赛", require_catalog="认可竞赛目录")]
        )
        ok = compute_claims(
            [make_claim("省一等奖", level="省级一等奖", catalog_listed=True, claim_id="c1")],
            rs, academic_year="2025-2026",
        )
        assert ok.total == 10
        bad = compute_claims(
            [make_claim("省一等奖", level="省级一等奖", catalog_listed=False, claim_id="c1")],
            rs, academic_year="2025-2026",
        )
        assert bad.total == 0
        assert any("认可竞赛目录" in (m.reason or "") for m in bad.matches)

    def test_unmatched_is_not_guessed(self, base_ruleset: Ruleset) -> None:
        """未命中 -> 0 分 + 拒答说明，绝不瞎给分。"""
        bd = compute_claims(
            [make_claim("宿舍卫生优秀", level=None, category="宿舍管理", claim_id="c1")],
            base_ruleset, academic_year="2025-2026",
        )
        assert bd.total == 0
        assert bd.unmatched_claims and bd.matches[0].reason

    def test_traceable_to_source(self, base_ruleset: Ruleset) -> None:
        """每条计入分值都能回溯到原文出处（简历核心卖点）。"""
        bd = compute_claims(
            [make_claim("省二等奖", level="省级二等奖", claim_id="c1")],
            base_ruleset, academic_year="2025-2026",
        )
        m = next(m for m in bd.matches if m.counted)
        assert m.source is not None
        assert m.source.doc == "合成细则.pdf" and m.source.page == 4
        assert "第三章第7条" in m.source.short()

    def test_review_flag_on_low_confidence(self, base_ruleset: Ruleset) -> None:
        claim = make_claim("参加了比赛", level=None, claim_id="c1")
        bd = compute_claims(
            [claim],
            base_ruleset, academic_year="2025-2026",
        )
        assert bd.review_claims
        match = bd.matches[0]
        assert match.needs_review
        assert not match.counted
        assert match.score > 0  # 保留候选分值供复核，但禁止进入总分
        assert bd.total == 0
        assert claim.status == "低置信"
        assert "不计入总分" in (match.reason or "")

    def test_empty_claims(self, base_ruleset: Ruleset) -> None:
        bd = compute_claims([], base_ruleset, academic_year="2025-2026")
        assert bd.total == 0 and bd.groups == []

    def test_deterministic_repeat_runs(self, base_ruleset: Ruleset) -> None:
        claims = [
            make_claim("省二等奖", level="省级二等奖", claim_id="c1"),
            make_claim("国赛一等奖", level="国家级一等奖", claim_id="c2"),
            make_claim("校运会一等奖", level="一等奖", category="文体活动", claim_id="c3"),
        ]
        totals = {
            compute_claims(claims, base_ruleset, academic_year="2025-2026").total for _ in range(5)
        }
        assert len(totals) == 1

    def test_rounding(self) -> None:
        rs = Ruleset(rules=[make_rule("省级二等奖", 7.777, group="g")])
        bd = compute_claims(
            [make_claim("省二等奖", level="省级二等奖", claim_id="c1")],
            rs, academic_year="2025-2026",
        )
        assert bd.total == 7.78

    def test_exclusive_groups_end_to_end(self) -> None:
        rs = Ruleset(
            rules=[
                make_rule("省级一等奖", 10, category="学科竞赛", group="竞赛",
                          exclusive_with=["荣誉"], rule_id="r1"),
                make_rule("国家奖学金", 8, category="荣誉称号", group="荣誉", rule_id="r2"),
            ]
        )
        bd = compute_claims(
            [
                make_claim("省一等奖", level="省级一等奖", category="学科竞赛", claim_id="c1"),
                make_claim("国家奖学金", level="国家奖学金", category="荣誉称号", claim_id="c2"),
            ],
            rs, academic_year="2025-2026",
        )
        assert bd.total == 10  # 竞赛 10 与荣誉 8 互斥，取高者
        assert any("互斥" in n for n in bd.notes)
