"""存储层测试：SQLite 往返、过滤、导出/导入 JSON。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scoreproof.rules.extractor import HeuristicExtractor, RuleDraft, drafts_to_rules
from scoreproof.rules.store import RuleStore, export_json, import_json
from scoreproof.schema import Ruleset

from .conftest import make_rule


@pytest.fixture
def store(tmp_path: Path) -> RuleStore:
    s = RuleStore(tmp_path / "rules.sqlite")
    yield s
    s.close()


class TestRuleStore:
    def test_upsert_and_count(self, store: RuleStore) -> None:
        rules = [
            make_rule("省级一等奖", 10, rule_id="r1"),
            make_rule("省级二等奖", 8, rule_id="r2"),
        ]
        assert store.upsert_rules(rules) == 2
        assert store.count() == 2

    def test_roundtrip_preserves_fields(self, store: RuleStore) -> None:
        original = make_rule(
            "省级二等奖", 8, group="竞赛", cap=20, team_factor=0.5,
            synonyms=["省二等奖", "省赛第二名"], require_catalog="认可竞赛目录", rule_id="r1",
        )
        store.upsert_rules([original])
        loaded = store.list_rules()[0]
        assert loaded.level == original.level
        assert loaded.score == original.score
        assert loaded.synonyms == original.synonyms
        assert loaded.constraints.dedup_group == "竞赛"
        assert loaded.constraints.cap == 20
        assert loaded.constraints.team_factor == 0.5
        assert loaded.constraints.require_catalog == "认可竞赛目录"
        assert loaded.source.doc == original.source.doc
        assert loaded.source.page == original.source.page

    def test_upsert_is_idempotent(self, store: RuleStore) -> None:
        rule = make_rule("省级一等奖", 10, rule_id="r1")
        store.upsert_rules([rule])
        store.upsert_rules([rule])
        assert store.count() == 1

    def test_upsert_updates_score(self, store: RuleStore) -> None:
        store.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
        store.upsert_rules([make_rule("省级一等奖", 12, rule_id="r1")])
        assert store.list_rules()[0].score == 12
        assert store.count() == 1

    def test_filter_by_year_and_college(self, store: RuleStore) -> None:
        store.upsert_rules([
            make_rule("省级一等奖", 10, academic_year="2024-2025", rule_id="old"),
            make_rule("省级一等奖", 12, academic_year="2025-2026", rule_id="new"),
            make_rule("省级一等奖", 14, academic_year="2025-2026", college="计算机学院",
                      rule_id="cs"),
        ])
        assert {r.id for r in store.list_rules(academic_year="2025-2026")} == {"new", "cs"}
        # 学院过滤时，校级通用规则（college=None）仍然生效
        assert {r.id for r in store.list_rules(college="计算机学院")} == {"old", "new", "cs"}

    def test_disable_and_delete(self, store: RuleStore) -> None:
        store.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
        assert store.set_enabled("r1", False)
        assert store.list_rules() == []
        assert len(store.list_rules(enabled_only=False)) == 1
        assert store.delete_rule("r1") is True
        assert store.count() == 0
        assert store.delete_rule("r1") is False

    def test_load_ruleset(self, store: RuleStore) -> None:
        store.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
        rs = store.load_ruleset()
        assert isinstance(rs, Ruleset) and len(rs) == 1

    def test_json_export_import_roundtrip(self, store: RuleStore, tmp_path: Path) -> None:
        store.upsert_rules([make_rule("省级一等奖", 10, synonyms=["省一等奖"], rule_id="r1")])
        path = export_json(store.load_ruleset(), tmp_path / "rules.json")
        assert path.exists()
        again = import_json(path)
        assert len(again) == 1
        assert again.rules[0].synonyms == ["省一等奖"]

    def test_reopen_persists(self, tmp_path: Path) -> None:
        db = tmp_path / "rules.sqlite"
        with RuleStore(db) as s1:
            s1.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
        with RuleStore(db) as s2:
            assert s2.count() == 1


class TestExtractor:
    def test_heuristic_extract(self) -> None:
        text = """
        第三章 加分标准
        省级一等奖 10分
        省级二等奖 8 分
        这个行没有分值
        """
        drafts = HeuristicExtractor().extract(text, category="学科竞赛")
        levels = {d.level for d in drafts}
        assert "省级一等奖" in levels and "省级二等奖" in levels
        assert all(d.evidence_quote for d in drafts)
        assert all(d.confidence < 1.0 for d in drafts)  # 程序抽取必须标记为"待校对"

    def test_draft_to_rule_normalizes_level(self) -> None:
        rule = RuleDraft(
            category="学科竞赛", level="省二等奖", score=8, evidence_quote="省二等奖 8分"
        ).to_rule(academic_year="2025", doc="细则.pdf", page=4)
        assert rule.level == "省级二等奖"
        assert rule.academic_year == "2025-2026"
        assert "省二等奖" in rule.synonyms
        assert rule.source.doc == "细则.pdf" and rule.source.page == 4

    def test_drafts_without_quote_are_dropped(self) -> None:
        """不可溯源 = 不可用：没有原文片段的草稿一律丢弃。"""
        drafts = [
            RuleDraft(category="学科竞赛", level="省级一等奖", score=10, evidence_quote=""),
            RuleDraft(category="学科竞赛", level="省级二等奖", score=8, evidence_quote="有原文"),
        ]
        rules = drafts_to_rules(drafts, academic_year="2025-2026", doc="细则.pdf")
        assert len(rules) == 1 and rules[0].score == 8

    def test_llm_extractor_unavailable_without_key(self, monkeypatch) -> None:
        from scoreproof.config import Settings
        from scoreproof.rules.extractor import LLMExtractor

        monkeypatch.setattr(
            "scoreproof.rules.extractor.get_settings",
            lambda: Settings(llm_api_key=None),
        )
        extractor = LLMExtractor()
        from scoreproof.errors import DataSourceError

        with pytest.raises(DataSourceError):
            extractor.extract("省级一等奖 10分")

    def test_llm_prompt_forbids_guessing(self) -> None:
        from scoreproof.rules.extractor import LLMExtractor

        messages = LLMExtractor().build_prompt("省级一等奖 10分")
        joined = " ".join(m["content"] for m in messages)
        assert "禁止推算" in joined and "evidence_quote" in joined
