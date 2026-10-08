"""存储层测试：SQLite 往返、过滤、导出/导入 JSON。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.errors import VersionConflict
from scoreproof.rules.extractor import HeuristicExtractor, RuleDraft, drafts_to_rules
from scoreproof.rules.store import RuleStore, export_json, import_json
from scoreproof.schema import Ruleset, stable_rule_id

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

    def test_stable_import_replaces_matching_legacy_random_id(self, store: RuleStore) -> None:
        legacy = make_rule("省级一等奖", 10, rule_id="legacy-random-id")
        store.upsert_rules([legacy])
        stable_id = stable_rule_id(
            academic_year=legacy.academic_year,
            college=legacy.college,
            category=legacy.category,
            level=legacy.level,
            rank=legacy.rank,
            item_name=legacy.item_name,
            source=legacy.source,
        )
        revised = legacy.model_copy(update={"id": stable_id, "score": 12})

        store.upsert_rules([revised])

        assert store.count() == 1
        assert store.list_rules()[0].id == stable_id
        assert store.list_rules()[0].score == 12

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

    def test_publish_creates_content_addressed_version_and_ruleset_declares_it(
        self, store: RuleStore
    ) -> None:
        rule = make_rule("省级一等奖", 10, rule_id="r1")
        store.upsert_rules(
            [rule], actor="tester", source_kind="unit", source_ref="fixture:rules"
        )

        active = store.active_rule_version()
        assert active is not None
        assert active.version_id.startswith("rv_") and active.active
        assert active.rule_count == 1
        assert store.load_ruleset().version == active.version_id
        assert store.load_ruleset().meta["rule_content_hash"] == active.content_hash
        event = store.list_rule_version_events()[0]
        assert event.action == "publish" and event.actor == "tester"

    def test_identical_active_snapshot_is_idempotent(self, store: RuleStore) -> None:
        rule = make_rule("省级一等奖", 10, rule_id="r1")
        store.upsert_rules([rule])
        active = store.active_rule_version()
        assert active is not None

        result = store.publish_rule_snapshot(
            [rule], actor="tester", source_kind="unit"
        )

        assert result.action == "unchanged"
        assert result.version_id == active.version_id
        assert len(store.list_rule_versions()) == 1
        assert len(store.list_rule_version_events()) == 1

    def test_rollback_restores_complete_snapshot_and_appends_audit(
        self, store: RuleStore
    ) -> None:
        store.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")], actor="alice")
        first = store.active_rule_version()
        assert first is not None
        store.upsert_rules([make_rule("省级一等奖", 12, rule_id="r1")], actor="bob")
        second = store.active_rule_version()
        assert second is not None and second.version_id != first.version_id

        result = store.rollback_rule_version(
            first.version_id,
            actor="reviewer",
            note="复核发现分值录入错误",
            expected_active_version=second.version_id,
        )

        assert result.action == "rolled_back"
        assert store.list_rules()[0].score == 10
        assert store.active_rule_version() == first.model_copy(update={"active": True})
        events = store.list_rule_version_events()
        assert [event.action for event in events] == ["publish", "publish", "rollback"]
        assert events[-1].previous_version_id == second.version_id
        assert events[-1].actor == "reviewer"

    def test_stale_expected_version_blocks_publish_and_rollback(self, store: RuleStore) -> None:
        first_rule = make_rule("省级一等奖", 10, rule_id="r1")
        store.upsert_rules([first_rule])
        first = store.active_rule_version()
        assert first is not None
        second_rule = make_rule("省级一等奖", 12, rule_id="r1")
        store.upsert_rules([second_rule])
        second = store.active_rule_version()
        assert second is not None

        with pytest.raises(VersionConflict):
            store.publish_rule_snapshot(
                [first_rule],
                actor="stale-writer",
                source_kind="unit",
                expected_active_version=first.version_id,
            )
        with pytest.raises(VersionConflict):
            store.rollback_rule_version(
                first.version_id,
                actor="stale-writer",
                expected_active_version=first.version_id,
            )
        assert store.active_rule_version() == second
        assert store.list_rules()[0].score == 12

    def test_snapshots_and_audit_events_are_database_immutable(self, store: RuleStore) -> None:
        store.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
        active = store.active_rule_version()
        assert active is not None

        with pytest.raises(
            sqlite3.IntegrityError, match="rule versions are immutable"
        ), store._conn:
            store._conn.execute(
                "UPDATE rule_versions SET note='tampered' WHERE id=?",
                (active.version_id,),
            )
        with pytest.raises(
            sqlite3.IntegrityError, match="rule version events are immutable"
        ), store._conn:
            store._conn.execute("DELETE FROM rule_version_events")

    def test_activation_failure_rolls_back_rules_snapshot_pointer_and_audit(
        self, tmp_path: Path
    ) -> None:
        class FailingRuleStore(RuleStore):
            fail_activation = False

            def _before_rule_version_activation(self, version_id: str) -> None:
                if self.fail_activation:
                    raise RuntimeError(f"fault:{version_id}")

        with FailingRuleStore(tmp_path / "fault.sqlite") as failing:
            failing.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
            before = failing.active_rule_version()
            assert before is not None
            failing.fail_activation = True
            with pytest.raises(RuntimeError, match="fault:rv_"):
                failing.upsert_rules([make_rule("省级一等奖", 12, rule_id="r1")])
            assert failing.active_rule_version() == before
            assert failing.list_rules()[0].score == 10
            assert len(failing.list_rule_versions()) == 1
            assert len(failing.list_rule_version_events()) == 1

    def test_cli_lists_audits_and_rolls_back_with_optimistic_lock(self, tmp_path: Path) -> None:
        db = tmp_path / "cli.sqlite"
        with RuleStore(db) as cli_store:
            cli_store.upsert_rules([make_rule("省级一等奖", 10, rule_id="r1")])
            first = cli_store.active_rule_version()
            assert first is not None
            cli_store.upsert_rules([make_rule("省级一等奖", 12, rule_id="r1")])
            second = cli_store.active_rule_version()
            assert second is not None

        runner = CliRunner()
        rolled_back = runner.invoke(
            app,
            [
                "rollback-rule-version",
                first.version_id,
                "--db",
                str(db),
                "--expected-version",
                second.version_id,
                "--actor",
                "cli-test",
            ],
        )
        assert rolled_back.exit_code == 0, rolled_back.stdout
        assert '"action": "rolled_back"' in rolled_back.stdout
        stale = runner.invoke(
            app,
            [
                "rollback-rule-version",
                second.version_id,
                "--db",
                str(db),
                "--expected-version",
                second.version_id,
                "--actor",
                "stale-cli-test",
            ],
        )
        assert stale.exit_code == 2
        assert "活动规则版本已变化" in stale.stdout
        listed = runner.invoke(app, ["list-rule-versions", "--db", str(db), "--json"])
        assert listed.exit_code == 0 and '"active": true' in listed.stdout
        audited = runner.invoke(app, ["rule-version-audit", "--db", str(db), "--json"])
        assert audited.exit_code == 0 and '"action": "rollback"' in audited.stdout

        with RuleStore(db) as verified:
            assert verified.list_rules()[0].score == 10


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

    def test_rule_id_stays_stable_when_score_is_revised_at_same_source(self) -> None:
        before = RuleDraft(
            category="学科竞赛", level="省二等奖", score=8, evidence_quote="省二等奖 8分"
        ).to_rule(academic_year="2025", doc="细则.pdf", page=4, char_start=20, char_end=27)
        after = RuleDraft(
            category="学科竞赛", level="省二等奖", score=10, evidence_quote="省二等奖 10分"
        ).to_rule(academic_year="2025", doc="细则.pdf", page=4, char_start=20, char_end=28)
        assert before.id == after.id

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
