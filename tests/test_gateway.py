"""LLM 抽取验证网关：五道校验、评测、缓存与独立发布门禁。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.errors import SchemaValidationError, VersionConflict
from scoreproof.eval.gateway import GatewayNegativeCase, evaluate_gateway_negatives, wilson_interval
from scoreproof.rules.extractor import LLMExtractor
from scoreproof.rules.gateway import ExtractionGateway, GatewayContext, GatewayIssue, chunk_hash
from scoreproof.rules.store import RuleStore

from .conftest import make_rule


def payload(**overrides) -> dict:
    value = {
        "category": "学科竞赛",
        "level": "省级二等奖",
        "score": 8.0,
        "evidence_quote": "省级二等奖加8分",
    }
    value.update(overrides)
    return value


@pytest.fixture
def context() -> GatewayContext:
    return GatewayContext(
        academic_year="2025-2026", doc="合成细则.pdf", college="计算机学院", page=4
    )


@pytest.fixture
def gateway() -> ExtractionGateway:
    return ExtractionGateway()


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.items[0].issues}


class TestGatewayLayers:
    def test_layer_1_rejects_missing_field(self, gateway, context) -> None:
        value = payload()
        del value["score"]
        report = gateway.validate_batch([value], source_text="省级二等奖加8分", context=context)
        assert report.items[0].status == "rejected"
        assert issue_codes(report) == {"schema_error"}

    def test_layer_1_rejects_extra_field_and_string_score(self, gateway, context) -> None:
        report = gateway.validate_batch(
            [payload(score="8", invented=True)], source_text="省级二等奖加8分", context=context
        )
        assert issue_codes(report) == {"schema_error"}

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"score": 10.0}, "score_not_in_quote"),
            ({"level": "省级一等奖"}, "level_not_in_quote"),
        ],
    )
    def test_layer_2_rejects_content_not_supported_by_quote(
        self, gateway, context, overrides, expected
    ) -> None:
        report = gateway.validate_batch(
            [payload(**overrides)], source_text="省级二等奖加8分", context=context
        )
        assert expected in issue_codes(report)
        assert report.items[0].status == "rejected"
        assert report.items[0].extract_confidence == 0

    def test_layer_2_accepts_chinese_score_and_normalized_level(self, gateway, context) -> None:
        text = "省二等奖加八分"
        report = gateway.validate_batch(
            [payload(level="省级二等奖", score=8.0, evidence_quote=text)],
            source_text=text,
            context=context,
        )
        assert report.publishable

    def test_layer_2_does_not_treat_clause_number_as_score(self, gateway, context) -> None:
        quote = "第8条规定省级二等奖加5分"
        report = gateway.validate_batch(
            [payload(score=8.0, evidence_quote=quote)], source_text=quote, context=context
        )
        assert "score_not_in_quote" in issue_codes(report)

    def test_layer_3_locates_exact_character_offsets(self, gateway, context) -> None:
        text = "第一条：省级二等奖加8分；本条结束。"
        quote = "省级二等奖加8分"
        report = gateway.validate_batch(
            [payload(evidence_quote=quote)], source_text=text, context=context
        )
        item = report.items[0]
        assert item.char_start == text.index(quote)
        assert item.char_end == item.char_start + len(quote)
        assert item.to_rule(context).source.char_start == text.index(quote)

    def test_layer_3_unlocatable_quote_requires_review(self, gateway, context) -> None:
        report = gateway.validate_batch(
            [payload()], source_text="原文只有省级二等奖，但没有分值", context=context
        )
        assert "quote_not_found" in issue_codes(report)
        assert report.items[0].status == "review"

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"score": 101.0, "evidence_quote": "省级二等奖加101分"}, "score_out_of_range"),
            ({"level": "火星级奖项", "evidence_quote": "火星级奖项加8分"}, "invalid_level"),
            ({"cap": 101.0}, "invalid_cap"),
            ({"team_factor": 1.5}, "invalid_team_factor"),
            ({"effective_date": "2025/10/01"}, "invalid_effective_date"),
            ({"effective_date": "2024-10-01"}, "effective_date_out_of_year"),
        ],
    )
    def test_layer_4_validates_boundaries(self, gateway, context, overrides, expected) -> None:
        report = gateway.validate_batch(
            [payload(**overrides)], source_text=payload(**overrides)["evidence_quote"], context=context
        )
        assert expected in issue_codes(report)
        assert not report.publishable

    def test_layer_4_validates_academic_year(self, gateway) -> None:
        context = GatewayContext(academic_year="本学年", doc="合成细则.pdf")
        report = gateway.validate_batch(
            [payload()], source_text="省级二等奖加8分", context=context
        )
        assert "invalid_academic_year" in issue_codes(report)

    def test_layer_4_allows_explicit_custom_level(self, gateway) -> None:
        context = GatewayContext(
            academic_year="2025-2026", doc="合成细则.pdf", allowed_levels=frozenset({"A类成果"})
        )
        report = gateway.validate_batch(
            [payload(level="A类成果", evidence_quote="A类成果加8分")],
            source_text="A类成果加8分",
            context=context,
        )
        assert report.publishable

    def test_layer_5_accepts_order_independent_equal_extraction(self, gateway, context) -> None:
        first = payload()
        second = payload(level="省二等奖")
        report = gateway.validate_batch(
            [first],
            source_text="省级二等奖加8分",
            context=context,
            secondary_payloads=[second],
        )
        assert report.publishable

    @pytest.mark.parametrize(
        ("secondary", "expected"),
        [([], "secondary_missing"), ([payload(score=10.0)], "secondary_mismatch")],
    )
    def test_layer_5_marks_independent_extraction_difference(
        self, gateway, context, secondary, expected
    ) -> None:
        report = gateway.validate_batch(
            [payload()],
            source_text="省级二等奖加8分",
            context=context,
            secondary_payloads=secondary,
        )
        assert expected in issue_codes(report)
        assert report.items[0].ambiguity_flag
        assert report.items[0].status == "review"

    def test_publication_gate_blocks_internal_conflict(self, gateway, context) -> None:
        text = "省级二等奖加8分；另一表写省级二等奖加10分"
        report = gateway.validate_batch(
            [
                payload(evidence_quote="省级二等奖加8分"),
                payload(score=10.0, evidence_quote="省级二等奖加10分"),
            ],
            source_text=text,
            context=context,
        )
        assert all("rule_conflict" in {issue.code for issue in item.issues} for item in report.items)
        assert all(
            issue.stage == "publication" and issue.layer is None
            for item in report.items
            for issue in item.issues
            if issue.code == "rule_conflict"
        )
        assert not report.publishable

    def test_publication_gate_blocks_conflict_with_existing_rule(self, gateway, context) -> None:
        existing = make_rule(
            "省级二等奖", 10, college="计算机学院", academic_year="2025-2026"
        )
        report = gateway.validate_batch(
            [payload()],
            source_text="省级二等奖加8分",
            context=context,
            existing_rules=[existing],
        )
        assert "rule_conflict" in issue_codes(report)

    def test_issue_contract_rejects_sixth_layer(self) -> None:
        with pytest.raises(ValueError):
            GatewayIssue(
                layer=6,
                code="rule_conflict",
                message="冲突门禁不得伪装成第六层",
            )


class TestGatewayReportAndEvaluation:
    def test_hash_ignores_outer_line_whitespace(self) -> None:
        assert chunk_hash(" 第一条 \n\n 第二条 ") == chunk_hash("第一条\n第二条")

    def test_metrics_expose_sample_size_and_reason_distribution(self, gateway, context) -> None:
        report = gateway.validate_batch(
            [payload(), payload(score=9.0)], source_text="省级二等奖加8分", context=context
        )
        metrics = report.metrics()
        assert metrics["sample_size"] == 2
        assert metrics["intercepted"] == 2  # 两条相同规则的分值冲突，安全地阻止整个批次
        assert metrics["reasons"]["score_not_in_quote"] == 1
        assert metrics["reasons"]["rule_conflict"] == 2
        assert metrics["validation_reasons"] == {"score_not_in_quote": 1}
        assert metrics["publication_gate_reasons"] == {"rule_conflict": 2}
        assert metrics["secondary_used"] is False

    def test_wilson_interval_for_30_of_30_discloses_uncertainty(self) -> None:
        lower, upper = wilson_interval(30, 30)
        assert lower == pytest.approx(0.8865, abs=0.001)
        assert upper == pytest.approx(1.0)

    def test_negative_case_rejects_code_from_another_target(self) -> None:
        with pytest.raises(ValueError):
            GatewayNegativeCase(
                id="bad-target",
                target="layer_3",
                expected_code="score_not_in_quote",
                source_text="省级二等奖加8分",
                payloads=[payload()],
            )

    def test_negative_evaluation_rejects_duplicate_case_ids(self, context) -> None:
        case = GatewayNegativeCase(
            id="duplicate",
            target="layer_2",
            expected_code="score_not_in_quote",
            source_text="省级二等奖加8分",
            payloads=[payload(score=10.0)],
        )
        with pytest.raises(ValueError, match="负例 id 重复"):
            evaluate_gateway_negatives(
                [case, case], context=context, dataset_version="duplicate-v1"
            )

    def test_frozen_negative_set_covers_all_five_layers_and_publication_gate(
        self, context
    ) -> None:
        fixture = Path(__file__).parent / "fixtures" / "gateway_negative_cases.json"
        raw = json.loads(fixture.read_text(encoding="utf-8"))
        cases = [GatewayNegativeCase.model_validate(item) for item in raw["cases"]]
        report = evaluate_gateway_negatives(
            cases, context=context, dataset_version=raw["dataset_version"]
        )
        assert report.sample_size == 100
        assert report.detected == 100
        assert report.detection_rate == 1.0
        assert report.missed_case_ids == []
        assert report.wilson_lower < report.detection_rate
        assert set(report.target_results) == {
            "layer_1",
            "layer_2",
            "layer_3",
            "layer_4",
            "layer_5",
            "publication_conflict",
        }
        assert all(result.detection_rate == 1.0 for result in report.target_results.values())
        assert len(report.config_hash) == 64


class TestGatewayStore:
    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        with RuleStore(tmp_path / "rules.sqlite") as store:
            values = [payload()]
            store.set_extraction_cache(chunk_hash="abc", model="m1", variant="direct", payloads=values)
            assert store.get_extraction_cache(
                chunk_hash="abc", model="m1", variant="direct"
            ) == values
            assert store.get_extraction_cache(
                chunk_hash="abc", model="m1", variant="clause_first"
            ) is None

    def test_publish_is_atomic_and_preserves_audit_and_offsets(
        self, tmp_path: Path, gateway, context
    ) -> None:
        report = gateway.validate_batch(
            [payload(rank="第二名", item_name="程序设计竞赛")],
            source_text="省级二等奖加8分",
            context=context,
        )
        with RuleStore(tmp_path / "rules.sqlite") as store:
            assert store.publish_extraction_report(report, model="mock-model") == 1
            rule = store.list_rules()[0]
            assert rule.rank == "第二名"
            assert rule.item_name == "程序设计竞赛"
            assert rule.source.char_start == 0 and rule.source.char_end == len("省级二等奖加8分")
            assert rule.source.chunk_hash == report.chunk_hash
            audit = store.list_extraction_audits(report.batch_id)
            assert len(audit) == 1 and audit[0]["status"] == "accepted"

    def test_publishing_same_batch_is_idempotent(self, tmp_path: Path, gateway, context) -> None:
        report = gateway.validate_batch(
            [payload()], source_text="省级二等奖加8分", context=context
        )
        with RuleStore(tmp_path / "rules.sqlite") as store:
            assert store.publish_extraction_report(report) == 1
            assert store.publish_extraction_report(report) == 1
            assert store.count() == 1

    def test_blocked_batch_is_audited_but_not_published(
        self, tmp_path: Path, gateway, context
    ) -> None:
        report = gateway.validate_batch(
            [payload(score=10.0)], source_text="省级二等奖加8分", context=context
        )
        with RuleStore(tmp_path / "rules.sqlite") as store:
            with pytest.raises(SchemaValidationError):
                store.publish_extraction_report(report)
            assert store.count() == 0
            audit = store.list_extraction_audits(report.batch_id)
            assert audit[0]["status"] == "rejected"

    def test_manual_correction_is_audited_and_does_not_bypass_gateway(
        self, tmp_path: Path, gateway, context
    ) -> None:
        report = gateway.validate_batch(
            [payload(score=10.0)], source_text="省级二等奖加8分", context=context
        )
        with RuleStore(tmp_path / "rules.sqlite") as store:
            store.record_extraction_report(report)
            assert store.record_extraction_correction(report.batch_id, 0, payload())
            assert store.count() == 0
            audit = store.list_extraction_audits(report.batch_id)[0]
            assert audit["manual_corrected"] is True
            assert audit["corrected_payload"] == payload()
            metrics = store.extraction_audit_metrics(report.batch_id)
            assert metrics["sample_size"] == 1
            assert metrics["manual_correction_rate"] == 1.0
            assert metrics["reasons"] == {"score_not_in_quote": 1}

    def test_store_rechecks_conflict_at_publish_time(
        self, tmp_path: Path, gateway, context
    ) -> None:
        report = gateway.validate_batch(
            [payload()], source_text="省级二等奖加8分", context=context
        )
        with RuleStore(tmp_path / "rules.sqlite") as store:
            store.upsert_rules(
                [make_rule("省级二等奖", 10, college="计算机学院", rule_id="existing")]
            )
            with pytest.raises(VersionConflict):
                store.publish_extraction_report(report)
            assert store.count() == 1

    def test_old_database_receives_non_destructive_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "old.sqlite"
        connection = sqlite3.connect(path)
        connection.execute(
            """CREATE TABLE rules (
                id TEXT PRIMARY KEY, academic_year TEXT NOT NULL, college TEXT,
                category TEXT NOT NULL, level TEXT NOT NULL, score REAL NOT NULL,
                synonyms TEXT NOT NULL DEFAULT '[]', constraints TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT '{}', priority INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1, raw_text TEXT, created_at TEXT
            )"""
        )
        connection.commit()
        connection.close()
        with RuleStore(path) as store:
            columns = {row[1] for row in store._conn.execute("PRAGMA table_info(rules)")}
        assert {"rank", "item_name", "extraction_batch_id"} <= columns


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.responses.pop(0))


class TestLLMExtractor:
    def test_cli_accepts_repeatable_document_allowed_levels(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        source = tmp_path / "rules.txt"
        source.write_text("第一专利人加5分", encoding="utf-8")
        captured: dict = {}

        class StubExtractor:
            model = "mock-model"

            def __init__(self, *, cache) -> None:
                captured["cache"] = cache

            def extract_validated(self, text, *, context, **kwargs):
                captured["context"] = context
                return ExtractionGateway().validate_batch(
                    [
                        payload(
                            category="知识产权",
                            level="第一专利人",
                            score=5.0,
                            evidence_quote=text,
                        )
                    ],
                    source_text=text,
                    context=context,
                )

        monkeypatch.setattr("scoreproof.cli.LLMExtractor", StubExtractor)
        result = CliRunner().invoke(
            app,
            [
                "extract-rules-llm",
                str(source),
                "--year",
                "2025-2026",
                "--allowed-level",
                "第一专利人",
                "--db",
                str(tmp_path / "rules.sqlite"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["context"].allowed_levels == frozenset({"第一专利人"})

    def test_real_adapter_parses_fenced_json_and_uses_cache(self, tmp_path: Path) -> None:
        raw = "```json\n" + json.dumps({"rules": [payload()]}, ensure_ascii=False) + "\n```"
        client = FakeClient([raw])
        with RuleStore(tmp_path / "rules.sqlite") as store:
            extractor = LLMExtractor(model="mock-model", client=client, cache=store)
            first = extractor.extract_payloads("省级二等奖加8分")
            second = extractor.extract_payloads("省级二等奖加8分")
        assert first == second == [payload()]
        assert client.calls == 1

    def test_invalid_json_fails_explicitly(self) -> None:
        extractor = LLMExtractor(client=FakeClient(["not-json"]))
        with pytest.raises(SchemaValidationError):
            extractor.extract_payloads("省级二等奖加8分")

    def test_high_risk_extraction_runs_two_prompt_variants(self, context) -> None:
        encoded = json.dumps([payload()], ensure_ascii=False)
        client = FakeClient([encoded, encoded])
        extractor = LLMExtractor(model="mock-model", client=client)
        report = extractor.extract_validated(
            "省级二等奖加8分", context=context, risk_level="high"
        )
        assert report.publishable
        assert client.calls == 2
        assert report.secondary_used is True

    def test_normal_risk_extraction_calls_model_once(self, context) -> None:
        client = FakeClient([json.dumps([payload()], ensure_ascii=False)])
        extractor = LLMExtractor(model="mock-model", client=client)
        report = extractor.extract_validated("省级二等奖加8分", context=context)
        assert report.publishable
        assert client.calls == 1
        assert report.secondary_used is False

    def test_low_confidence_primary_automatically_runs_second_extraction(self, context) -> None:
        unlocatable = payload(evidence_quote="省级二等奖按规定加8分")
        encoded = json.dumps([unlocatable], ensure_ascii=False)
        client = FakeClient([encoded, encoded])
        extractor = LLMExtractor(model="mock-model", client=client)
        report = extractor.extract_validated("省级二等奖加8分", context=context)
        assert report.secondary_used is True
        assert client.calls == 2
        assert report.items[0].status == "review"
