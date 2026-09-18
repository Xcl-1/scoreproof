"""阶段 7.2：统一 token/成本账本、汇总与真实 CLI 入口。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.observability import CostEvent, CostLedger, ModelPricing
from scoreproof.observability.costs import anonymize_subject, extract_usage
from scoreproof.rules.extractor import LLMExtractor
from scoreproof.rules.store import RuleStore


def _response(input_tokens: int = 100, output_tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        content='{"rules": []}',
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _response()


def test_cost_event_is_strict_and_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValidationError):
        CostEvent.model_validate(
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "purpose": "test",
                "input_tokens": 5,
                "output_tokens": 4,
                "total_tokens": 8,
                "usage_source": "usage_metadata",
                "unexpected": True,
            }
        )


def test_extract_usage_supports_langchain_and_openai_metadata() -> None:
    assert extract_usage(_response()) == (100, 20, 120, "usage_metadata")
    response = SimpleNamespace(
        response_metadata={
            "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        }
    )
    assert extract_usage(response) == (7, 3, 10, "response_metadata")


def test_ledger_records_priced_usage_without_raw_subject(tmp_path) -> None:
    db = tmp_path / "cost.sqlite"
    with CostLedger(db, subject_salt="test-only-salt") as ledger:
        event = ledger.record_response(
            _response(),
            provider="deepseek",
            model="deepseek-v4-pro",
            model_tier="strong",
            purpose="rule_extraction_primary",
            material_id="document:abc",
            batch_id="batch-1",
            subject_id="2023001",
            pricing=ModelPricing(
                input_cny_per_million=2.0,
                output_cny_per_million=4.0,
                version="invoice-2026-09",
            ),
        )
        loaded = ledger.list_events()[0]
    assert event.monetary_cost_cny == pytest.approx(0.00028)
    assert loaded.subject_id_hash == anonymize_subject("2023001", salt="test-only-salt")
    assert "2023001" not in db.read_bytes().decode("latin1")


def test_missing_usage_is_not_fabricated_as_zero(tmp_path) -> None:
    with CostLedger(tmp_path / "cost.sqlite") as ledger:
        event = ledger.record_response(
            SimpleNamespace(content="ok"),
            provider="deepseek",
            model="mock",
            purpose="answer_generation",
            subject_id="must-not-be-stored-without-hmac-salt",
        )
        report = ledger.report()
    assert event.usage_source == "missing"
    assert event.total_tokens is None
    assert event.subject_id_hash is None
    assert report.usage_covered_calls == 0
    assert report.usage_coverage_rate == 0
    assert any("token usage" in item for item in report.limitations)


def test_report_aggregates_cache_failures_tiers_and_unit_costs(tmp_path) -> None:
    pricing = ModelPricing(
        input_cny_per_million=2.0,
        output_cny_per_million=4.0,
        version="invoice-2026-09",
    )
    with CostLedger(tmp_path / "cost.sqlite", subject_salt="test-only-salt") as ledger:
        ledger.record_response(
            _response(100, 20),
            provider="deepseek",
            model="deepseek-v4-pro",
            model_tier="strong",
            purpose="rule_extraction_primary",
            material_id="document:1",
            subject_id="student-1",
            pricing=pricing,
        )
        ledger.record_response(
            _response(50, 10),
            provider="deepseek",
            model="deepseek-v4-pro",
            model_tier="strong",
            purpose="rule_extraction_secondary",
            material_id="document:1",
            subject_id="student-1",
            pricing=pricing,
        )
        ledger.record_cache_hit(
            provider="deepseek",
            model="deepseek-v4-pro",
            model_tier="strong",
            purpose="rule_extraction_primary",
            material_id="document:2",
            subject_id="student-2",
        )
        ledger.record_failure(
            provider="deepseek",
            model="deepseek-v4-pro",
            model_tier="strong",
            purpose="answer_generation",
            error=TimeoutError(),
            material_id="document:2",
            subject_id="student-2",
        )
        report = ledger.report()
    assert report.logical_requests == 4
    assert report.external_calls == 3 and report.failed_calls == 1
    assert report.cache_hit_rate == 0.25
    assert report.total_tokens == 180 and report.usage_coverage_rate == 1
    assert report.monetary_cost_available is True
    assert report.monetary_cost_cny == pytest.approx(0.00042)
    assert report.cost_per_100_materials_cny == pytest.approx(0.021)
    assert report.cost_per_subject_cny == pytest.approx(0.00021)
    assert report.strong_model_ratio == 1
    assert report.secondary_extraction_ratio == 0.5


def test_rule_extractor_records_external_call_then_cache_hit(tmp_path) -> None:
    db = tmp_path / "rules.sqlite"
    client = _Client()
    with RuleStore(db) as store, CostLedger(db) as ledger:
        extractor = LLMExtractor(
            model="deepseek-v4-flash",
            client=client,
            cache=store,
            cost_ledger=ledger,
            material_id="document:test",
            batch_id="batch-test",
        )
        assert extractor.extract_payloads("没有可抽取规则") == []
        assert extractor.extract_payloads("没有可抽取规则") == []
        report = ledger.report()
    assert client.calls == 1
    assert report.external_calls == 1 and report.cache_hits == 1
    assert report.total_tokens == 120


def test_cost_report_cli_reads_real_sqlite(tmp_path) -> None:
    db = tmp_path / "cost.sqlite"
    output = tmp_path / "cost-report.json"
    with CostLedger(db) as ledger:
        ledger.record_response(
            _response(9, 1),
            provider="deepseek",
            model="deepseek-v4-flash",
            purpose="certificate_text_extraction",
            material_id="image:abc",
        )
    result = CliRunner().invoke(app, ["cost-report", "--db", str(db), "--out", str(output)])
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["external_calls"] == 1
    assert payload["total_tokens"] == 10
    assert payload["monetary_cost_available"] is False


def test_failure_event_stores_only_error_type(tmp_path) -> None:
    with CostLedger(tmp_path / "cost.sqlite") as ledger:
        event = ledger.record_failure(
            provider="deepseek",
            model="deepseek-v4-flash",
            purpose="rule_extraction_primary",
            error=RuntimeError("sensitive prompt content"),
        )
    assert event.error_type == "RuntimeError"
    assert "sensitive prompt content" not in (tmp_path / "cost.sqlite").read_bytes().decode("latin1")
