"""规则结构化抽取完整正确率评测与正式门禁。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import scoreproof.cli as cli_module
from scoreproof.cli import app
from scoreproof.eval.rule_extraction import (
    RULE_EXTRACTION_FIELDS,
    RuleExtractionCase,
    RuleExtractionDataset,
    RuleExtractionReport,
    evaluate_rule_extraction,
)
from scoreproof.rules.gateway import ExtractionGateway, GatewayContext, GatewayReport


def _payload(
    *,
    level: str = "省级一等奖",
    score: float = 3.0,
    quote: str = "省级一等奖计3分。",
) -> dict[str, Any]:
    return {
        "category": "学科竞赛",
        "level": level,
        "score": score,
        "synonyms": [],
        "cap": None,
        "team_factor": None,
        "clause": None,
        "evidence_quote": quote,
        "rank": None,
        "item_name": None,
        "effective_date": None,
    }


def _case(
    case_id: str = "case-1",
    *,
    text: str = "省级一等奖计3分。",
    expected: list[dict[str, Any]] | None = None,
    real_source: bool = False,
    synthetic: bool = True,
) -> RuleExtractionCase:
    return RuleExtractionCase.model_validate(
        {
            "case_id": case_id,
            "source_id": f"source-{case_id}",
            "source_text": text,
            "academic_year": "2025-2026",
            "category_hint": "学科竞赛",
            "real_source": real_source,
            "synthetic": synthetic,
            "expected_rules": expected if expected is not None else [_payload(quote=text)],
        }
    )


class FakeExtractor:
    model = "fake-rule-model"

    def __init__(
        self,
        responses: dict[str, list[dict[str, Any]]] | None = None,
        *,
        fail: bool = False,
        **_: Any,
    ) -> None:
        self.responses = responses or {}
        self.fail = fail

    def available(self) -> bool:
        return False

    def extract_validated(
        self,
        text: str,
        *,
        context: GatewayContext,
        risk_level: str = "normal",
        existing_rules: tuple[()] = (),
        **_: Any,
    ) -> GatewayReport:
        if self.fail:
            raise RuntimeError("upstream unavailable")
        return ExtractionGateway().validate_batch(
            self.responses.get(text, [_payload(quote=text)]),
            source_text=text,
            context=context,
            secondary_payloads=(
                self.responses.get(text, [_payload(quote=text)])
                if risk_level == "high"
                else None
            ),
            existing_rules=existing_rules,
        )


def test_gold_labels_must_pass_the_same_code_gateway() -> None:
    with pytest.raises(ValidationError, match="人工金标未通过"):
        _case(expected=[_payload(score=9.0)])
    with pytest.raises(ValidationError, match="金标规则不得重复"):
        _case(expected=[_payload(), _payload()])
    with pytest.raises(ValidationError, match="相同原文块"):
        RuleExtractionDataset(
            dataset_version="duplicate-source",
            cases=[_case("duplicate-1"), _case("duplicate-2")],
        )


def test_exact_rule_and_each_field_are_scored() -> None:
    case = _case()
    dataset = RuleExtractionDataset(dataset_version="smoke", cases=[case])
    report = evaluate_rule_extraction(
        dataset,
        extractor=FakeExtractor(),
        provider="fake",
        real_external_service=False,
    )
    assert report.exact_rule_accuracy.value == 1.0
    assert report.field_micro_accuracy.value == 1.0
    assert report.field_micro_prf.f1 == 1.0
    assert report.field_prf["level"].precision == 1.0
    assert set(report.field_metrics) == set(RULE_EXTRACTION_FIELDS)
    assert report.results[0].complete_correct is True
    assert report.smoke_test_only is True
    assert report.formal_gate_eligible is False


def test_missing_or_extra_rules_are_penalized() -> None:
    first = "省级一等奖计3分。"
    second = "省级二等奖计2分。"
    text = first + second
    expected = [_payload(quote=first)]
    predictions = {
        text: [
            _payload(quote=first),
            _payload(level="省级二等奖", score=2.0, quote=second),
        ]
    }
    report = evaluate_rule_extraction(
        RuleExtractionDataset(dataset_version="extra", cases=[_case(text=text, expected=expected)]),
        extractor=FakeExtractor(predictions),
        provider="fake",
        real_external_service=False,
    )
    assert report.sample_size == 1
    assert report.scored_rule_units == 2
    assert report.exact_rule_accuracy.value == 0.5
    assert report.results[0].complete_correct is False


def test_extraction_failure_is_visible_and_cannot_be_formal() -> None:
    report = evaluate_rule_extraction(
        RuleExtractionDataset(dataset_version="failure", cases=[_case()]),
        extractor=FakeExtractor(fail=True),
        provider="fake",
        real_external_service=True,
    )
    assert report.real_external_service is False
    assert report.results[0].extraction_error == "RuntimeError: upstream unavailable"
    assert report.exact_rule_accuracy.value == 0.0
    assert any("调用失败" in item for item in report.limitations)


def test_fifty_real_gold_rules_can_become_formal() -> None:
    cases = [
        _case(
            f"formal-{index}",
            text=f"省级一等奖计3分。规则编号{index}。",
            expected=[_payload(quote=f"省级一等奖计3分。规则编号{index}。")],
            real_source=True,
            synthetic=False,
        )
        for index in range(50)
    ]
    report = evaluate_rule_extraction(
        RuleExtractionDataset(
            dataset_version="formal-shape",
            authorization_reference="internal://approval/1",
            independent_real_samples=True,
            cases=cases,
        ),
        extractor=FakeExtractor(),
        provider="deepseek",
        real_external_service=True,
    )
    assert report.sample_size == 50
    assert report.formal_gate_eligible is True
    assert report.passed is True
    assert report.exact_rule_accuracy.value == 1.0


def test_report_round_trips_under_strict_schema() -> None:
    report = evaluate_rule_extraction(
        RuleExtractionDataset(dataset_version="roundtrip", cases=[_case()]),
        extractor=FakeExtractor(),
        provider="fake",
        real_external_service=False,
    )
    decoded = RuleExtractionReport.model_validate_json(report.model_dump_json())
    assert decoded == report


def test_cli_writes_smoke_report_but_enforced_mode_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = RuleExtractionDataset(dataset_version="cli-smoke", cases=[_case()])
    source = tmp_path / "dataset.json"
    source.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr(cli_module, "LLMExtractor", FakeExtractor)
    output = tmp_path / "report.json"
    relaxed = CliRunner().invoke(
        app,
        ["eval-rule-extraction", str(source), "--out", str(output), "--no-enforce"],
    )
    assert relaxed.exit_code == 0, relaxed.output
    assert json.loads(output.read_text(encoding="utf-8"))["smoke_test_only"] is True
    enforced = CliRunner().invoke(
        app,
        ["eval-rule-extraction", str(source), "--out", str(output)],
    )
    assert enforced.exit_code == 2
