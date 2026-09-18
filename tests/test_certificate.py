"""奖状字段结构化、校验、置信度、VLM 决策与字段评测。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("cv2")
pytest.importorskip("imagehash")

from typer.testing import CliRunner  # noqa: E402

from scoreproof.cli import app  # noqa: E402
from scoreproof.errors import DataSourceError, SchemaValidationError  # noqa: E402
from scoreproof.eval.certificate import evaluate_certificate_fields  # noqa: E402
from scoreproof.evidence.certificate import (  # noqa: E402
    CertificateDraft,
    CertificateField,
    CertificateFieldSet,
    CertificateTextExtractor,
    FieldLocation,
    FieldSignals,
    VlmRegion,
    decide_vlm_fallback,
    extract_certificate,
    extract_certificate_fields,
    extract_with_vlm,
)
from scoreproof.ingest.image_loader import OcrLine, OcrResult  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = json.dumps(payload, ensure_ascii=False)


class FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.messages: Any = None

    def invoke(self, messages: Any) -> FakeResponse:
        self.messages = messages
        return FakeResponse(self.payload)


class BrokenClient:
    def invoke(self, messages: Any) -> FakeResponse:
        raise RuntimeError("secret upstream detail")


def _ocr() -> OcrResult:
    values = [
        ("荣誉证书", 0.99),
        ("学生001", 0.96),
        ("在全国大学生学科竞赛中荣获", 0.94),
        ("国家级一等奖", 0.92),
        ("团队项目", 0.85),
        ("全国大学生学科竞赛组织委员会", 0.90),
        ("2025年4月2日", 0.88),
    ]
    return OcrResult(
        lines=[
            OcrLine(text=text, bbox=(0.0, float(i * 20), 300.0, float(i * 20 + 15)), confidence=conf)
            for i, (text, conf) in enumerate(values)
        ]
    )


def _field(raw: str | None, evidence: str | None, indexes: list[int]) -> dict[str, Any]:
    return {"raw_value": raw, "evidence_text": evidence, "line_indexes": indexes}


def _payload() -> dict[str, Any]:
    return {
        "姓名": _field("学生001", "学生001", [1]),
        "赛事名称": _field("全国大学生学科竞赛", "在全国大学生学科竞赛中荣获", [2]),
        "级别": _field("国家级一等奖", "国家级一等奖", [3]),
        "奖项/名次": _field("国家级一等奖", "国家级一等奖", [3]),
        "获奖日期": _field("2025年4月2日", "2025年4月2日", [6]),
        "颁发单位": _field(
            "全国大学生学科竞赛组织委员会", "全国大学生学科竞赛组织委员会", [5]
        ),
        "团队属性": _field("团队项目", "团队项目", [4]),
    }


class TestStrictDraft:
    def test_extra_field_is_forbidden(self) -> None:
        payload = _payload()
        payload["姓名"] = {**payload["姓名"], "confidence": 0.99}
        with pytest.raises(ValidationError):
            CertificateDraft.model_validate(payload)

    def test_model_response_extra_field_is_domain_error(self) -> None:
        payload = _payload()
        payload["unexpected"] = "nope"
        with pytest.raises(SchemaValidationError, match="严格 schema"):
            CertificateTextExtractor(client=FakeClient(payload)).extract(_ocr())

    def test_upstream_failure_is_sanitized_domain_error(self) -> None:
        with pytest.raises(DataSourceError) as caught:
            CertificateTextExtractor(client=BrokenClient()).extract(_ocr())
        assert caught.value.detail["error_type"] == "RuntimeError"
        assert "secret upstream detail" not in str(caught.value.detail)


class TestFieldValidation:
    def test_normalizes_all_supported_fields_and_keeps_locations(self) -> None:
        result = extract_certificate_fields(
            _ocr(), extractor=CertificateTextExtractor(client=FakeClient(_payload()))
        )
        fields = result.fields
        assert fields.name.normalized_value == "学生001"
        assert fields.level.normalized_value == "国家级"
        assert fields.award.normalized_value == "一等奖"
        assert fields.award_date.normalized_value == "2025-04-02"
        assert fields.team.normalized_value == "团队"
        assert fields.level.location.line_indexes == [3]
        assert fields.level.location.bboxes == [(0.0, 60.0, 300.0, 75.0)]
        assert fields.level.location.char_start is not None
        assert fields.level.signals.dictionary_match is True
        assert fields.level.signals.multi_source_agreement is True
        assert fields.level.confidence != 1.0  # 来自 OCR 行置信度等多信号，不采用模型分数
        assert result.manual_review_required is False
        assert result.manual_review_status == "not_required"

    def test_unsupported_name_is_cleared_not_guessed(self) -> None:
        payload = _payload()
        payload["姓名"] = _field("张三", "学生001", [1])
        result = extract_certificate_fields(
            _ocr(), extractor=CertificateTextExtractor(client=FakeClient(payload))
        )
        assert result.fields.name.raw_value is None
        assert result.fields.name.normalized_value is None
        assert "unsupported_by_ocr" in result.fields.name.validation_errors
        assert result.vlm.requested is True
        assert result.vlm.status == "provider_missing"
        assert result.manual_review_required is True
        assert result.manual_review_status == "pending_manual_review"

    def test_invalid_date_and_unknown_award_are_cleared(self) -> None:
        ocr = _ocr()
        ocr.lines[6].text = "2025年2月30日"
        ocr.lines[3].text = "国家级至尊奖"
        payload = _payload()
        payload["获奖日期"] = _field("2025年2月30日", "2025年2月30日", [6])
        payload["级别"] = _field("国家级至尊奖", "国家级至尊奖", [3])
        payload["奖项/名次"] = _field("至尊奖", "国家级至尊奖", [3])
        result = extract_certificate_fields(
            ocr, extractor=CertificateTextExtractor(client=FakeClient(payload))
        )
        assert result.fields.award_date.normalized_value is None
        assert result.fields.award.normalized_value is None
        assert result.fields.level.normalized_value == "国家级"

    def test_multi_source_disagreement_forces_low_confidence(self) -> None:
        payload = _payload()
        payload["赛事名称"] = _field("大学生学科竞赛", "在全国大学生学科竞赛中荣获", [2])
        result = extract_certificate_fields(
            _ocr(), extractor=CertificateTextExtractor(client=FakeClient(payload))
        )
        assert result.fields.event_name.normalized_value == "大学生学科竞赛"
        assert result.fields.event_name.confidence == 0.55
        assert "multi_source_disagreement" in result.fields.event_name.validation_errors
        assert "赛事名称" in result.vlm.low_confidence_fields

    def test_model_cannot_report_its_own_confidence(self) -> None:
        prompt = CertificateTextExtractor(client=FakeClient(_payload())).build_prompt(_ocr())
        schema = json.loads(prompt[1]["content"])["schema"]
        field_schema = schema["$defs"]["CertificateFieldDraft"]
        assert "confidence" not in field_schema["properties"]
        assert field_schema["additionalProperties"] is False


def _low_fields(*, with_bbox: bool = True) -> CertificateFieldSet:
    good_signals = FieldSignals(
        ocr_line_confidence=1.0,
        source_supported=True,
        format_valid=True,
        dictionary_match=None,
        multi_source_agreement=None,
    )
    missing_signals = FieldSignals(
        ocr_line_confidence=0.0,
        source_supported=False,
        format_valid=False,
        dictionary_match=True,
        multi_source_agreement=None,
    )
    good = CertificateField(
        raw_value="有值",
        normalized_value="有值",
        evidence_text="有值",
        confidence=1.0,
        signals=good_signals,
    )
    low = CertificateField(
        raw_value=None,
        normalized_value=None,
        evidence_text=None,
        confidence=0.0,
        location=FieldLocation(bboxes=[(1.0, 2.0, 3.0, 4.0)] if with_bbox else []),
        signals=missing_signals,
    )
    return CertificateFieldSet(
        姓名=good,
        赛事名称=good,
        级别=good,
        **{"奖项/名次": low},
        获奖日期=good,
        颁发单位=good,
        团队属性=good,
    )


class TestVlmDecision:
    def test_no_provider_routes_to_manual_review(self) -> None:
        decision = decide_vlm_fallback(_low_fields(), provider="none")
        assert decision.requested is True and decision.called is False
        assert decision.status == "provider_missing"
        assert decision.low_confidence_fields == ["奖项/名次"]
        assert decision.eligible_fields == ["奖项/名次"]
        assert decision.regions[0].field == "奖项/名次"

    def test_provider_without_key_routes_to_manual_review(self) -> None:
        decision = decide_vlm_fallback(
            _low_fields(), provider="qwen-vl-plus", dashscope_configured=False
        )
        assert decision.status == "key_missing"
        assert any("DASHSCOPE_API_KEY" in reason for reason in decision.reasons)

    def test_missing_crop_forbids_full_image_fallback(self) -> None:
        decision = decide_vlm_fallback(
            _low_fields(with_bbox=False),
            provider="glm-4v",
            zhipu_configured=True,
        )
        assert decision.status == "crop_missing"
        assert decision.regions == []

    def test_only_low_fields_and_regions_reach_injected_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "cert.png"
        Image.new("RGB", (20, 20), "white").save(source)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
        client = _VlmClient()
        result = extract_with_vlm(
            source,
            fields=["奖项/名次"],
            regions=[VlmRegion(field="奖项/名次", bbox=(1, 2, 3, 4), reason="low")],
            provider="qwen-vl-plus",
            client=client,
            out_dir=tmp_path / "crops",
        )
        assert result == {"奖项/名次": "一等奖"}
        assert client.payload["fields"] == ["奖项/名次"]
        assert "image_path" not in client.payload
        assert client.payload["crops"][0]["field"] == "奖项/名次"
        assert Path(client.payload["crops"][0]["crop_path"]).exists()


class _VlmClient:
    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, str]:
        self.payload = payload
        return {"奖项/名次": "一等奖"}


class TestPipelineAndEvaluation:
    def test_real_file_pipeline_and_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "certificate.png"
        Image.new("RGB", (800, 500), "white").save(source)
        fake_extractor = CertificateTextExtractor(client=FakeClient(_payload()))
        direct = extract_certificate(
            source, run_preprocess=False, extractor=fake_extractor, ocr_result=_ocr()
        )
        assert direct.evidence.fields["级别"] == "国家级"
        assert direct.evidence.extra["field_details"]["获奖日期"]["raw_value"] == "2025年4月2日"

        import scoreproof.evidence.certificate as module

        monkeypatch.setattr(module, "run_ocr", lambda _: _ocr())
        monkeypatch.setattr(
            CertificateTextExtractor,
            "_client_for",
            lambda self: FakeClient(_payload()),
        )
        monkeypatch.setattr(CertificateTextExtractor, "available", lambda self: True)
        output = tmp_path / "result.json"
        invoked = CliRunner().invoke(
            app, ["extract-certificate", str(source), "--raw", "--out", str(output)]
        )
        assert invoked.exit_code == 0, invoked.output
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["extraction"]["fields"]["姓名"]["normalized_value"] == "学生001"
        assert saved["extraction"]["vlm"]["called"] is False

    def test_evaluation_reports_smoke_only_and_all_metrics(self) -> None:
        labels = [
            {
                "evidence_id": "E1",
                "name": "学生001",
                "event_name": "全国大学生学科竞赛",
                "level": "国家级一等奖",
                "award_date": "2025-04-02",
                "issuer": "全国大学生学科竞赛组织委员会",
                "is_team": True,
                "synthetic": True,
            }
        ]
        fields = {
            name: {"normalized_value": value}
            for name, value in {
                "姓名": "学生001",
                "赛事名称": "全国大学生学科竞赛",
                "级别": "国家级",
                "奖项/名次": "一等奖",
                "获奖日期": "2025-04-02",
                "颁发单位": "全国大学生学科竞赛组织委员会",
                "团队属性": "团队",
            }.items()
        }
        predictions = [
            {
                "evidence_id": "E1",
                "extraction": {
                    "fields": fields,
                    "vlm": {"requested": False, "called": False},
                },
            }
        ]
        report = evaluate_certificate_fields(labels, predictions, dataset_version="synthetic-v1")
        assert report.micro.f1 == 1.0
        assert report.micro.precision_ci95[1] == pytest.approx(1.0)
        assert report.micro.recall_ci95[1] == pytest.approx(1.0)
        assert report.per_field["姓名"].f1 == 1.0
        assert report.exact_certificate_rate == 1.0
        assert report.vlm_trigger_rate == 0.0
        assert report.vlm_call_ci95[0] == 0.0
        assert report.sample_size == 1
        assert report.smoke_test_only is True and report.formal_gate_eligible is False
        assert report.raw_micro is None and report.raw_labeled_values == 0

    def test_raw_and_normalized_values_are_evaluated_separately(self) -> None:
        label = {
            "evidence_id": "E1",
            "name": "张三",
            "event_name": "某竞赛",
            "level": "省级一等奖",
            "award_date": "2025-01-02",
            "issuer": "某委员会",
            "team_attribute": "团队",
            "raw_fields": {
                "姓名": "张三同学",
                "赛事名称": "某竞赛",
                "级别": "省级",
                "奖项/名次": "第一名",
                "获奖日期": "2025年1月2日",
                "颁发单位": "某委员会",
                "团队属性": "团队项目",
            },
        }
        normalized = _expected_prediction_values()
        normalized.update(
            {
                "姓名": "张三",
                "赛事名称": "某竞赛",
                "级别": "省级",
                "奖项/名次": "一等奖",
                "获奖日期": "2025-01-02",
                "颁发单位": "某委员会",
                "团队属性": "团队",
            }
        )
        raw = dict(label["raw_fields"])
        fields = {
            name: {"normalized_value": normalized[name], "raw_value": raw[name]}
            for name in normalized
        }
        report = evaluate_certificate_fields(
            [label],
            [{"evidence_id": "E1", "extraction": {"fields": fields, "vlm": {}}}],
        )
        assert report.micro.f1 == 1.0
        assert report.raw_micro is not None and report.raw_micro.f1 == 1.0
        assert report.raw_complete_samples == 1 and report.raw_exact_certificate_rate == 1.0


def _expected_prediction_values() -> dict[str, str]:
    return {name: "" for name in ("姓名", "赛事名称", "级别", "奖项/名次", "获奖日期", "颁发单位", "团队属性")}
