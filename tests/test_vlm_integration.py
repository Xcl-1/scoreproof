"""真实 VLM 必要裁剪适配器、报告与发布门禁。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

Image = pytest.importorskip("PIL.Image")

import scoreproof.evidence.certificate as certificate_module  # noqa: E402
from scoreproof.cli import app  # noqa: E402
from scoreproof.eval.readiness import build_release_readiness  # noqa: E402
from scoreproof.eval.vlm import VlmIntegrationReport, build_vlm_integration_report  # noqa: E402
from scoreproof.evidence.vlm import VlmProviderResponse  # noqa: E402


class _ProviderClient:
    def invoke(self, payload: dict[str, Any]) -> VlmProviderResponse:
        assert payload["fields"] == ["奖项/名次"]
        assert len(payload["crops"]) == 1
        assert "image_path" not in payload
        return VlmProviderResponse(
            values={"奖项/名次": "国家级一等奖"},
            usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        )


def test_cli_calls_injected_provider_with_crop_and_writes_strict_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "certificate.png"
    Image.new("RGB", (300, 160), "white").save(image)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(certificate_module, "make_vlm_client", lambda _: _ProviderClient())
    report_path = tmp_path / "vlm-report.json"
    result = CliRunner().invoke(
        app,
        [
            "eval-vlm-integration",
            str(image),
            "--field",
            "奖项/名次",
            "--bbox",
            "10,20,250,100",
            "--expected",
            "国家级一等奖",
            "--provider",
            "qwen-vl-plus",
            "--dataset-kind",
            "synthetic",
            "--crops-dir",
            str(tmp_path / "crops"),
            "--cost-db",
            str(tmp_path / "cost.sqlite"),
            "--out",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = VlmIntegrationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.result == "passed"
    assert report.real_external_service is True
    assert report.whole_image_sent is False
    assert report.cost_event_recorded is True
    assert report.smoke_test_only is True
    assert "国家级一等奖" not in report_path.read_text(encoding="utf-8")


def test_report_schema_rejects_extra_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    crop = tmp_path / "crop.png"
    Image.new("RGB", (20, 20), "white").save(source)
    Image.new("RGB", (10, 10), "white").save(crop)
    report = build_vlm_integration_report(
        provider="qwen-vl-plus",
        model="qwen-vl-plus",
        source=source,
        crop_paths=[crop],
        requested_fields=["级别"],
        response={"级别": "国家级"},
        dataset_kind="synthetic",
        expected={"级别": "国家级"},
        cost_event_recorded=True,
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
    )
    (tmp_path / "vlm-integration-v1.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    readiness = build_release_readiness(tmp_path, candidate_version="candidate")
    gate = next(item for item in readiness.gates if item.id == "vlm_integration")
    assert gate.status == "通过"
    assert gate.metrics["whole_image_sent"] is False
    payload = report.model_dump(mode="json")
    payload["raw_response"] = "forbidden"
    with pytest.raises(ValidationError):
        VlmIntegrationReport.model_validate(payload)


def test_readiness_requires_strict_vlm_report(tmp_path: Path) -> None:
    (tmp_path / "vlm-integration-v1.json").write_text(
        json.dumps({"result": "passed", "real_external_service": True, "sample_size": 1}),
        encoding="utf-8",
    )
    report = build_release_readiness(tmp_path, candidate_version="candidate")
    gate = next(item for item in report.gates if item.id == "vlm_integration")
    assert gate.status == "阻塞"
    assert "Schema 无效" in gate.reasons[0]
