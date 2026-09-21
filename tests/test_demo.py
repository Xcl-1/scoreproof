"""阶段 7.4：一键演示报告与真实 CLI 子进程链路。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.eval.demo import DemoArtifact, DemoReport, DemoStep, run_smoke_demo


def test_demo_schema_forbids_extra_fields() -> None:
    payload = {
        "generated_at": "2026-09-20T00:00:00+00:00",
        "all_steps_passed": True,
        "business_checks_passed": True,
        "sample_students": 5,
        "sample_claims": 13,
        "imported_rules": 13,
        "steps": [],
        "artifacts": [],
        "limitations": [],
        "unreviewed_claim": "不应被接受",
    }
    with pytest.raises(ValidationError):
        DemoReport.model_validate(payload)


def test_demo_step_does_not_accept_captured_business_output() -> None:
    with pytest.raises(ValidationError):
        DemoStep.model_validate(
            {
                "id": "calculate_scores",
                "status": "passed",
                "exit_code": 0,
                "duration_seconds": 1.0,
                "output": "不应持久化命令输出",
            }
        )


def test_demo_artifact_rejects_traversal_and_invalid_hash() -> None:
    with pytest.raises(ValidationError):
        DemoArtifact(
            name="calculation",
            relative_path="../outside.json",
            size_bytes=1,
            sha256="not-a-sha256",
        )


def test_demo_refuses_to_overwrite_nonempty_directory(tmp_path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        run_smoke_demo(tmp_path, destination)
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_real_demo_cli_runs_files_database_calculation_and_backtest(tmp_path) -> None:
    destination = tmp_path / "demo"
    report_path = tmp_path / "demo-smoke.json"
    result = CliRunner().invoke(
        app,
        [
            "demo",
            "--out-dir",
            str(destination),
            "--report-out",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = DemoReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.all_steps_passed is True
    assert report.business_checks_passed is True
    assert report.sample_students == 5
    assert report.sample_claims == 13
    assert report.imported_rules == 13
    assert report.person_agreement == 1.0
    assert report.item_agreement == 1.0
    assert report.smoke_test_only is True
    assert report.formal_gate_eligible is False
    assert [step.id for step in report.steps] == [
        "generate_sample_files",
        "import_rules",
        "calculate_scores",
        "backtest_items",
        "explain_claim",
    ]
    assert all(step.status == "passed" for step in report.steps)
    assert all(len(artifact.sha256) == 64 for artifact in report.artifacts)
    assert json.loads((destination / "backtest.json").read_text(encoding="utf-8"))[
        "gate_passed"
    ] is True
