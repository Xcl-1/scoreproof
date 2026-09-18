"""阶段 7：候选冻结、质量命令与统一评测门禁。"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import scoreproof.eval.readiness as readiness_module
from scoreproof.cli import app
from scoreproof.eval.readiness import (
    QualityCommandResult,
    QualityGateReport,
    build_release_readiness,
    run_quality_gates,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _quality(*, clean: bool = True) -> dict:
    return QualityGateReport(
        generated_at="2026-09-15T00:00:00+00:00",
        git_head="abc123",
        source_tree_clean=clean,
        pytest_total=382,
        all_checks_passed=True,
        commands=[],
    ).model_dump(mode="json")


def _copy_existing_reports(target: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "reports"
    for filename in (
        "gateway-negative-v2.json",
        "retrieval-ablation-v1.json",
        "citation-refusal-v1.json",
        "orchestration-guardrails-v1.json",
        "certificate-fields-smoke-v1.json",
        "evidence-dedup-smoke-v1.json",
        "cost-summary-v1.json",
    ):
        (target / filename).write_bytes((source / filename).read_bytes())


class TestQualityGates:
    def test_runs_fixed_commands_and_reads_git_state(self, tmp_path: Path, monkeypatch) -> None:
        def fake_run(name: str, command: list[str], **_: object) -> QualityCommandResult:
            output = "382 passed in 1.0s" if name == "pytest" else "passed"
            return QualityCommandResult(
                name=name,
                command=command,
                exit_code=0,
                duration_seconds=0.1,
                passed=True,
                output_tail=output,
            )

        def fake_git(_: Path, *args: str) -> str:
            return "abc123" if args[0] == "rev-parse" else ""

        monkeypatch.setattr(readiness_module, "_run_command", fake_run)
        monkeypatch.setattr(readiness_module, "_git_output", fake_git)
        report = run_quality_gates(tmp_path)
        assert report.pytest_total == 382
        assert report.all_checks_passed is True
        assert report.source_tree_clean is True
        assert [item.name for item in report.commands] == [
            "pytest",
            "ruff",
            "mypy",
            "uv_lock",
            "git_diff_check",
        ]


class TestReleaseReadiness:
    def test_existing_formal_reports_pass_but_smoke_never_passes_formal_gates(
        self, tmp_path: Path
    ) -> None:
        _copy_existing_reports(tmp_path)
        _write(tmp_path / "quality-gates-v1.json", _quality())

        report = build_release_readiness(
            tmp_path,
            candidate_version="abc123",
            generated_at="2026-09-15T00:00:00+00:00",
        )

        statuses = {gate.id: gate.status for gate in report.gates}
        assert statuses["quality"] == "通过"
        assert statuses["gateway"] == "通过"
        assert statuses["retrieval_ablation"] == "通过"
        assert statuses["citation_refusal"] == "通过"
        assert statuses["orchestration"] == "通过"
        assert statuses["certificate_fields"] == "仅烟雾"
        assert statuses["evidence_dedup"] == "仅烟雾"
        assert report.ready is False
        assert report.status == "阻塞"
        assert "backtest_52" in report.blocking_gate_ids
        assert "cost_observability" in report.warning_gate_ids
        assert report.cost_summary.total_tokens == 1232
        assert report.cost_summary.usage_coverage_rate == 1.0
        assert report.cost_summary.monetary_cost_available is False

    def test_dirty_worktree_blocks_candidate_freeze(self, tmp_path: Path) -> None:
        _write(tmp_path / "quality-gates-v1.json", _quality(clean=False))
        report = build_release_readiness(tmp_path, candidate_version="working-tree")
        gate = next(item for item in report.gates if item.id == "quality")
        assert gate.status == "阻塞"
        assert any("工作树不干净" in reason for reason in gate.reasons)

    def test_zero_vlm_trigger_is_valid_for_formal_field_threshold(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "certificate-fields-formal-v1.json",
            {
                "sample_size": 30,
                "formal_gate_eligible": True,
                "micro": {"f1": 0.91},
                "vlm_trigger_rate": 0.0,
            },
        )
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        gate = next(item for item in report.gates if item.id == "certificate_fields")
        assert gate.status == "通过"

    def test_formal_dedup_requires_intervals_and_both_targets(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "evidence-dedup-formal-v1.json",
            {
                "sample_size": 50,
                "formal_gate_eligible": True,
                "recall": 0.96,
                "precision": 0.95,
                "recall_ci95": [0.8, 0.99],
                "precision_ci95": [0.8, 0.99],
            },
        )
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        gate = next(item for item in report.gates if item.id == "evidence_dedup")
        assert gate.status == "通过"

    def test_invalid_json_is_snapshotted_and_cannot_pass(self, tmp_path: Path) -> None:
        (tmp_path / "gateway-negative-v2.json").write_text("not-json", encoding="utf-8")
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        artifact = next(item for item in report.artifacts if item.name == "gateway")
        assert artifact.exists is True
        assert artifact.valid_json is False
        assert next(item for item in report.gates if item.id == "gateway").status == "缺失"

    def test_manifest_hash_is_stable_for_same_candidate_and_artifacts(self, tmp_path: Path) -> None:
        _write(tmp_path / "quality-gates-v1.json", _quality())
        first = build_release_readiness(tmp_path, candidate_version="abc", generated_at="one")
        second = build_release_readiness(tmp_path, candidate_version="abc", generated_at="two")
        assert first.manifest_hash == second.manifest_hash

    def test_cli_writes_blocked_report_and_enforces_exit_code(self, tmp_path: Path) -> None:
        output = tmp_path / "readiness.json"
        relaxed = CliRunner().invoke(
            app,
            [
                "release-readiness",
                "--candidate-version",
                "abc",
                "--report-dir",
                str(tmp_path),
                "--out",
                str(output),
                "--no-enforce",
            ],
        )
        assert relaxed.exit_code == 0, relaxed.output
        assert json.loads(output.read_text(encoding="utf-8"))["ready"] is False
        enforced = CliRunner().invoke(
            app,
            [
                "release-readiness",
                "--candidate-version",
                "abc",
                "--report-dir",
                str(tmp_path),
                "--out",
                str(output),
            ],
        )
        assert enforced.exit_code == 2
