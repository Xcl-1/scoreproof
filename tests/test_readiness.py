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
        "complex-pdf-regression-v1.json",
        "rule-extraction-smoke-v1.json",
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
        assert statuses["complex_pdf"] == "仅烟雾"
        assert statuses["rule_extraction"] == "仅烟雾"
        assert statuses["certificate_fields"] == "仅烟雾"
        assert statuses["evidence_dedup"] == "仅烟雾"
        assert report.ready is False
        assert report.status == "阻塞"
        assert "backtest_52" in report.blocking_gate_ids
        assert "cost_observability" in report.warning_gate_ids
        stored_cost = json.loads((tmp_path / "cost-summary-v1.json").read_text(encoding="utf-8"))
        assert report.cost_summary.total_tokens == stored_cost["total_tokens"]
        assert report.cost_summary.usage_coverage_rate == stored_cost["usage_coverage_rate"]
        assert (
            report.cost_summary.monetary_cost_available
            is stored_cost["monetary_cost_available"]
        )

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

    def test_formal_rule_extraction_requires_strict_report_and_n50(self, tmp_path: Path) -> None:
        smoke = json.loads(
            (Path(__file__).resolve().parents[1] / "reports" / "rule-extraction-smoke-v1.json")
            .read_text(encoding="utf-8")
        )
        base_result = smoke["results"][0]
        smoke.update(
            {
                "dataset_version": "formal-rules-v1",
                "source_sha256": "f" * 64,
                "sample_size": 50,
                "scored_rule_units": 50,
                "case_count": 50,
                "real_sources": True,
                "authorization_verified": True,
                "independent_real_samples": True,
                "real_external_service": True,
                "exact_rule_accuracy": {
                    "correct": 50,
                    "total": 50,
                    "value": 1.0,
                    "ci95": [0.9286524009, 1.0],
                },
                "field_micro_accuracy": {
                    "correct": 550,
                    "total": 550,
                    "value": 1.0,
                    "ci95": [0.993064, 1.0],
                },
                "field_micro_prf": {
                    "true_positive": 200,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "precision_ci95": [0.981154, 1.0],
                    "recall_ci95": [0.981154, 1.0],
                },
                "complete_case_accuracy": {
                    "correct": 50,
                    "total": 50,
                    "value": 1.0,
                    "ci95": [0.9286524009, 1.0],
                },
                "formal_gate_eligible": True,
                "passed": True,
                "smoke_test_only": False,
                "limitations": [],
            }
        )
        for metric in smoke["field_metrics"].values():
            metric.update(correct=50, total=50, value=1.0, ci95=[0.9286524009, 1.0])
        for metric in smoke["field_prf"].values():
            if metric["true_positive"]:
                metric.update(
                    true_positive=50,
                    precision_ci95=[0.9286524009, 1.0],
                    recall_ci95=[0.9286524009, 1.0],
                )
        smoke["results"] = [
            {
                **base_result,
                "case_id": f"case-{index}",
                "source_id": f"source-{index}",
                "source_sha256": f"{index:064x}",
            }
            for index in range(50)
        ]
        _write(tmp_path / "rule-extraction-formal-v1.json", smoke)
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        gate = next(item for item in report.gates if item.id == "rule_extraction")
        assert gate.status == "通过"
        assert gate.sample_size == 50

    def test_forged_rule_extraction_counts_cannot_pass(self, tmp_path: Path) -> None:
        smoke = json.loads(
            (Path(__file__).resolve().parents[1] / "reports" / "rule-extraction-smoke-v1.json")
            .read_text(encoding="utf-8")
        )
        smoke.update(
            sample_size=50,
            formal_gate_eligible=True,
            passed=True,
            smoke_test_only=False,
            real_sources=True,
            authorization_verified=True,
            independent_real_samples=True,
        )
        _write(tmp_path / "rule-extraction-formal-v1.json", smoke)
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        gate = next(item for item in report.gates if item.id == "rule_extraction")
        assert gate.status == "阻塞"
        assert any("Schema 无效" in reason for reason in gate.reasons)

    def test_user_trial_requires_consent_hash_intervals_and_latency(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "user-trial-v1.json",
            {
                "sample_size": 2,
                "session_count": 2,
                "formal_gate_eligible": True,
                "real_users": True,
                "consent_verified": True,
                "authorization_verified": True,
                "independent_real_users": True,
                "source_sha256": "a" * 64,
                "task_success_rate": {
                    "successes": 2,
                    "total": 2,
                    "value": 1.0,
                    "ci95": [0.34238, 1.0],
                },
                "duration_p50_seconds": 60.0,
                "duration_p90_seconds": 90.0,
            },
        )
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        gate = next(item for item in report.gates if item.id == "user_trial")
        assert gate.status == "通过"
        assert gate.sample_size == 2

    def test_one_click_demo_requires_smoke_flags_steps_and_artifact_hashes(
        self, tmp_path: Path
    ) -> None:
        artifact_names = (
            "rules_input",
            "claims_input",
            "totals_reference",
            "items_reference",
            "rules_database",
            "rules_export",
            "calculation",
            "backtest",
            "backtest_diffs",
        )
        artifacts = [
            {
                "name": name,
                "relative_path": f"artifact-{index}.json",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
            for index, name in enumerate(artifact_names)
        ]
        steps = [
            {
                "id": step_id,
                "status": "passed",
                "exit_code": 0,
                "duration_seconds": 0.1,
            }
            for step_id in (
                "generate_sample_files",
                "import_rules",
                "calculate_scores",
                "backtest_items",
                "explain_claim",
            )
        ]
        _write(
            tmp_path / "demo-smoke-v1.json",
            {
                "schema_version": "1.0",
                "generated_at": "2026-09-20T00:00:00+00:00",
                "dataset_kind": "synthetic",
                "smoke_test_only": True,
                "formal_gate_eligible": False,
                "all_steps_passed": True,
                "business_checks_passed": True,
                "sample_students": 5,
                "sample_claims": 13,
                "imported_rules": 13,
                "person_agreement": 1.0,
                "item_agreement": 1.0,
                "steps": steps,
                "artifacts": artifacts,
                "limitations": ["仅合成烟雾测试"],
            },
        )
        report = build_release_readiness(tmp_path, candidate_version="candidate")
        gate = next(item for item in report.gates if item.id == "one_click_demo")
        assert gate.status == "通过"
        assert gate.sample_size == 5
        assert gate.metrics["smoke_test_only"] is True

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
