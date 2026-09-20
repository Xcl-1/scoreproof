"""阶段 7.3：真实用户试用记录、隐私门禁与效率评测。"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.eval.user_trial import (
    TrialSession,
    UserTrialDataset,
    evaluate_user_trials,
    load_user_trial_dataset,
    new_participant_token,
    user_trial_template,
)


def _session(
    index: int,
    *,
    duration: int = 60,
    completed: bool = True,
    verified: bool = True,
    synthetic: bool = False,
    consent: bool = True,
    real_user: bool = True,
) -> TrialSession:
    started = datetime(2026, 9, 20, 9, index, tzinfo=UTC)
    return TrialSession(
        trial_id=f"trial-{index:012x}",
        participant_token=f"participant-{index:024x}",
        consent_confirmed=consent,
        real_user_confirmed=real_user,
        synthetic=synthetic,
        entrypoint="cli" if index % 2 else "api",
        task="score_calculation",
        started_at=started,
        completed_at=started + timedelta(seconds=duration),
        task_completed=completed,
        result_verified=verified,
        manual_review_required=not completed,
        issue_code="none" if completed else "result_mismatch",
        satisfaction_score=5 if completed else 2,
    )


def _formal_dataset() -> UserTrialDataset:
    return UserTrialDataset(
        dataset_version="user-trial-test-v1",
        authorization_reference="approval:trial-2026-09",
        consent_document_version="consent-v1",
        independent_real_users=True,
        sessions=[
            _session(1, duration=60),
            _session(2, duration=180, completed=False, verified=False),
        ],
    )


def test_schema_forbids_pii_and_free_text_fields() -> None:
    payload = _session(1).model_dump(mode="python")
    payload["student_name"] = "不应进入评测记录"
    with pytest.raises(ValidationError):
        TrialSession.model_validate(payload)


def test_session_requires_timezone_and_forward_time() -> None:
    payload = _session(1).model_dump(mode="python")
    payload["completed_at"] = payload["started_at"]
    with pytest.raises(ValidationError):
        TrialSession.model_validate(payload)


def test_empty_template_can_never_pass_formal_gate() -> None:
    report = evaluate_user_trials(user_trial_template())
    assert report.sample_size == 0
    assert report.task_success_rate.value is None
    assert report.duration_p90_seconds is None
    assert report.real_users is False
    assert report.formal_gate_eligible is False
    assert report.smoke_test_only is True


def test_synthetic_or_unconsented_session_is_blocked() -> None:
    dataset = UserTrialDataset(
        dataset_version="smoke",
        authorization_reference="approval:smoke",
        consent_document_version="consent-v1",
        independent_real_users=True,
        sessions=[_session(1, synthetic=True, consent=False)],
    )
    report = evaluate_user_trials(dataset)
    assert report.real_users is False
    assert report.consent_verified is False
    assert report.formal_gate_eligible is False


def test_authorized_metadata_computes_rates_intervals_and_latency() -> None:
    report = evaluate_user_trials(_formal_dataset())
    assert report.session_count == report.sample_size == 2
    assert report.task_success_rate.value == 0.5
    assert report.task_success_rate.ci95 is not None
    assert report.verified_result_rate.value == 0.5
    assert report.manual_review_rate.value == 0.5
    assert report.duration_p50_seconds == 120
    assert report.duration_p90_seconds == 180
    assert report.satisfaction_mean == 3.5
    assert report.formal_gate_eligible is True


def test_duplicate_trial_ids_are_rejected() -> None:
    first = _session(1)
    second = _session(2).model_copy(update={"trial_id": first.trial_id})
    with pytest.raises(ValidationError):
        UserTrialDataset(
            dataset_version="duplicate",
            independent_real_users=True,
            sessions=[first, second],
        )


def test_random_participant_token_has_no_identity_payload() -> None:
    first = new_participant_token()
    second = new_participant_token()
    assert first.startswith("participant-")
    assert len(first) == 44
    assert first != second


def test_json_loader_preserves_strict_datetime_contract(tmp_path) -> None:
    source = tmp_path / "trials.json"
    source.write_text(_formal_dataset().model_dump_json(indent=2), encoding="utf-8")
    loaded = load_user_trial_dataset(source)
    assert loaded.sessions[0].started_at.tzinfo is not None
    assert loaded.dataset_version == "user-trial-test-v1"


def test_cli_template_is_explicitly_blocked_and_formal_fixture_can_run(tmp_path) -> None:
    runner = CliRunner()
    template = tmp_path / "template.json"
    exported = runner.invoke(app, ["export-user-trial-template", "--out", str(template)])
    assert exported.exit_code == 0, exported.output
    assert json.loads(template.read_text(encoding="utf-8"))["sessions"] == []

    blocked_report = tmp_path / "blocked.json"
    blocked = runner.invoke(
        app,
        ["eval-user-trial", str(template), "--out", str(blocked_report)],
    )
    assert blocked.exit_code == 2
    assert json.loads(blocked_report.read_text(encoding="utf-8"))["formal_gate_eligible"] is False

    formal_source = tmp_path / "formal.json"
    formal_report = tmp_path / "formal-report.json"
    formal_source.write_text(_formal_dataset().model_dump_json(indent=2), encoding="utf-8")
    accepted = runner.invoke(
        app,
        ["eval-user-trial", str(formal_source), "--out", str(formal_report)],
    )
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(formal_report.read_text(encoding="utf-8"))["sample_size"] == 2
