"""持久化批量核算任务、幂等与导出测试。"""

from __future__ import annotations

import csv
import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scoreproof.batch import (
    BatchArtifact,
    BatchInput,
    BatchStatus,
    BatchStore,
    artifact_to_csv,
)
from scoreproof.calc import compute_all
from scoreproof.errors import SchemaValidationError, VersionConflict
from scoreproof.schema import Claim, Ruleset

from .conftest import make_rule


def _input(*, text: str = "省一等奖", student_name: str | None = None) -> BatchInput:
    return BatchInput(
        claims=[
            Claim(
                id="claim-1",
                student_id="2023001",
                student_name=student_name,
                academic_year="2025-2026",
                category="学科竞赛",
                raw_text=text,
                level="省级一等奖",
            )
        ],
        academic_year="2025-2026",
    )


def _artifact(batch_id: str, batch_input: BatchInput) -> BatchArtifact:
    ruleset = Ruleset(
        rules=[make_rule("省级一等奖", 10, group="学科竞赛", rule_id="r1")]
    )
    evaluated = [claim.model_copy(deep=True) for claim in batch_input.claims]
    results = compute_all(evaluated, ruleset, academic_year="2025-2026")
    return BatchArtifact(
        batch_id=batch_id,
        rule_version_id="rv_test",
        rule_content_hash="a" * 64,
        calculated_at=datetime.now(UTC),
        input=batch_input,
        evaluated_claims=evaluated,
        results=results,
    )


def test_batch_persists_lifecycle_and_terminal_artifact_is_immutable(tmp_path: Path) -> None:
    db = tmp_path / "batch.sqlite"
    with BatchStore(db) as store:
        task, created = store.create(
            _input(),
            rule_version_id="rv_test",
            rule_content_hash="a" * 64,
            rule_count=1,
            idempotency_key="batch-test-001",
        )
        assert created is True and task.status is BatchStatus.QUEUED
        assert store.start(task.batch_id).status is BatchStatus.RUNNING
        wrong_snapshot = _artifact(task.batch_id, _input()).model_copy(
            update={"rule_content_hash": "b" * 64}
        )
        with pytest.raises(SchemaValidationError, match="规则快照"):
            store.complete(task.batch_id, wrong_snapshot)
        completed = store.complete(task.batch_id, _artifact(task.batch_id, _input()))
        assert completed.status is BatchStatus.SUCCEEDED
        assert completed.result_hash is not None
        assert [event.event_type for event in store.events(task.batch_id)] == [
            "created",
            "started",
            "succeeded",
        ]
        detail = store.detail(task.batch_id)
        assert detail is not None and detail.artifact is not None
        assert detail.artifact.results["2023001"].total == 10
        with pytest.raises(sqlite3.IntegrityError, match="terminal score batches are immutable"):
            store._conn.execute(  # noqa: SLF001 - 明确验证数据库级不可变约束
                "UPDATE score_batches SET result_hash=? WHERE batch_id=?",
                ("b" * 64, task.batch_id),
            )

    with BatchStore(db) as reopened:
        persisted = reopened.get_task(task.batch_id)
        assert persisted is not None and persisted.status is BatchStatus.SUCCEEDED
        assert reopened.get_artifact(task.batch_id) is not None


def test_idempotency_reuses_identical_request_and_rejects_changed_input(tmp_path: Path) -> None:
    with BatchStore(tmp_path / "batch.sqlite") as store:
        first, created = store.create(
            _input(),
            rule_version_id="rv_test",
            rule_content_hash="a" * 64,
            rule_count=1,
            idempotency_key="batch-test-002",
        )
        assert created is True
        second, created = store.create(
            _input(),
            rule_version_id="rv_test",
            rule_content_hash="a" * 64,
            rule_count=1,
            idempotency_key="batch-test-002",
        )
        assert created is False and second.batch_id == first.batch_id
        with pytest.raises(VersionConflict):
            store.create(
                _input(text="不同输入"),
                rule_version_id="rv_test",
                rule_content_hash="a" * 64,
                rule_count=1,
                idempotency_key="batch-test-002",
            )


def test_failed_task_is_audited_and_cannot_transition_again(tmp_path: Path) -> None:
    with BatchStore(tmp_path / "batch.sqlite") as store:
        task, _ = store.create(
            _input(),
            rule_version_id="rv_test",
            rule_content_hash="a" * 64,
            rule_count=1,
        )
        failed = store.fail(task.batch_id, error_code="test_error", error_message="预期失败")
        assert failed.status is BatchStatus.FAILED
        assert failed.error_code == "test_error"
        assert [event.event_type for event in store.events(task.batch_id)] == ["created", "failed"]
        with pytest.raises(VersionConflict):
            store.start(task.batch_id)


def test_csv_is_traceable_and_neutralizes_spreadsheet_formulas() -> None:
    batch_input = _input(student_name="=HYPERLINK(\"https://example.invalid\")")
    payload = artifact_to_csv(_artifact("bat_export", batch_input))
    assert payload.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert len(rows) == 1
    assert rows[0]["student_name"].startswith("'=")
    assert rows[0]["rule_id"] == "r1"
    assert rows[0]["source_doc"] == "合成细则.pdf"
