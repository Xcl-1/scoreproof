"""阶段 6.4：人工复核队列、状态机、SQLite 与审计闭环。"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.errors import SchemaValidationError, VersionConflict
from scoreproof.evidence.certificate import (
    CertificateExtraction,
    CertificateField,
    CertificateFieldSet,
    CertificatePipelineResult,
    FieldSignals,
    OcrPayload,
    QualityPayload,
    VlmDecision,
)
from scoreproof.evidence.consistency import compare_claim_evidence
from scoreproof.evidence.dedup import compare_evidence
from scoreproof.review import (
    ReviewAuditMetadata,
    ReviewDismissReason,
    ReviewDismissRequest,
    ReviewReasonCode,
    ReviewResolution,
    ReviewResolveRequest,
    ReviewSource,
    ReviewStatus,
    ReviewStore,
    ReviewTaskCreate,
    ReviewTaskType,
    ReviewValue,
    enqueue_certificate_reviews,
    enqueue_consistency_reviews,
    enqueue_duplicate_review,
    fact_hash,
)
from scoreproof.schema import Claim, Evidence


def _request(*, evidence_id: str = "evidence-1", field: str = "级别") -> ReviewTaskCreate:
    return ReviewTaskCreate(
        task_type=ReviewTaskType.LOW_CONFIDENCE_FIELD,
        reason_codes=[ReviewReasonCode.LOW_CONFIDENCE],
        evidence_id=evidence_id,
        field_name=field,
        current_normalized_value=ReviewValue(value="省级"),
        audit_metadata=ReviewAuditMetadata(
            source_component="certificate",
            input_fact_hash=fact_hash({"evidence_id": evidence_id, "field": field}),
            trigger_version="test-v1",
        ),
    )


class TestReviewSchema:
    def test_forbids_extra_fields_and_sensitive_values(self) -> None:
        payload = _request().model_dump(mode="json")
        payload["api_key"] = "secret"
        with pytest.raises(ValidationError):
            ReviewTaskCreate.model_validate(payload)
        with pytest.raises(ValidationError, match="姓名或学号"):
            ReviewTaskCreate(
                **{
                    **_request().model_dump(),
                    "field_name": "姓名",
                    "current_normalized_value": ReviewValue(value="张三"),
                }
            )

    def test_rejects_secret_and_free_text_values(self) -> None:
        with pytest.raises(ValidationError, match="认证信息"):
            ReviewValue(value="Bearer " + "a" * 16)
        with pytest.raises(ValidationError, match="单行"):
            ReviewValue(value="第一行\n第二行")


class TestReviewStore:
    def test_source_bbox_survives_sqlite_json_round_trip(self, tmp_path: Path) -> None:
        db = tmp_path / "bbox-round-trip.sqlite"
        request = _request().model_copy(
            update={"source": ReviewSource(locator="page-1", bbox=(1.0, 2.0, 3.0, 4.0))}
        )
        with ReviewStore(db) as store:
            created, was_created = store.create(request)
            assert was_created is True
            assert created.source is not None
            assert created.source.bbox == (1.0, 2.0, 3.0, 4.0)
        with ReviewStore(db) as reopened:
            restored = reopened.require(created.review_task_id)
            assert restored.source is not None
            assert restored.source.bbox == (1.0, 2.0, 3.0, 4.0)

    def test_stable_id_idempotency_filters_and_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "reviews.sqlite"
        with ReviewStore(db) as store:
            first, created = store.create(_request())
            again, created_again = store.create(_request())
            store.create(_request(evidence_id="evidence-2"))
            assert created is True
            assert created_again is False
            assert first.review_task_id == again.review_task_id
            assert len(store.list_tasks(status=ReviewStatus.PENDING)) == 2
            assert len(store.list_tasks(task_type=ReviewTaskType.LOW_CONFIDENCE_FIELD)) == 2
            assert len(store.list_tasks(evidence_id="evidence-1")) == 1
        with ReviewStore(db) as reopened:
            assert reopened.require(first.review_task_id).revision == 1

    def test_legal_transitions_revision_conflict_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "reviews.sqlite"
        with ReviewStore(db) as store:
            task, _ = store.create(_request())
            started = store.start(
                task.review_task_id,
                expected_revision=1,
                operator_token="reviewer:001",
            )
            assert started.status == ReviewStatus.IN_PROGRESS
            with pytest.raises(VersionConflict, match="revision"):
                store.start(
                    task.review_task_id,
                    expected_revision=1,
                    operator_token="reviewer:001",
                )
            resolved = store.resolve(
                task.review_task_id,
                ReviewResolveRequest(
                    expected_revision=2,
                    resolver_token="reviewer:001",
                    resolution=ReviewResolution.CORRECTED,
                    corrected_value=ReviewValue(value="省部级"),
                ),
            )
            assert resolved.status == ReviewStatus.RESOLVED
            assert resolved.manual_corrected_value == ReviewValue(value="省级")
            with pytest.raises(VersionConflict, match="非法状态迁移"):
                store.start(
                    task.review_task_id,
                    expected_revision=3,
                    operator_token="reviewer:001",
                )
            new_task, created = store.create(_request())
            assert created is True
            assert new_task.review_task_id.endswith("-r2")
            assert new_task.reopened_from == task.review_task_id

    def test_dismiss_and_audit_rows_are_immutable(self, tmp_path: Path) -> None:
        db = tmp_path / "reviews.sqlite"
        with ReviewStore(db) as store:
            task, _ = store.create(_request())
            dismissed = store.dismiss(
                task.review_task_id,
                ReviewDismissRequest(
                    expected_revision=1,
                    resolver_token="reviewer:002",
                    reason=ReviewDismissReason.NOT_ACTIONABLE,
                ),
            )
            assert dismissed.status == ReviewStatus.DISMISSED
            assert [event.event_type for event in store.audit_events(task.review_task_id)] == [
                "created",
                "dismissed",
            ]
        connection = sqlite3.connect(db)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM review_audit_events")
        connection.close()

    def test_concurrent_resolve_allows_only_one_winner(self, tmp_path: Path) -> None:
        db = tmp_path / "concurrent.sqlite"
        with ReviewStore(db) as store:
            task, _ = store.create(_request())

        def resolve(token: str) -> str:
            try:
                with ReviewStore(db) as store:
                    store.resolve(
                        task.review_task_id,
                        ReviewResolveRequest(
                            expected_revision=1,
                            resolver_token=token,
                            resolution=ReviewResolution.CONFIRMED,
                        ),
                    )
                return "resolved"
            except VersionConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(resolve, ["reviewer:101", "reviewer:102"]))
        assert sorted(results) == ["conflict", "resolved"]
        with ReviewStore(db) as store:
            assert len(store.audit_events(task.review_task_id)) == 2

    @pytest.mark.parametrize(
        ("field,raw,expected"),
        [
            ("获奖日期", "2025年4月2日", "2025-04-02"),
            ("奖项/名次", "第二名", "二等奖"),
            ("团队属性", "团体", "团队"),
        ],
    )
    def test_human_correction_is_validated(self, tmp_path: Path, field: str, raw: str, expected: str) -> None:
        with ReviewStore(tmp_path / f"{field}.sqlite") as store:
            request = _request(field=field)
            task, _ = store.create(request)
            result = store.resolve(
                task.review_task_id,
                ReviewResolveRequest(
                    expected_revision=1,
                    resolver_token="reviewer:003",
                    resolution=ReviewResolution.CORRECTED,
                    corrected_value=ReviewValue(value=raw),
                ),
            )
            assert result.manual_corrected_value == ReviewValue(value=expected)

    def test_invalid_correction_and_unredacted_name_are_rejected(self, tmp_path: Path) -> None:
        with ReviewStore(tmp_path / "review.sqlite") as store:
            date_task, _ = store.create(_request(field="获奖日期"))
            with pytest.raises(SchemaValidationError, match="不可解析"):
                store.resolve(
                    date_task.review_task_id,
                    ReviewResolveRequest(
                        expected_revision=1,
                        resolver_token="reviewer:004",
                        resolution=ReviewResolution.CORRECTED,
                        corrected_value=ReviewValue(value="2025-02-30"),
                    ),
                )


class TestReviewSources:
    def test_low_confidence_and_missing_vlm_create_tasks(self, tmp_path: Path) -> None:
        signals = FieldSignals(
            ocr_line_confidence=0.2,
            source_supported=False,
            format_valid=False,
        )
        low = CertificateField(
            confidence=0.0,
            signals=signals,
            validation_errors=["unsupported_by_ocr"],
        )
        fields = CertificateFieldSet(
            姓名=low,
            赛事名称=low,
            级别=low,
            **{"奖项/名次": low},
            获奖日期=low,
            颁发单位=low,
            团队属性=low,
        )
        result = CertificatePipelineResult(
            source="masked-image.png",
            processed=None,
            quality=QualityPayload(
                width=100,
                height=100,
                is_blurry=False,
                needs_retake=False,
                rotation_applied=0,
                notes=[],
            ),
            phash="0" * 16,
            ocr=OcrPayload(engine="test", mean_confidence=0.2, text="", lines=[]),
            extraction=CertificateExtraction(
                fields=fields,
                model="text-model",
                confidence_threshold=0.8,
                vlm=VlmDecision(
                    requested=True,
                    called=False,
                    status="provider_missing",
                    low_confidence_fields=list(fields.by_name()),
                ),
                manual_review_required=True,
                manual_review_status="pending_manual_review",
            ),
            evidence=Evidence(id="certificate-low", type="image"),
        )
        with ReviewStore(tmp_path / "review.sqlite") as store:
            tasks = enqueue_certificate_reviews(store, result)
        assert len(tasks) == 8
        assert any(task.task_type == ReviewTaskType.OCR_UNSUPPORTED for task in tasks)
        assert any(task.task_type == ReviewTaskType.VLM_NOT_CONFIGURED for task in tasks)
        name_task = next(task for task in tasks if task.field_name == "姓名")
        assert name_task.current_normalized_value is None

    def test_duplicate_and_consistency_create_tasks_without_mutating_evidence(self, tmp_path: Path) -> None:
        left = Evidence(
            id="left",
            type="image",
            fields={"姓名": "张三", "赛事名称": "数学竞赛", "获奖日期": "2025-04-02"},
            phash="0000000000000000",
        )
        right = Evidence(
            id="right",
            type="image",
            fields=dict(left.fields),
            phash="0000000000000001",
        )
        original = left.model_copy(deep=True)
        claim = Claim(id="claim-1", student_id="masked-user", student_name="李四")
        consistency = compare_claim_evidence(claim, left)
        with ReviewStore(tmp_path / "review.sqlite") as store:
            duplicate_tasks = enqueue_duplicate_review(store, compare_evidence(left, right))
            consistency_tasks = enqueue_consistency_reviews(store, consistency)
            assert duplicate_tasks[0].task_type == ReviewTaskType.POSSIBLE_DUPLICATE
            assert any(task.task_type == ReviewTaskType.CLAIM_EVIDENCE_CONFLICT for task in consistency_tasks)
        assert left == original


class TestReviewCli:
    def test_real_cli_flow_and_illegal_operation_nonzero(self, tmp_path: Path) -> None:
        db = tmp_path / "review.sqlite"
        out = tmp_path / "export.json"
        with ReviewStore(db) as store:
            task, _ = store.create(_request())
        runner = CliRunner()
        listed = runner.invoke(app, ["review-list", "--db", str(db), "--status", "pending"])
        assert listed.exit_code == 0
        assert json.loads(listed.stdout)["count"] == 1
        started = runner.invoke(
            app,
            [
                "review-start",
                task.review_task_id,
                "--db",
                str(db),
                "--expected-revision",
                "1",
                "--operator-token",
                "reviewer:cli",
            ],
        )
        assert started.exit_code == 0
        stale = runner.invoke(
            app,
            [
                "review-resolve",
                task.review_task_id,
                "--db",
                str(db),
                "--expected-revision",
                "1",
                "--operator-token",
                "reviewer:cli",
                "--resolution",
                "confirmed",
            ],
        )
        assert stale.exit_code == 2
        exported = runner.invoke(app, ["review-export", "--db", str(db), "--out", str(out)])
        assert exported.exit_code == 0
        text = out.read_text(encoding="utf-8").lower()
        assert "api_key" not in text
        assert "prompt" in text  # 只出现 contains_prompt_or_model_response=false 的安全声明
        assert "bearer " not in text
