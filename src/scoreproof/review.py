"""人工复核队列、校对闭环与不可变审计。

本模块只保存结构化、最小化的复核事实。它不会保存提示词、模型原始回复、
图片内容或未脱敏身份，也不会在任务解决时隐式修改原始 ``Evidence``。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .errors import SchemaValidationError, VersionConflict
from .evidence.certificate import CertificatePipelineResult
from .evidence.consistency import EvidenceConsistencyReport
from .evidence.dedup import DuplicateDecision
from .normalize import parse_prize, parse_tier


class ReviewTaskType(StrEnum):
    LOW_CONFIDENCE_FIELD = "low_confidence_field"
    OCR_UNSUPPORTED = "ocr_unsupported"
    VLM_NOT_CONFIGURED = "vlm_not_configured"
    VLM_CALL_FAILED = "vlm_call_failed"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    CLAIM_EVIDENCE_CONFLICT = "claim_evidence_conflict"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    MANUAL_REQUEST = "manual_request"


class ReviewReasonCode(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_BY_OCR = "unsupported_by_ocr"
    INVALID_FORMAT_OR_ENUM = "invalid_format_or_enum"
    MULTI_SOURCE_DISAGREEMENT = "multi_source_disagreement"
    VLM_PROVIDER_MISSING = "vlm_provider_missing"
    VLM_KEY_MISSING = "vlm_key_missing"
    VLM_CROP_MISSING = "vlm_crop_missing"
    VLM_CALL_FAILED = "vlm_call_failed"
    DUPLICATE_CONFIRMED = "duplicate_confirmed"
    DUPLICATE_SUSPECTED = "duplicate_suspected"
    CLAIM_EVIDENCE_MISMATCH = "claim_evidence_mismatch"
    CLAIM_EVIDENCE_INSUFFICIENT = "claim_evidence_insufficient"
    IMAGE_QUALITY = "image_quality"
    MANUAL_REQUEST = "manual_request"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ReviewPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewResolution(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    DUPLICATE_CONFIRMED = "duplicate_confirmed"
    NOT_DUPLICATE = "not_duplicate"
    INSUFFICIENT = "insufficient"


class ReviewDismissReason(StrEnum):
    NOT_ACTIONABLE = "not_actionable"
    DUPLICATE_TASK = "duplicate_task"
    SUPERSEDED = "superseded"
    WRONG_SCOPE = "wrong_scope"


ReviewScalar: TypeAlias = StrictStr | StrictBool | StrictInt | StrictFloat


_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|authorization\s*:|bearer\s+|sk-[a-z0-9_-]{8,}|"
    r"dashscope_api_key|zhipuai_api_key|deepseek_api_key)"
)
_OPERATOR_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_STUDENT_ID_RE = re.compile(r"(?<!\d)\d{8,18}(?!\d)")
_SENSITIVE_FIELDS = {"姓名", "学号", "student_name", "student_id"}


def _validate_safe_text(value: str, *, label: str, max_length: int = 200) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label}不能为空")
    if len(cleaned) > max_length:
        raise ValueError(f"{label}长度不能超过 {max_length}")
    if "\n" in cleaned or "\r" in cleaned or "\x00" in cleaned:
        raise ValueError(f"{label}只能是单行结构化值")
    if _SECRET_RE.search(cleaned):
        raise ValueError(f"{label}疑似包含密钥或认证信息")
    return cleaned


class ReviewValue(BaseModel):
    """受限的单字段值；禁止自由文本对象和任意扩展字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    value: ReviewScalar

    @field_validator("value")
    @classmethod
    def _safe_value(cls, value: ReviewScalar) -> ReviewScalar:
        if isinstance(value, str):
            cleaned = _validate_safe_text(value, label="复核值")
            if _STUDENT_ID_RE.search(cleaned):
                raise ValueError("复核值疑似包含未脱敏学号")
            return cleaned
        return value


class ReviewSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    locator: StrictStr | None = Field(default=None, max_length=200)
    bbox: tuple[float, float, float, float] | None = None

    @field_validator("locator")
    @classmethod
    def _safe_locator(cls, value: str | None) -> str | None:
        return None if value is None else _validate_safe_text(value, label="source locator")


class ReviewAuditMetadata(BaseModel):
    """允许持久化的最小审计元数据，不接受任意字典。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_component: Literal["certificate", "vlm", "dedup", "consistency", "manual"]
    input_fact_hash: StrictStr
    trigger_version: StrictStr = Field(max_length=80)
    source_event_id: StrictStr | None = Field(default=None, max_length=128)
    related_task_id: StrictStr | None = Field(default=None, max_length=80)
    sensitive_fields_redacted: Literal[True] = True

    @field_validator("input_fact_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError("input_fact_hash 必须是 SHA-256 十六进制")
        return value

    @field_validator("trigger_version", "source_event_id", "related_task_id")
    @classmethod
    def _safe_metadata(cls, value: str | None) -> str | None:
        return None if value is None else _validate_safe_text(value, label="审计元数据", max_length=128)


class ReviewTaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: ReviewTaskType
    reason_codes: list[ReviewReasonCode] = Field(min_length=1, max_length=12)
    priority: ReviewPriority = ReviewPriority.NORMAL
    evidence_id: StrictStr = Field(min_length=1, max_length=128)
    claim_id: StrictStr | None = Field(default=None, max_length=128)
    field_name: StrictStr | None = Field(default=None, max_length=64)
    source: ReviewSource | None = None
    current_normalized_value: ReviewValue | None = None
    audit_metadata: ReviewAuditMetadata

    @field_validator("reason_codes")
    @classmethod
    def _unique_reasons(cls, value: list[ReviewReasonCode]) -> list[ReviewReasonCode]:
        return list(dict.fromkeys(value))

    @field_validator("evidence_id", "claim_id", "field_name")
    @classmethod
    def _safe_identifiers(cls, value: str | None) -> str | None:
        return None if value is None else _validate_safe_text(value, label="复核标识", max_length=128)

    @model_validator(mode="after")
    def _no_pii_value(self) -> ReviewTaskCreate:
        if self.field_name in _SENSITIVE_FIELDS and self.current_normalized_value is not None:
            raise ValueError("复核队列不得保存未脱敏姓名或学号")
        allowed_reasons = {
            ReviewTaskType.LOW_CONFIDENCE_FIELD: {
                ReviewReasonCode.LOW_CONFIDENCE,
                ReviewReasonCode.INVALID_FORMAT_OR_ENUM,
                ReviewReasonCode.MULTI_SOURCE_DISAGREEMENT,
            },
            ReviewTaskType.OCR_UNSUPPORTED: {
                ReviewReasonCode.UNSUPPORTED_BY_OCR,
                ReviewReasonCode.LOW_CONFIDENCE,
                ReviewReasonCode.INVALID_FORMAT_OR_ENUM,
                ReviewReasonCode.MULTI_SOURCE_DISAGREEMENT,
            },
            ReviewTaskType.VLM_NOT_CONFIGURED: {
                ReviewReasonCode.VLM_PROVIDER_MISSING,
                ReviewReasonCode.VLM_KEY_MISSING,
                ReviewReasonCode.VLM_CROP_MISSING,
            },
            ReviewTaskType.VLM_CALL_FAILED: {ReviewReasonCode.VLM_CALL_FAILED},
            ReviewTaskType.POSSIBLE_DUPLICATE: {
                ReviewReasonCode.DUPLICATE_CONFIRMED,
                ReviewReasonCode.DUPLICATE_SUSPECTED,
            },
            ReviewTaskType.CLAIM_EVIDENCE_CONFLICT: {
                ReviewReasonCode.CLAIM_EVIDENCE_MISMATCH
            },
            ReviewTaskType.INSUFFICIENT_INFORMATION: {
                ReviewReasonCode.CLAIM_EVIDENCE_INSUFFICIENT,
                ReviewReasonCode.IMAGE_QUALITY,
            },
            ReviewTaskType.MANUAL_REQUEST: {ReviewReasonCode.MANUAL_REQUEST},
        }
        unexpected = set(self.reason_codes) - allowed_reasons[self.task_type]
        if unexpected:
            raise ValueError(f"task_type 与 reason_codes 不匹配：{sorted(item.value for item in unexpected)}")
        return self


class ReviewTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_task_id: str
    stable_key: str
    generation: int = Field(ge=1)
    task_type: ReviewTaskType
    reason_codes: list[ReviewReasonCode]
    status: ReviewStatus
    priority: ReviewPriority
    evidence_id: str
    claim_id: str | None = None
    field_name: str | None = None
    source: ReviewSource | None = None
    current_normalized_value: ReviewValue | None = None
    manual_corrected_value: ReviewValue | None = None
    resolution: ReviewResolution | None = None
    dismiss_reason: ReviewDismissReason | None = None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None
    resolver_token: str | None = None
    revision: int = Field(ge=1)
    audit_metadata: ReviewAuditMetadata
    reopened_from: str | None = None


class ReviewAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int
    review_task_id: str
    event_type: Literal["created", "reopened", "started", "resolved", "dismissed"]
    from_status: ReviewStatus | None = None
    to_status: ReviewStatus
    revision_before: int = Field(ge=0)
    revision_after: int = Field(ge=1)
    operator_token: str
    created_at: datetime
    resolution: ReviewResolution | None = None
    dismiss_reason: ReviewDismissReason | None = None


class ReviewWorkflowSmokeReport(BaseModel):
    """阶段 6.4 工程烟雾报告；永远不能替代真实业务用户验收。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: StrictStr
    dataset_kind: Literal["synthetic_engineering_smoke"]
    smoke_test_only: Literal[True]
    formal_gate_eligible: Literal[False]
    sample_task_count: StrictInt = Field(ge=1)
    sqlite_persistence_passed: StrictBool
    cli_flow_passed: StrictBool
    uvicorn_http_passed: StrictBool
    idempotency_passed: StrictBool
    state_machine_passed: StrictBool
    revision_conflict_passed: StrictBool
    immutable_audit_passed: StrictBool
    multimodal_sources_passed: StrictBool
    no_implicit_evidence_mutation: Literal[True]
    sensitive_export_scan_passed: StrictBool
    limitations: list[StrictStr] = Field(min_length=1)

    @property
    def engineering_passed(self) -> bool:
        return all(
            (
                self.sqlite_persistence_passed,
                self.cli_flow_passed,
                self.uvicorn_http_passed,
                self.idempotency_passed,
                self.state_machine_passed,
                self.revision_conflict_passed,
                self.immutable_audit_passed,
                self.multimodal_sources_passed,
                self.sensitive_export_scan_passed,
            )
        )


class ReviewResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: StrictInt = Field(ge=1)
    resolver_token: StrictStr
    resolution: ReviewResolution
    corrected_value: ReviewValue | None = None
    apply_to_evidence: Literal[False] = False

    @model_validator(mode="after")
    def _correction_contract(self) -> ReviewResolveRequest:
        if self.resolution == ReviewResolution.CORRECTED and self.corrected_value is None:
            raise ValueError("resolution=corrected 时必须提供 corrected_value")
        if self.resolution != ReviewResolution.CORRECTED and self.corrected_value is not None:
            raise ValueError("只有 resolution=corrected 可提供 corrected_value")
        return self


class ReviewStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: StrictInt = Field(ge=1)
    operator_token: StrictStr


class ReviewDismissRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: StrictInt = Field(ge=1)
    resolver_token: StrictStr
    reason: ReviewDismissReason


REVIEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review_tasks (
    review_task_id TEXT PRIMARY KEY,
    stable_key TEXT NOT NULL,
    generation INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    claim_id TEXT,
    field_name TEXT,
    source TEXT,
    current_normalized_value TEXT,
    manual_corrected_value TEXT,
    resolution TEXT,
    dismiss_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    resolver_token TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    audit_metadata TEXT NOT NULL,
    reopened_from TEXT,
    UNIQUE(stable_key, generation)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_review_tasks_active
ON review_tasks(stable_key) WHERE status IN ('pending', 'in_progress');
CREATE INDEX IF NOT EXISTS idx_review_tasks_filter
ON review_tasks(status, task_type, priority, evidence_id, updated_at);

CREATE TABLE IF NOT EXISTS review_audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_task_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    revision_before INTEGER NOT NULL,
    revision_after INTEGER NOT NULL,
    operator_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolution TEXT,
    dismiss_reason TEXT,
    FOREIGN KEY(review_task_id) REFERENCES review_tasks(review_task_id)
);
CREATE INDEX IF NOT EXISTS idx_review_audit_task
ON review_audit_events(review_task_id, event_id);
CREATE TRIGGER IF NOT EXISTS review_audit_no_update
BEFORE UPDATE ON review_audit_events BEGIN
    SELECT RAISE(ABORT, 'review audit events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS review_audit_no_delete
BEFORE DELETE ON review_audit_events BEGIN
    SELECT RAISE(ABORT, 'review audit events are immutable');
END;
"""


def fact_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _operator_token(value: str) -> str:
    if not _OPERATOR_RE.fullmatch(value) or _SECRET_RE.search(value):
        raise SchemaValidationError("操作员标识必须是 8～128 位脱敏令牌")
    return value


def _dump(value: BaseModel | None) -> str | None:
    return None if value is None else value.model_dump_json()


class ReviewStore:
    """SQLite 复核存储；状态变更使用 ``BEGIN IMMEDIATE`` 与 revision 乐观锁。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(REVIEW_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ReviewStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    @staticmethod
    def _stable_key(request: ReviewTaskCreate) -> str:
        payload = {
            "task_type": request.task_type.value,
            "evidence_id": request.evidence_id,
            "claim_id": request.claim_id,
            "field_name": request.field_name,
            "reason_codes": sorted(code.value for code in request.reason_codes),
            "input_fact_hash": request.audit_metadata.input_fact_hash,
            "trigger_version": request.audit_metadata.trigger_version,
        }
        return fact_hash(payload)

    def create(self, request: ReviewTaskCreate) -> tuple[ReviewTask, bool]:
        stable_key = self._stable_key(request)
        now = datetime.now(UTC).isoformat()
        with self._transaction():
            active = self._conn.execute(
                "SELECT * FROM review_tasks WHERE stable_key=? AND status IN ('pending','in_progress')",
                (stable_key,),
            ).fetchone()
            if active is not None:
                return self._row_to_task(active), False
            previous = self._conn.execute(
                "SELECT review_task_id, generation FROM review_tasks WHERE stable_key=? "
                "ORDER BY generation DESC LIMIT 1",
                (stable_key,),
            ).fetchone()
            generation = int(previous["generation"]) + 1 if previous else 1
            base_id = f"rvw_{stable_key[:24]}"
            task_id = base_id if generation == 1 else f"{base_id}-r{generation}"
            reopened_from = str(previous["review_task_id"]) if previous else None
            self._conn.execute(
                """
                INSERT INTO review_tasks (
                    review_task_id, stable_key, generation, task_type, reason_codes, status,
                    priority, evidence_id, claim_id, field_name, source,
                    current_normalized_value, created_at, updated_at, revision,
                    audit_metadata, reopened_from
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    task_id,
                    stable_key,
                    generation,
                    request.task_type.value,
                    json.dumps([item.value for item in request.reason_codes]),
                    request.priority.value,
                    request.evidence_id,
                    request.claim_id,
                    request.field_name,
                    _dump(request.source),
                    _dump(request.current_normalized_value),
                    now,
                    now,
                    request.audit_metadata.model_dump_json(),
                    reopened_from,
                ),
            )
            self._audit(
                task_id,
                "reopened" if reopened_from else "created",
                None,
                ReviewStatus.PENDING,
                0,
                1,
                "system:review-queue",
                now,
            )
            row = self._conn.execute(
                "SELECT * FROM review_tasks WHERE review_task_id=?", (task_id,)
            ).fetchone()
        assert row is not None
        return self._row_to_task(row), True

    def get(self, task_id: str) -> ReviewTask | None:
        row = self._conn.execute("SELECT * FROM review_tasks WHERE review_task_id=?", (task_id,)).fetchone()
        return None if row is None else self._row_to_task(row)

    def require(self, task_id: str) -> ReviewTask:
        task = self.get(task_id)
        if task is None:
            raise LookupError(f"复核任务不存在：{task_id}")
        return task

    def list_tasks(
        self,
        *,
        status: ReviewStatus | None = None,
        task_type: ReviewTaskType | None = None,
        priority: ReviewPriority | None = None,
        evidence_id: str | None = None,
    ) -> list[ReviewTask]:
        sql = "SELECT * FROM review_tasks WHERE 1=1"
        params: list[str] = []
        for column, value in (
            ("status", status.value if status else None),
            ("task_type", task_type.value if task_type else None),
            ("priority", priority.value if priority else None),
            ("evidence_id", evidence_id),
        ):
            if value is not None:
                sql += f" AND {column}=?"
                params.append(value)
        sql += " ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END, created_at, review_task_id"
        return [self._row_to_task(row) for row in self._conn.execute(sql, params).fetchall()]

    def start(self, task_id: str, *, expected_revision: int, operator_token: str) -> ReviewTask:
        return self._transition(
            task_id,
            expected_revision=expected_revision,
            operator_token=_operator_token(operator_token),
            allowed={ReviewStatus.PENDING},
            target=ReviewStatus.IN_PROGRESS,
            event_type="started",
        )

    def resolve(self, task_id: str, request: ReviewResolveRequest) -> ReviewTask:
        operator = _operator_token(request.resolver_token)
        task = self.require(task_id)
        correction = request.corrected_value
        if correction is not None:
            correction = ReviewValue(value=_validate_correction(task.field_name, correction.value))
        return self._transition(
            task_id,
            expected_revision=request.expected_revision,
            operator_token=operator,
            allowed={ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS},
            target=ReviewStatus.RESOLVED,
            event_type="resolved",
            correction=correction,
            resolution=request.resolution,
        )

    def dismiss(self, task_id: str, request: ReviewDismissRequest) -> ReviewTask:
        return self._transition(
            task_id,
            expected_revision=request.expected_revision,
            operator_token=_operator_token(request.resolver_token),
            allowed={ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS},
            target=ReviewStatus.DISMISSED,
            event_type="dismissed",
            dismiss_reason=request.reason,
        )

    def _transition(
        self,
        task_id: str,
        *,
        expected_revision: int,
        operator_token: str,
        allowed: set[ReviewStatus],
        target: ReviewStatus,
        event_type: Literal["started", "resolved", "dismissed"],
        correction: ReviewValue | None = None,
        resolution: ReviewResolution | None = None,
        dismiss_reason: ReviewDismissReason | None = None,
    ) -> ReviewTask:
        now = datetime.now(UTC).isoformat()
        with self._transaction():
            row = self._conn.execute(
                "SELECT * FROM review_tasks WHERE review_task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"复核任务不存在：{task_id}")
            current = ReviewStatus(row["status"])
            actual_revision = int(row["revision"])
            if actual_revision != expected_revision:
                raise VersionConflict(
                    "复核任务 revision 冲突",
                    detail={"expected_revision": expected_revision, "actual_revision": actual_revision},
                )
            if current not in allowed:
                raise VersionConflict(
                    f"非法状态迁移：{current.value} -> {target.value}",
                    detail={"task_id": task_id, "revision": actual_revision},
                )
            next_revision = actual_revision + 1
            cursor = self._conn.execute(
                """
                UPDATE review_tasks SET status=?, updated_at=?, resolved_at=?, resolver_token=?,
                    revision=?, manual_corrected_value=?, resolution=?, dismiss_reason=?
                WHERE review_task_id=? AND revision=?
                """,
                (
                    target.value,
                    now,
                    now if target in {ReviewStatus.RESOLVED, ReviewStatus.DISMISSED} else None,
                    operator_token,
                    next_revision,
                    _dump(correction),
                    resolution.value if resolution else None,
                    dismiss_reason.value if dismiss_reason else None,
                    task_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflict("复核任务已被其他操作更新")
            self._audit(
                task_id,
                event_type,
                current,
                target,
                actual_revision,
                next_revision,
                operator_token,
                now,
                resolution=resolution,
                dismiss_reason=dismiss_reason,
            )
            updated = self._conn.execute(
                "SELECT * FROM review_tasks WHERE review_task_id=?", (task_id,)
            ).fetchone()
        assert updated is not None
        return self._row_to_task(updated)

    def _audit(
        self,
        task_id: str,
        event_type: str,
        from_status: ReviewStatus | None,
        to_status: ReviewStatus,
        revision_before: int,
        revision_after: int,
        operator_token: str,
        created_at: str,
        *,
        resolution: ReviewResolution | None = None,
        dismiss_reason: ReviewDismissReason | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO review_audit_events (
                review_task_id, event_type, from_status, to_status, revision_before,
                revision_after, operator_token, created_at, resolution, dismiss_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                event_type,
                from_status.value if from_status else None,
                to_status.value,
                revision_before,
                revision_after,
                operator_token,
                created_at,
                resolution.value if resolution else None,
                dismiss_reason.value if dismiss_reason else None,
            ),
        )

    def audit_events(self, task_id: str) -> list[ReviewAuditEvent]:
        rows = self._conn.execute(
            "SELECT * FROM review_audit_events WHERE review_task_id=? ORDER BY event_id",
            (task_id,),
        ).fetchall()
        return [ReviewAuditEvent.model_validate(dict(row)) for row in rows]

    def export(self) -> dict[str, Any]:
        tasks = self.list_tasks()
        return {
            "schema_version": "1.0",
            "exported_at": datetime.now(UTC).isoformat(),
            "task_count": len(tasks),
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "audit_events": [
                event.model_dump(mode="json")
                for task in tasks
                for event in self.audit_events(task.review_task_id)
            ],
            "contains_secrets": False,
            "contains_prompt_or_model_response": False,
            "contains_raw_image": False,
        }

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> ReviewTask:
        def parsed(name: str) -> Any:
            return json.loads(row[name]) if row[name] else None

        return ReviewTask.model_validate(
            {
                **dict(row),
                "reason_codes": parsed("reason_codes"),
                "source": parsed("source"),
                "current_normalized_value": parsed("current_normalized_value"),
                "manual_corrected_value": parsed("manual_corrected_value"),
                "audit_metadata": parsed("audit_metadata"),
            }
        )


def _validate_correction(field_name: str | None, value: ReviewScalar) -> ReviewScalar:
    if field_name in _SENSITIVE_FIELDS:
        raise SchemaValidationError("复核队列不得持久化未脱敏姓名或学号修正值")
    if isinstance(value, str):
        cleaned = _validate_safe_text(value, label="人工修正值")
        if field_name == "获奖日期":
            normalized = cleaned.replace("年", "-").replace("月", "-").replace("日", "")
            normalized = normalized.replace("/", "-").replace(".", "-")
            try:
                parts = normalized.split("-")
                if len(parts) != 3:
                    raise ValueError
                return date(*(int(part) for part in parts)).isoformat()
            except (TypeError, ValueError) as exc:
                raise SchemaValidationError("人工修正的获奖日期不可解析") from exc
        if field_name == "级别":
            normalized_tier = parse_tier(cleaned)
            if normalized_tier is None:
                raise SchemaValidationError("人工修正的级别不在允许枚举/归一化范围")
            return normalized_tier
        if field_name == "奖项/名次":
            normalized_prize = parse_prize(cleaned)
            if normalized_prize is None:
                raise SchemaValidationError("人工修正的奖项/名次不在允许枚举/归一化范围")
            return normalized_prize
        if field_name == "团队属性":
            if cleaned in {"团队", "团体", "集体"}:
                return "团队"
            if cleaned in {"个人", "单人"}:
                return "个人"
            raise SchemaValidationError("人工修正的团队属性只能是个人或团队")
        return cleaned
    if field_name == "团队属性" and isinstance(value, bool):
        return "团队" if value else "个人"
    return value


def _metadata(
    component: Literal["certificate", "vlm", "dedup", "consistency", "manual"],
    payload: Any,
    *,
    source_event_id: str | None = None,
) -> ReviewAuditMetadata:
    return ReviewAuditMetadata(
        source_component=component,
        input_fact_hash=fact_hash(payload),
        trigger_version="review-workflow-v1",
        source_event_id=source_event_id,
    )


def enqueue_certificate_reviews(store: ReviewStore, result: CertificatePipelineResult) -> list[ReviewTask]:
    tasks: list[ReviewTask] = []
    for field_name, field in result.extraction.fields.by_name().items():
        if field.normalized_value is not None and field.confidence >= result.extraction.confidence_threshold:
            continue
        error_reasons = {
            "unsupported_by_ocr": ReviewReasonCode.UNSUPPORTED_BY_OCR,
            "invalid_format_or_enum": ReviewReasonCode.INVALID_FORMAT_OR_ENUM,
            "multi_source_disagreement": ReviewReasonCode.MULTI_SOURCE_DISAGREEMENT,
        }
        reasons = [error_reasons[item] for item in field.validation_errors if item in error_reasons]
        if field.confidence < result.extraction.confidence_threshold:
            reasons.append(ReviewReasonCode.LOW_CONFIDENCE)
        task_type = (
            ReviewTaskType.OCR_UNSUPPORTED
            if ReviewReasonCode.UNSUPPORTED_BY_OCR in reasons
            else ReviewTaskType.LOW_CONFIDENCE_FIELD
        )
        request = ReviewTaskCreate(
            task_type=task_type,
            reason_codes=reasons or [ReviewReasonCode.LOW_CONFIDENCE],
            priority=ReviewPriority.HIGH if field.normalized_value is None else ReviewPriority.NORMAL,
            evidence_id=result.evidence.id,
            field_name=field_name,
            source=ReviewSource(
                locator=result.evidence.source_locator,
                bbox=field.location.bboxes[0] if field.location.bboxes else None,
            ),
            current_normalized_value=(
                None
                if field_name in _SENSITIVE_FIELDS or field.normalized_value is None
                else ReviewValue(value=field.normalized_value)
            ),
            audit_metadata=_metadata(
                "certificate",
                {
                    "evidence_id": result.evidence.id,
                    "field": field_name,
                    "confidence": field.confidence,
                    "errors": field.validation_errors,
                },
            ),
        )
        tasks.append(store.create(request)[0])
    vlm_reason = {
        "provider_missing": ReviewReasonCode.VLM_PROVIDER_MISSING,
        "key_missing": ReviewReasonCode.VLM_KEY_MISSING,
        "crop_missing": ReviewReasonCode.VLM_CROP_MISSING,
        "failed": ReviewReasonCode.VLM_CALL_FAILED,
    }.get(result.extraction.vlm.status)
    if vlm_reason is not None:
        request = ReviewTaskCreate(
            task_type=(
                ReviewTaskType.VLM_CALL_FAILED
                if result.extraction.vlm.status == "failed"
                else ReviewTaskType.VLM_NOT_CONFIGURED
            ),
            reason_codes=[vlm_reason],
            priority=ReviewPriority.HIGH,
            evidence_id=result.evidence.id,
            audit_metadata=_metadata(
                "vlm",
                {
                    "evidence_id": result.evidence.id,
                    "status": result.extraction.vlm.status,
                    "provider": result.extraction.vlm.provider,
                    "fields": result.extraction.vlm.low_confidence_fields,
                    "called": result.extraction.vlm.called,
                },
            ),
        )
        tasks.append(store.create(request)[0])
    if result.quality.needs_retake:
        request = ReviewTaskCreate(
            task_type=ReviewTaskType.INSUFFICIENT_INFORMATION,
            reason_codes=[ReviewReasonCode.IMAGE_QUALITY],
            priority=ReviewPriority.HIGH,
            evidence_id=result.evidence.id,
            audit_metadata=_metadata(
                "certificate",
                {"evidence_id": result.evidence.id, "quality": result.quality.model_dump(mode="json")},
            ),
        )
        tasks.append(store.create(request)[0])
    return list({task.review_task_id: task for task in tasks}.values())


def enqueue_duplicate_review(store: ReviewStore, decision: DuplicateDecision) -> list[ReviewTask]:
    if not decision.flagged:
        return []
    reason = (
        ReviewReasonCode.DUPLICATE_CONFIRMED
        if decision.status == "确定重复"
        else ReviewReasonCode.DUPLICATE_SUSPECTED
    )
    request = ReviewTaskCreate(
        task_type=ReviewTaskType.POSSIBLE_DUPLICATE,
        reason_codes=[reason],
        priority=ReviewPriority.CRITICAL if decision.status == "确定重复" else ReviewPriority.HIGH,
        evidence_id=decision.left_id,
        audit_metadata=_metadata(
            "dedup",
            {
                "left_id": decision.left_id,
                "right_id": decision.right_id,
                "status": decision.status,
                "sha256_match": decision.sha256_match,
                "phash_distance": decision.phash_distance,
                "fact_fingerprint_match": decision.fact_fingerprint_match,
            },
            source_event_id=decision.right_id,
        ),
    )
    return [store.create(request)[0]]


def enqueue_consistency_reviews(store: ReviewStore, report: EvidenceConsistencyReport) -> list[ReviewTask]:
    if not report.requires_review:
        return []
    tasks: list[ReviewTask] = []
    for item in report.fields:
        if item.status not in {"不一致", "信息不足"}:
            continue
        mismatch = item.status == "不一致"
        current = item.evidence_normalized or item.evidence_value
        request = ReviewTaskCreate(
            task_type=(
                ReviewTaskType.CLAIM_EVIDENCE_CONFLICT
                if mismatch
                else ReviewTaskType.INSUFFICIENT_INFORMATION
            ),
            reason_codes=[
                ReviewReasonCode.CLAIM_EVIDENCE_MISMATCH
                if mismatch
                else ReviewReasonCode.CLAIM_EVIDENCE_INSUFFICIENT
            ],
            priority=ReviewPriority.HIGH if mismatch else ReviewPriority.NORMAL,
            evidence_id=report.evidence_id,
            claim_id=report.claim_id,
            field_name=item.field,
            current_normalized_value=(
                None if item.field in _SENSITIVE_FIELDS or current is None else ReviewValue(value=current)
            ),
            audit_metadata=_metadata(
                "consistency",
                {
                    "claim_id": report.claim_id,
                    "evidence_id": report.evidence_id,
                    "field": item.field,
                    "status": item.status,
                    "claim_normalized": None if item.field in _SENSITIVE_FIELDS else item.claim_normalized,
                    "evidence_normalized": None
                    if item.field in _SENSITIVE_FIELDS
                    else item.evidence_normalized,
                },
            ),
        )
        tasks.append(store.create(request)[0])
    return tasks


__all__ = [
    "REVIEW_SCHEMA_SQL",
    "ReviewAuditEvent",
    "ReviewAuditMetadata",
    "ReviewDismissReason",
    "ReviewDismissRequest",
    "ReviewPriority",
    "ReviewReasonCode",
    "ReviewResolution",
    "ReviewResolveRequest",
    "ReviewSource",
    "ReviewStartRequest",
    "ReviewStatus",
    "ReviewStore",
    "ReviewTask",
    "ReviewTaskCreate",
    "ReviewTaskType",
    "ReviewValue",
    "ReviewWorkflowSmokeReport",
    "enqueue_certificate_reviews",
    "enqueue_consistency_reviews",
    "enqueue_duplicate_review",
    "fact_hash",
]
