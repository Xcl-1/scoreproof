"""可持久化批量核算任务与可追溯导出。

批任务只负责编排确定性计算：输入、规则快照标识和结果均写入 SQLite，
数值仍全部由 :mod:`scoreproof.calc` 计算。终态任务与审计事件由数据库触发器保护，
避免导出后被静默改写。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calc import EngineConfig
from .errors import SchemaValidationError, VersionConflict
from .schema import Claim, ScoreBreakdown


class BatchStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BatchRunConfig(BaseModel):
    """批量入口允许调用者调整的确定性引擎开关。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strict_year: bool = True
    fuzzy_fallback: bool = True

    def to_engine_config(self) -> EngineConfig:
        return EngineConfig(strict_year=self.strict_year, fuzzy_fallback=self.fuzzy_fallback)


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[Claim] = Field(min_length=1, max_length=5000)
    academic_year: str | None = None
    college: str | None = None
    config: BatchRunConfig = Field(default_factory=BatchRunConfig)

    @model_validator(mode="after")
    def _unique_claim_ids(self) -> BatchInput:
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("批任务中的 claim id 必须唯一")
        return self


class BatchTask(BaseModel):
    """不包含申报正文的任务元数据，可安全用于列表页。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    status: BatchStatus
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_version_id: str
    rule_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_count: int = Field(ge=0)
    claim_count: int = Field(ge=1)
    student_count: int = Field(ge=1)
    academic_year: str | None = None
    college: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_code: str | None = None
    error_message: str | None = None


class BatchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(ge=1)
    batch_id: str
    event_type: Literal["created", "started", "succeeded", "failed"]
    from_status: BatchStatus | None = None
    to_status: BatchStatus
    created_at: datetime
    error_code: str | None = None


class BatchArtifact(BaseModel):
    """JSON 导出物；同时保留提交值、核算后状态和逐学生账本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    batch_id: str
    rule_version_id: str
    rule_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calculated_at: datetime
    input: BatchInput
    evaluated_claims: list[Claim]
    results: dict[str, ScoreBreakdown]

    @model_validator(mode="after")
    def _traceability_contract(self) -> BatchArtifact:
        submitted = {claim.id: claim for claim in self.input.claims}
        evaluated = {claim.id: claim for claim in self.evaluated_claims}
        if submitted.keys() != evaluated.keys():
            raise ValueError("核算后申报集合必须与批任务输入一致")
        for claim_id, original in submitted.items():
            before = original.model_dump(mode="json", exclude={"status"})
            after = evaluated[claim_id].model_dump(mode="json", exclude={"status"})
            if before != after:
                raise ValueError("核算只允许更新申报 status，不得改写业务输入")
        expected_students = {claim.student_id for claim in self.evaluated_claims}
        if set(self.results) != expected_students:
            raise ValueError("逐学生账本必须完整覆盖批任务输入")
        for student_id, breakdown in self.results.items():
            if breakdown.student_id != student_id:
                raise ValueError("账本键与 student_id 不一致")
            valid_claim_ids = {
                claim.id for claim in self.evaluated_claims if claim.student_id == student_id
            }
            if any(match.claim_id not in valid_claim_ids for match in breakdown.matches):
                raise ValueError("账本包含不属于该学生的 claim id")
        return self


class BatchDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task: BatchTask
    artifact: BatchArtifact | None = None
    events: list[BatchEvent] = Field(default_factory=list)
    created: bool | None = None


BATCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS score_batches (
    batch_id TEXT PRIMARY KEY,
    idempotency_key_hash TEXT UNIQUE,
    status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'succeeded', 'failed')),
    input_hash TEXT NOT NULL,
    rule_version_id TEXT NOT NULL,
    rule_content_hash TEXT NOT NULL,
    rule_count INTEGER NOT NULL,
    claim_count INTEGER NOT NULL,
    student_count INTEGER NOT NULL,
    academic_year TEXT,
    college TEXT,
    request_json TEXT NOT NULL,
    result_json TEXT,
    result_hash TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    CHECK(
        (status = 'succeeded' AND result_json IS NOT NULL AND result_hash IS NOT NULL
            AND completed_at IS NOT NULL AND error_code IS NULL)
        OR (status = 'failed' AND result_json IS NULL AND result_hash IS NULL
            AND completed_at IS NOT NULL AND error_code IS NOT NULL)
        OR (status IN ('queued', 'running') AND result_json IS NULL
            AND result_hash IS NULL AND completed_at IS NULL AND error_code IS NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_score_batches_created
ON score_batches(created_at DESC, batch_id DESC);

CREATE TABLE IF NOT EXISTS score_batch_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('created', 'started', 'succeeded', 'failed')),
    from_status TEXT,
    to_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    error_code TEXT,
    FOREIGN KEY(batch_id) REFERENCES score_batches(batch_id)
);
CREATE INDEX IF NOT EXISTS idx_score_batch_events_batch
ON score_batch_events(batch_id, event_id);

CREATE TRIGGER IF NOT EXISTS score_batches_terminal_no_update
BEFORE UPDATE ON score_batches
WHEN OLD.status IN ('succeeded', 'failed') BEGIN
    SELECT RAISE(ABORT, 'terminal score batches are immutable');
END;
CREATE TRIGGER IF NOT EXISTS score_batches_no_delete
BEFORE DELETE ON score_batches BEGIN
    SELECT RAISE(ABORT, 'score batches are immutable');
END;
CREATE TRIGGER IF NOT EXISTS score_batches_valid_transition
BEFORE UPDATE OF status ON score_batches
WHEN NOT (
    (OLD.status = 'queued' AND NEW.status IN ('running', 'failed'))
    OR (OLD.status = 'running' AND NEW.status IN ('succeeded', 'failed'))
) BEGIN
    SELECT RAISE(ABORT, 'invalid score batch transition');
END;
CREATE TRIGGER IF NOT EXISTS score_batch_events_no_update
BEFORE UPDATE ON score_batch_events BEGIN
    SELECT RAISE(ABORT, 'score batch events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS score_batch_events_no_delete
BEFORE DELETE ON score_batch_events BEGIN
    SELECT RAISE(ABORT, 'score batch events are immutable');
END;
"""


_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")


def canonical_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _idempotency_hash(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not _IDEMPOTENCY_RE.fullmatch(cleaned):
        raise SchemaValidationError("Idempotency-Key 必须是 8～128 位字母、数字或 ._:-")
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


class BatchStore:
    """SQLite 批任务存储；所有状态迁移均使用立即事务。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(BATCH_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> BatchStore:
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

    def create(
        self,
        batch_input: BatchInput,
        *,
        rule_version_id: str,
        rule_content_hash: str,
        rule_count: int,
        idempotency_key: str | None = None,
    ) -> tuple[BatchTask, bool]:
        key_hash = _idempotency_hash(idempotency_key)
        input_hash = canonical_hash(batch_input)
        now = datetime.now(UTC).isoformat()
        request_json = batch_input.model_dump_json()
        with self._transaction():
            if key_hash is not None:
                existing = self._conn.execute(
                    "SELECT * FROM score_batches WHERE idempotency_key_hash=?", (key_hash,)
                ).fetchone()
                if existing is not None:
                    if (
                        existing["input_hash"] != input_hash
                        or existing["rule_content_hash"] != rule_content_hash
                    ):
                        raise VersionConflict(
                            "Idempotency-Key 已绑定到不同输入或规则版本",
                            detail={"batch_id": existing["batch_id"]},
                        )
                    return self._row_to_task(existing), False
            batch_id = f"bat_{uuid.uuid4().hex}"
            student_count = len({claim.student_id for claim in batch_input.claims})
            self._conn.execute(
                """
                INSERT INTO score_batches (
                    batch_id, idempotency_key_hash, status, input_hash,
                    rule_version_id, rule_content_hash, rule_count, claim_count,
                    student_count, academic_year, college, request_json, created_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    key_hash,
                    input_hash,
                    rule_version_id,
                    rule_content_hash,
                    rule_count,
                    len(batch_input.claims),
                    student_count,
                    batch_input.academic_year,
                    batch_input.college,
                    request_json,
                    now,
                ),
            )
            self._event(batch_id, "created", None, BatchStatus.QUEUED, now)
            row = self._require_row(batch_id)
        return self._row_to_task(row), True

    def start(self, batch_id: str) -> BatchTask:
        now = datetime.now(UTC).isoformat()
        with self._transaction():
            row = self._require_row(batch_id)
            if row["status"] != BatchStatus.QUEUED.value:
                raise VersionConflict(
                    "批任务只能从 queued 进入 running",
                    detail={"batch_id": batch_id, "status": row["status"]},
                )
            self._conn.execute(
                "UPDATE score_batches SET status='running', started_at=? WHERE batch_id=?",
                (now, batch_id),
            )
            self._event(
                batch_id, "started", BatchStatus.QUEUED, BatchStatus.RUNNING, now
            )
            updated = self._require_row(batch_id)
        return self._row_to_task(updated)

    def complete(self, batch_id: str, artifact: BatchArtifact) -> BatchTask:
        if artifact.batch_id != batch_id:
            raise SchemaValidationError("批任务结果与 batch_id 不一致")
        payload = artifact.model_dump(mode="json")
        result_hash = canonical_hash(payload)
        result_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        now = datetime.now(UTC).isoformat()
        with self._transaction():
            row = self._require_row(batch_id)
            if row["status"] != BatchStatus.RUNNING.value:
                raise VersionConflict(
                    "批任务只能从 running 进入 succeeded",
                    detail={"batch_id": batch_id, "status": row["status"]},
                )
            if (
                artifact.rule_version_id != row["rule_version_id"]
                or artifact.rule_content_hash != row["rule_content_hash"]
                or canonical_hash(artifact.input) != row["input_hash"]
            ):
                raise SchemaValidationError("批任务结果与已锁定的输入或规则快照不一致")
            self._conn.execute(
                """
                UPDATE score_batches SET status='succeeded', result_json=?, result_hash=?,
                    completed_at=? WHERE batch_id=?
                """,
                (result_json, result_hash, now, batch_id),
            )
            self._event(
                batch_id, "succeeded", BatchStatus.RUNNING, BatchStatus.SUCCEEDED, now
            )
            updated = self._require_row(batch_id)
        return self._row_to_task(updated)

    def fail(self, batch_id: str, *, error_code: str, error_message: str) -> BatchTask:
        code = error_code.strip()[:80] or "batch_failed"
        message = error_message.strip().replace("\x00", "")[:500] or "批任务执行失败"
        now = datetime.now(UTC).isoformat()
        with self._transaction():
            row = self._require_row(batch_id)
            current = BatchStatus(row["status"])
            if current not in {BatchStatus.QUEUED, BatchStatus.RUNNING}:
                raise VersionConflict(
                    "终态批任务不可再次失败",
                    detail={"batch_id": batch_id, "status": current.value},
                )
            self._conn.execute(
                """
                UPDATE score_batches SET status='failed', error_code=?, error_message=?,
                    completed_at=? WHERE batch_id=?
                """,
                (code, message, now, batch_id),
            )
            self._event(batch_id, "failed", current, BatchStatus.FAILED, now, error_code=code)
            updated = self._require_row(batch_id)
        return self._row_to_task(updated)

    def get_task(self, batch_id: str) -> BatchTask | None:
        row = self._conn.execute(
            "SELECT * FROM score_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        return None if row is None else self._row_to_task(row)

    def get_input(self, batch_id: str) -> BatchInput | None:
        row = self._conn.execute(
            "SELECT request_json FROM score_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        return None if row is None else BatchInput.model_validate_json(row["request_json"])

    def get_artifact(self, batch_id: str) -> BatchArtifact | None:
        row = self._conn.execute(
            "SELECT result_json FROM score_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if row is None or row["result_json"] is None:
            return None
        return BatchArtifact.model_validate_json(row["result_json"])

    def detail(self, batch_id: str, *, created: bool | None = None) -> BatchDetail | None:
        task = self.get_task(batch_id)
        if task is None:
            return None
        return BatchDetail(
            task=task,
            artifact=self.get_artifact(batch_id),
            events=self.events(batch_id),
            created=created,
        )

    def list_tasks(self, *, limit: int = 50, offset: int = 0) -> list[BatchTask]:
        rows = self._conn.execute(
            "SELECT * FROM score_batches ORDER BY created_at DESC, batch_id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def events(self, batch_id: str) -> list[BatchEvent]:
        rows = self._conn.execute(
            "SELECT * FROM score_batch_events WHERE batch_id=? ORDER BY event_id", (batch_id,)
        ).fetchall()
        return [BatchEvent.model_validate(dict(row)) for row in rows]

    def _require_row(self, batch_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM score_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"批任务不存在：{batch_id}")
        return row

    def _event(
        self,
        batch_id: str,
        event_type: str,
        from_status: BatchStatus | None,
        to_status: BatchStatus,
        created_at: str,
        *,
        error_code: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO score_batch_events (
                batch_id, event_type, from_status, to_status, created_at, error_code
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                event_type,
                from_status.value if from_status else None,
                to_status.value,
                created_at,
                error_code,
            ),
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> BatchTask:
        return BatchTask.model_validate(
            {key: row[key] for key in BatchTask.model_fields}
        )


CSV_EXPORT_FIELDS = (
    "batch_id",
    "rule_version_id",
    "rule_content_hash",
    "student_id",
    "student_name",
    "student_total",
    "claim_id",
    "category",
    "raw_text",
    "level",
    "claim_status",
    "rule_id",
    "matched_key",
    "raw_score",
    "score",
    "counted",
    "needs_review",
    "channel",
    "reason",
    "source_doc",
    "source_page",
    "source_table",
    "source_row",
)


def _csv_safe(value: Any) -> Any:
    """阻止 Excel/表格软件把不可信文本当作公式执行。"""
    if not isinstance(value, str):
        return value
    return f"'{value}" if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def artifact_to_csv(artifact: BatchArtifact) -> bytes:
    claims = {claim.id: claim for claim in artifact.evaluated_claims}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for student_id, breakdown in sorted(artifact.results.items()):
        for match in breakdown.matches:
            claim = claims.get(match.claim_id)
            source = match.source
            row = {
                "batch_id": artifact.batch_id,
                "rule_version_id": artifact.rule_version_id,
                "rule_content_hash": artifact.rule_content_hash,
                "student_id": student_id,
                "student_name": claim.student_name if claim else None,
                "student_total": breakdown.total,
                "claim_id": match.claim_id,
                "category": claim.category if claim else None,
                "raw_text": claim.raw_text if claim else None,
                "level": claim.level if claim else None,
                "claim_status": claim.status if claim else None,
                "rule_id": match.rule_id,
                "matched_key": match.matched_key,
                "raw_score": match.raw_score,
                "score": match.score,
                "counted": match.counted,
                "needs_review": match.needs_review,
                "channel": match.channel,
                "reason": match.reason,
                "source_doc": source.doc if source else None,
                "source_page": source.page if source else None,
                "source_table": source.table if source else None,
                "source_row": source.row if source else None,
            }
            writer.writerow({key: _csv_safe(value) for key, value in row.items()})
    return b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8")


__all__ = [
    "BATCH_SCHEMA_SQL",
    "CSV_EXPORT_FIELDS",
    "BatchArtifact",
    "BatchDetail",
    "BatchEvent",
    "BatchInput",
    "BatchRunConfig",
    "BatchStatus",
    "BatchStore",
    "BatchTask",
    "artifact_to_csv",
    "canonical_hash",
]
