"""规则库持久化：SQLite + JSON 双向。

为什么是 SQLite：规则库是**结构化数据**，需要按学年/学院/类别/等级精确查表，
天然适合关系表；同时导出 JSON 便于人工校对与 diff（也是简历里的可验证产物）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import SchemaValidationError, VersionConflict
from ..schema import Rule, Ruleset, stable_rule_id

if TYPE_CHECKING:
    from .gateway import GatewayReport

_EXPECTED_VERSION_UNSET = object()

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS rules (
    id            TEXT PRIMARY KEY,
    academic_year TEXT NOT NULL,
    college       TEXT,
    category      TEXT NOT NULL,
    level         TEXT NOT NULL,
    rank          TEXT,
    item_name     TEXT,
    score         REAL NOT NULL,
    synonyms      TEXT NOT NULL DEFAULT '[]',
    constraints   TEXT NOT NULL DEFAULT '{}',
    source        TEXT NOT NULL DEFAULT '{}',
    priority      INTEGER NOT NULL DEFAULT 0,
    enabled       INTEGER NOT NULL DEFAULT 1,
    raw_text      TEXT,
    extraction_batch_id TEXT,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_rules_lookup ON rules (academic_year, college, category, level);
CREATE INDEX IF NOT EXISTS idx_rules_group  ON rules (category);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    id            TEXT PRIMARY KEY,
    student_id    TEXT NOT NULL,
    academic_year TEXT,
    college       TEXT,
    category      TEXT,
    raw_text      TEXT,
    level         TEXT,
    team          INTEGER NOT NULL DEFAULT 0,
    evidence_ids  TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT '待核对',
    payload       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_claims_student ON claims (student_id, academic_year);

CREATE TABLE IF NOT EXISTS extraction_batches (
    id            TEXT PRIMARY KEY,
    chunk_hash    TEXT NOT NULL,
    academic_year TEXT NOT NULL,
    college       TEXT,
    doc           TEXT NOT NULL,
    model         TEXT,
    status        TEXT NOT NULL,
    secondary_used INTEGER NOT NULL DEFAULT 0,
    metrics       TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_audits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      TEXT NOT NULL,
    item_index    INTEGER NOT NULL,
    status        TEXT NOT NULL,
    raw_payload   TEXT NOT NULL,
    issues        TEXT NOT NULL DEFAULT '[]',
    char_start    INTEGER,
    char_end      INTEGER,
    ambiguity_flag INTEGER NOT NULL DEFAULT 0,
    extract_confidence REAL NOT NULL DEFAULT 0,
    manual_corrected INTEGER NOT NULL DEFAULT 0,
    corrected_payload TEXT,
    UNIQUE(batch_id, item_index),
    FOREIGN KEY(batch_id) REFERENCES extraction_batches(id)
);
CREATE INDEX IF NOT EXISTS idx_extraction_audits_batch ON extraction_audits (batch_id);

CREATE TABLE IF NOT EXISTS extraction_cache (
    chunk_hash    TEXT NOT NULL,
    model         TEXT NOT NULL,
    variant       TEXT NOT NULL,
    payloads      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY(chunk_hash, model, variant)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    tool_name      TEXT NOT NULL,
    input_json     TEXT NOT NULL,
    result_summary TEXT NOT NULL,
    duration_ms    REAL NOT NULL,
    model_version  TEXT NOT NULL,
    status         TEXT NOT NULL,
    error          TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls (session_id, created_at);

CREATE TABLE IF NOT EXISTS rule_versions (
    id                TEXT PRIMARY KEY,
    content_hash      TEXT NOT NULL UNIQUE,
    parent_version_id TEXT,
    rule_count        INTEGER NOT NULL,
    snapshot_json     TEXT NOT NULL,
    source_kind       TEXT NOT NULL,
    source_ref        TEXT,
    note              TEXT,
    created_by        TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    FOREIGN KEY(parent_version_id) REFERENCES rule_versions(id)
);
CREATE INDEX IF NOT EXISTS idx_rule_versions_created
    ON rule_versions(created_at, id);
CREATE TRIGGER IF NOT EXISTS rule_versions_no_update
BEFORE UPDATE ON rule_versions BEGIN
    SELECT RAISE(ABORT, 'rule versions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS rule_versions_no_delete
BEFORE DELETE ON rule_versions BEGIN
    SELECT RAISE(ABORT, 'rule versions are immutable');
END;

CREATE TABLE IF NOT EXISTS rule_version_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    action              TEXT NOT NULL,
    version_id          TEXT NOT NULL,
    previous_version_id TEXT,
    actor               TEXT NOT NULL,
    note                TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY(version_id) REFERENCES rule_versions(id),
    FOREIGN KEY(previous_version_id) REFERENCES rule_versions(id)
);
CREATE INDEX IF NOT EXISTS idx_rule_version_events_version
    ON rule_version_events(version_id, id);
CREATE TRIGGER IF NOT EXISTS rule_version_events_no_update
BEFORE UPDATE ON rule_version_events BEGIN
    SELECT RAISE(ABORT, 'rule version events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS rule_version_events_no_delete
BEFORE DELETE ON rule_version_events BEGIN
    SELECT RAISE(ABORT, 'rule version events are immutable');
END;
"""


class RuleVersionRecord(BaseModel):
    """不可变规则快照的元数据；是否活动由 ``meta`` 指针实时计算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_version_id: str | None = None
    rule_count: int = Field(ge=0)
    source_kind: str
    source_ref: str | None = None
    note: str | None = None
    created_by: str
    created_at: datetime
    active: bool = False


class RuleVersionEvent(BaseModel):
    """只追加的发布/回滚审计事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(ge=1)
    action: Literal["bootstrap", "publish", "rollback"]
    version_id: str
    previous_version_id: str | None = None
    actor: str
    note: str | None = None
    created_at: datetime


class RuleVersionResult(BaseModel):
    """规则快照切换结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["published", "unchanged", "rolled_back"]
    version_id: str
    previous_version_id: str | None = None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_count: int = Field(ge=0)


class RuleStore:
    """规则库存储。线程内复用连接，跨线程请各自实例化。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    # ---------- 生命周期 ----------

    def init_db(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)
            # 兼容已由旧版本创建的本地规则库；SQLite 的 IF NOT EXISTS 不会补列。
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(rules)")}
            migrations = {
                "rank": "ALTER TABLE rules ADD COLUMN rank TEXT",
                "item_name": "ALTER TABLE rules ADD COLUMN item_name TEXT",
                "extraction_batch_id": "ALTER TABLE rules ADD COLUMN extraction_batch_id TEXT",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    self._conn.execute(statement)
            audit_columns = {
                row[1] for row in self._conn.execute("PRAGMA table_info(extraction_audits)")
            }
            audit_migrations = {
                "manual_corrected": (
                    "ALTER TABLE extraction_audits ADD COLUMN manual_corrected INTEGER "
                    "NOT NULL DEFAULT 0"
                ),
                "corrected_payload": (
                    "ALTER TABLE extraction_audits ADD COLUMN corrected_payload TEXT"
                ),
            }
            for column, statement in audit_migrations.items():
                if column not in audit_columns:
                    self._conn.execute(statement)
            batch_columns = {
                row[1] for row in self._conn.execute("PRAGMA table_info(extraction_batches)")
            }
            if "secondary_used" not in batch_columns:
                self._conn.execute(
                    "ALTER TABLE extraction_batches ADD COLUMN secondary_used INTEGER "
                    "NOT NULL DEFAULT 0"
                )
            self._bootstrap_rule_version()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RuleStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """使用立即事务串行化活动版本切换，异常时恢复规则和版本指针。"""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def _bootstrap_rule_version(self) -> None:
        """为旧数据库中已有的原地规则建立一次不可变基线，不改动规则内容。"""
        if self._active_rule_version_id() is not None or self.count() == 0:
            return
        entries = self._current_snapshot_entries()
        version_id, digest = self._ensure_rule_version(
            entries,
            parent_version_id=None,
            source_kind="migration",
            source_ref=None,
            note="为旧规则库建立版本化基线",
            actor="scoreproof:migration",
        )
        now = datetime.now().isoformat()
        self._set_active_rule_version_id(version_id)
        self._append_rule_version_event(
            action="bootstrap",
            version_id=version_id,
            previous_version_id=None,
            actor="scoreproof:migration",
            note=f"legacy-content-sha256:{digest}",
            created_at=now,
        )

    # ---------- 写入 ----------

    def upsert_rules(
        self,
        rules: list[Rule],
        *,
        actor: str = "scoreproof:store",
        source_kind: str = "manual",
        source_ref: str | None = None,
        note: str | None = None,
    ) -> int:
        current = {entry["rule"]["id"]: entry for entry in self._current_snapshot_entries()}
        for rule in rules:
            identity_id = stable_rule_id(
                academic_year=rule.academic_year,
                college=rule.college,
                category=rule.category,
                level=rule.level,
                rank=rule.rank,
                item_name=rule.item_name,
                source=rule.source,
            )
            matching_ids = [
                current_id
                for current_id, entry in current.items()
                if self._stable_identity_for_entry(entry) == identity_id
            ]
            previous_entries = [current.pop(current_id) for current_id in matching_ids]
            previous = next(
                (entry for entry in previous_entries if entry["extraction_batch_id"]),
                previous_entries[0] if previous_entries else None,
            )
            current[rule.id] = self._snapshot_entry(
                rule,
                extraction_batch_id=(previous["extraction_batch_id"] if previous else None),
            )
        self.publish_rule_snapshot(
            [entry["rule"] for entry in current.values()],
            extraction_batch_ids={
                rule_id: entry["extraction_batch_id"] for rule_id, entry in current.items()
            },
            actor=actor,
            source_kind=source_kind,
            source_ref=source_ref,
            note=note,
            expected_active_version=self._active_rule_version_id(),
        )
        return len(rules)

    def delete_rule(self, rule_id: str) -> bool:
        entries = self._current_snapshot_entries()
        remaining = [entry for entry in entries if entry["rule"]["id"] != rule_id]
        if len(remaining) == len(entries):
            return False
        self.publish_rule_snapshot(
            [entry["rule"] for entry in remaining],
            extraction_batch_ids={
                entry["rule"]["id"]: entry["extraction_batch_id"] for entry in remaining
            },
            actor="scoreproof:store",
            source_kind="delete",
            source_ref=rule_id,
            note=f"删除规则 {rule_id}",
            expected_active_version=self._active_rule_version_id(),
        )
        return True

    # ---------- 不可变规则版本 ----------

    def publish_rule_snapshot(
        self,
        rules: Sequence[Rule | dict[str, Any]],
        *,
        actor: str,
        source_kind: str,
        source_ref: str | None = None,
        note: str | None = None,
        expected_active_version: str | None | object = _EXPECTED_VERSION_UNSET,
        extraction_batch_ids: dict[str, str | None] | None = None,
    ) -> RuleVersionResult:
        """将完整规则快照与活动指针在同一事务中发布。

        ``expected_active_version`` 用于跨进程乐观锁；不传表示接受当前活动版本。
        快照按内容寻址且不可修改，重复发布相同活动内容不会追加审计噪声。
        """
        actor = actor.strip()
        source_kind = source_kind.strip()
        if not actor or not source_kind:
            raise ValueError("actor 与 source_kind 不能为空")
        batch_ids = extraction_batch_ids or {}
        validated_rules = [Rule.model_validate(rule) for rule in rules]
        entries = [
            self._snapshot_entry(
                rule,
                extraction_batch_id=batch_ids.get(rule.id),
            )
            for rule in validated_rules
        ]
        Ruleset(rules=[Rule.model_validate(entry["rule"]) for entry in entries])
        entries.sort(key=lambda entry: str(entry["rule"]["id"]))

        with self._transaction():
            previous_id = self._active_rule_version_id()
            if (
                expected_active_version is not _EXPECTED_VERSION_UNSET
                and previous_id != expected_active_version
            ):
                raise VersionConflict(
                    "活动规则版本已变化，拒绝覆盖较新的发布",
                    detail={
                        "expected_active_version": expected_active_version,
                        "active_version": previous_id,
                    },
                )
            digest = self._rule_snapshot_hash(entries)
            existing = self._conn.execute(
                "SELECT id, rule_count FROM rule_versions WHERE content_hash=?", (digest,)
            ).fetchone()
            if existing is not None and existing["id"] == previous_id:
                return RuleVersionResult(
                    action="unchanged",
                    version_id=str(existing["id"]),
                    previous_version_id=previous_id,
                    content_hash=digest,
                    rule_count=int(existing["rule_count"]),
                )
            version_id, digest = self._ensure_rule_version(
                entries,
                parent_version_id=previous_id,
                source_kind=source_kind,
                source_ref=source_ref,
                note=note,
                actor=actor,
            )
            self._replace_live_rules(entries)
            self._before_rule_version_activation(version_id)
            self._set_active_rule_version_id(version_id)
            self._append_rule_version_event(
                action="publish",
                version_id=version_id,
                previous_version_id=previous_id,
                actor=actor,
                note=note,
            )
        return RuleVersionResult(
            action="published",
            version_id=version_id,
            previous_version_id=previous_id,
            content_hash=digest,
            rule_count=len(entries),
        )

    def rollback_rule_version(
        self,
        version_id: str,
        *,
        actor: str,
        note: str | None = None,
        expected_active_version: str | None | object = _EXPECTED_VERSION_UNSET,
    ) -> RuleVersionResult:
        """把活动规则原子切回已有不可变快照，并追加回滚审计。"""
        version_id = version_id.strip()
        actor = actor.strip()
        if not version_id or not actor:
            raise ValueError("version_id 与 actor 不能为空")
        with self._transaction():
            previous_id = self._active_rule_version_id()
            if (
                expected_active_version is not _EXPECTED_VERSION_UNSET
                and previous_id != expected_active_version
            ):
                raise VersionConflict(
                    "活动规则版本已变化，拒绝执行过期回滚",
                    detail={
                        "expected_active_version": expected_active_version,
                        "active_version": previous_id,
                    },
                )
            row = self._conn.execute(
                "SELECT * FROM rule_versions WHERE id=?", (version_id,)
            ).fetchone()
            if row is None:
                raise SchemaValidationError(
                    f"规则版本不存在：{version_id}", detail={"version_id": version_id}
                )
            if previous_id == version_id:
                return RuleVersionResult(
                    action="unchanged",
                    version_id=version_id,
                    previous_version_id=previous_id,
                    content_hash=str(row["content_hash"]),
                    rule_count=int(row["rule_count"]),
                )
            entries = self._decode_snapshot(str(row["snapshot_json"]), version_id=version_id)
            self._replace_live_rules(entries)
            self._before_rule_version_activation(version_id)
            self._set_active_rule_version_id(version_id)
            self._append_rule_version_event(
                action="rollback",
                version_id=version_id,
                previous_version_id=previous_id,
                actor=actor,
                note=note,
            )
        return RuleVersionResult(
            action="rolled_back",
            version_id=version_id,
            previous_version_id=previous_id,
            content_hash=str(row["content_hash"]),
            rule_count=int(row["rule_count"]),
        )

    def active_rule_version(self) -> RuleVersionRecord | None:
        version_id = self._active_rule_version_id()
        if version_id is None:
            return None
        row = self._conn.execute("SELECT * FROM rule_versions WHERE id=?", (version_id,)).fetchone()
        return self._row_to_rule_version(row, active_id=version_id) if row is not None else None

    def list_rule_versions(self) -> list[RuleVersionRecord]:
        active_id = self._active_rule_version_id()
        rows = self._conn.execute(
            "SELECT * FROM rule_versions ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        return [self._row_to_rule_version(row, active_id=active_id) for row in rows]

    def list_rule_version_events(self) -> list[RuleVersionEvent]:
        rows = self._conn.execute(
            "SELECT * FROM rule_version_events ORDER BY id"
        ).fetchall()
        return [
            RuleVersionEvent(
                event_id=int(row["id"]),
                action=row["action"],
                version_id=row["version_id"],
                previous_version_id=row["previous_version_id"],
                actor=row["actor"],
                note=row["note"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _active_rule_version_id(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='active_rule_version_id'"
        ).fetchone()
        return str(row["value"]) if row is not None and row["value"] else None

    def _set_active_rule_version_id(self, version_id: str) -> None:
        self._conn.execute(
            """
            INSERT INTO meta (key, value) VALUES ('active_rule_version_id', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (version_id,),
        )

    @staticmethod
    def _snapshot_entry(rule: Rule, *, extraction_batch_id: str | None) -> dict[str, Any]:
        return {
            "rule": rule.model_dump(mode="json"),
            "extraction_batch_id": extraction_batch_id,
        }

    @staticmethod
    def _stable_identity_for_entry(entry: dict[str, Any]) -> str:
        rule = Rule.model_validate(entry["rule"])
        return stable_rule_id(
            academic_year=rule.academic_year,
            college=rule.college,
            category=rule.category,
            level=rule.level,
            rank=rule.rank,
            item_name=rule.item_name,
            source=rule.source,
        )

    def _current_snapshot_entries(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM rules ORDER BY id").fetchall()
        return [
            self._snapshot_entry(
                self._row_to_rule(row), extraction_batch_id=row["extraction_batch_id"]
            )
            for row in rows
        ]

    @staticmethod
    def _rule_snapshot_hash(entries: Sequence[dict[str, Any]]) -> str:
        canonical: list[dict[str, Any]] = []
        for entry in entries:
            rule = dict(entry["rule"])
            rule.pop("created_at", None)
            canonical.append(
                {"rule": rule, "extraction_batch_id": entry.get("extraction_batch_id")}
            )
        raw = json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _ensure_rule_version(
        self,
        entries: Sequence[dict[str, Any]],
        *,
        parent_version_id: str | None,
        source_kind: str,
        source_ref: str | None,
        note: str | None,
        actor: str,
    ) -> tuple[str, str]:
        digest = self._rule_snapshot_hash(entries)
        existing = self._conn.execute(
            "SELECT id FROM rule_versions WHERE content_hash=?", (digest,)
        ).fetchone()
        if existing is not None:
            return str(existing["id"]), digest
        version_id = f"rv_{digest[:24]}"
        self._conn.execute(
            """
            INSERT INTO rule_versions (
                id, content_hash, parent_version_id, rule_count, snapshot_json,
                source_kind, source_ref, note, created_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                digest,
                parent_version_id,
                len(entries),
                json.dumps(entries, ensure_ascii=False, sort_keys=True),
                source_kind,
                source_ref,
                note,
                actor,
                datetime.now().isoformat(),
            ),
        )
        return version_id, digest

    def _replace_live_rules(self, entries: Sequence[dict[str, Any]]) -> None:
        self._conn.execute("DELETE FROM rules")
        self._conn.executemany(
            """
            INSERT INTO rules (
                id, academic_year, college, category, level, rank, item_name, score,
                synonyms, constraints, source, priority, enabled, raw_text,
                extraction_batch_id, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [self._rule_insert_row(entry) for entry in entries],
        )

    @staticmethod
    def _rule_insert_row(entry: dict[str, Any]) -> tuple[Any, ...]:
        rule = Rule.model_validate(entry["rule"])
        return (
            rule.id,
            rule.academic_year,
            rule.college,
            rule.category,
            rule.level,
            rule.rank,
            rule.item_name,
            rule.score,
            json.dumps(rule.synonyms, ensure_ascii=False),
            rule.constraints.model_dump_json(),
            rule.source.model_dump_json(),
            rule.priority,
            int(rule.enabled),
            rule.raw_text,
            entry.get("extraction_batch_id"),
            rule.created_at.isoformat(),
        )

    def _append_rule_version_event(
        self,
        *,
        action: Literal["bootstrap", "publish", "rollback"],
        version_id: str,
        previous_version_id: str | None,
        actor: str,
        note: str | None,
        created_at: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO rule_version_events (
                action, version_id, previous_version_id, actor, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                action,
                version_id,
                previous_version_id,
                actor,
                note,
                created_at or datetime.now().isoformat(),
            ),
        )

    @staticmethod
    def _decode_snapshot(snapshot_json: str, *, version_id: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(snapshot_json)
            if not isinstance(payload, list):
                raise TypeError("snapshot must be a list")
            entries = []
            for entry in payload:
                if not isinstance(entry, dict) or "rule" not in entry:
                    raise TypeError("snapshot entry is invalid")
                rule = Rule.model_validate(entry["rule"])
                entries.append(
                    {
                        "rule": rule.model_dump(mode="json"),
                        "extraction_batch_id": entry.get("extraction_batch_id"),
                    }
                )
            Ruleset(rules=[Rule.model_validate(entry["rule"]) for entry in entries])
            return entries
        except Exception as exc:
            raise SchemaValidationError(
                f"规则版本快照损坏：{version_id}", detail={"version_id": version_id}
            ) from exc

    @staticmethod
    def _row_to_rule_version(
        row: sqlite3.Row, *, active_id: str | None
    ) -> RuleVersionRecord:
        return RuleVersionRecord(
            version_id=row["id"],
            content_hash=row["content_hash"],
            parent_version_id=row["parent_version_id"],
            rule_count=row["rule_count"],
            source_kind=row["source_kind"],
            source_ref=row["source_ref"],
            note=row["note"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            active=row["id"] == active_id,
        )

    def _before_rule_version_activation(self, version_id: str) -> None:
        """故障注入钩子；异常会回滚规则表、快照、指针与审计。"""

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        entries = self._current_snapshot_entries()
        changed = False
        for entry in entries:
            if entry["rule"]["id"] == rule_id:
                entry["rule"]["enabled"] = enabled
                changed = True
                break
        if not changed:
            return False
        self.publish_rule_snapshot(
            [entry["rule"] for entry in entries],
            extraction_batch_ids={
                entry["rule"]["id"]: entry["extraction_batch_id"] for entry in entries
            },
            actor="scoreproof:store",
            source_kind="status_change",
            source_ref=rule_id,
            note=f"设置 enabled={enabled}",
            expected_active_version=self._active_rule_version_id(),
        )
        return True

    # ---------- 读取 ----------

    def list_rules(
        self,
        *,
        academic_year: str | None = None,
        college: str | None = None,
        category: str | None = None,
        enabled_only: bool = True,
    ) -> list[Rule]:
        sql = "SELECT * FROM rules WHERE 1=1"
        params: list = []
        if academic_year:
            sql += " AND academic_year = ?"
            params.append(academic_year)
        if college:
            sql += " AND (college IS NULL OR college = ?)"
            params.append(college)
        if category:
            sql += " AND category = ?"
            params.append(category)
        if enabled_only:
            sql += " AND enabled = 1"
        sql += " ORDER BY category, level, priority DESC, id"
        with closing(self._conn.execute(sql, params)) as cur:
            return [self._row_to_rule(row) for row in cur.fetchall()]

    def load_ruleset(self, **kwargs) -> Ruleset:
        active = self.active_rule_version()
        return Ruleset(
            rules=self.list_rules(**kwargs),
            version=active.version_id if active is not None else "unversioned",
            meta={
                "rule_version_id": active.version_id,
                "rule_content_hash": active.content_hash,
            }
            if active is not None
            else {},
        )

    def count(self, *, academic_year: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM rules"
        params: list = []
        if academic_year:
            sql += " WHERE academic_year = ?"
            params.append(academic_year)
        return int(self._conn.execute(sql, params).fetchone()[0])

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> Rule:
        payload = {
            "id": row["id"],
            "academic_year": row["academic_year"],
            "college": row["college"],
            "category": row["category"],
            "level": row["level"],
            "rank": row["rank"],
            "item_name": row["item_name"],
            "score": row["score"],
            "synonyms": json.loads(row["synonyms"] or "[]"),
            "constraints": json.loads(row["constraints"] or "{}"),
            "source": json.loads(row["source"] or "{}"),
            "priority": row["priority"],
            "enabled": bool(row["enabled"]),
            "raw_text": row["raw_text"],
            "created_at": row["created_at"] or datetime.now().isoformat(),
        }
        try:
            return Rule.model_validate(payload)
        except Exception as exc:  # pragma: no cover - 数据损坏时给出明确错误
            raise SchemaValidationError(
                f"规则 {row['id']} 反序列化失败：{exc}", detail={"row": dict(row)}
            ) from exc

    # ---------- LLM 抽取缓存、审计与发布门禁 ----------

    def get_extraction_cache(
        self, *, chunk_hash: str, model: str, variant: str
    ) -> list[dict] | None:
        row = self._conn.execute(
            "SELECT payloads FROM extraction_cache WHERE chunk_hash=? AND model=? AND variant=?",
            (chunk_hash, model, variant),
        ).fetchone()
        return json.loads(row["payloads"]) if row is not None else None

    def set_extraction_cache(
        self, *, chunk_hash: str, model: str, variant: str, payloads: list[dict]
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO extraction_cache (chunk_hash, model, variant, payloads, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_hash, model, variant) DO UPDATE SET
                    payloads=excluded.payloads, created_at=excluded.created_at
                """,
                (
                    chunk_hash,
                    model,
                    variant,
                    json.dumps(payloads, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )

    # ---------- 工具编排审计 ----------

    def record_tool_call(
        self,
        *,
        call_id: str,
        session_id: str,
        tool_name: str,
        input_payload: dict,
        result_summary: str,
        duration_ms: float,
        model_version: str,
        status: Literal["ok", "error", "degraded"],
        error: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """记录一次工具调用；调用参数只存业务字段，调用方不得传入密钥。"""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tool_calls (
                    id, session_id, tool_name, input_json, result_summary,
                    duration_ms, model_version, status, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    session_id,
                    tool_name,
                    json.dumps(input_payload, ensure_ascii=False, default=str),
                    result_summary,
                    duration_ms,
                    model_version,
                    status,
                    error,
                    (created_at or datetime.now()).isoformat(),
                ),
            )

    def list_tool_calls(self, *, session_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM tool_calls"
        params: list[str] = []
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        # Windows 上 datetime.now() 可能让连续调用共享同一时间刻度；rowid 保留真实插入顺序。
        sql += " ORDER BY created_at, rowid"
        with closing(self._conn.execute(sql, params)) as cur:
            return [dict(row) for row in cur.fetchall()]

    def record_extraction_report(
        self,
        report: GatewayReport,
        *,
        model: str | None = None,
        status: Literal["validated", "blocked", "published"] | None = None,
    ) -> None:
        final_status = status or ("validated" if report.publishable else "blocked")
        with self._conn:
            self._write_extraction_report(report, model=model, status=final_status)

    def publish_extraction_report(self, report: GatewayReport, *, model: str | None = None) -> int:
        """原子发布已通过网关的规则；这是 LLM 草稿唯一允许使用的写入口。"""
        published = self._conn.execute(
            "SELECT status FROM extraction_batches WHERE id=?", (report.batch_id,)
        ).fetchone()
        if published is not None and published["status"] == "published":
            active_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM rules WHERE extraction_batch_id=?", (report.batch_id,)
                ).fetchone()[0]
            )
            if active_count:
                return active_count
            historical = self._conn.execute(
                """
                SELECT id FROM rule_versions
                WHERE source_kind='llm_extraction' AND source_ref=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (report.batch_id,),
            ).fetchone()
            raise VersionConflict(
                "抽取批次曾发布但当前快照已回滚；请显式回滚到对应规则版本",
                detail={
                    "batch_id": report.batch_id,
                    "rule_version_id": historical["id"] if historical is not None else None,
                    "active_version": self._active_rule_version_id(),
                },
            )
        if not report.publishable:
            self.record_extraction_report(report, model=model, status="blocked")
            raise SchemaValidationError(
                "抽取批次未通过五道验证或发布前冲突门禁，禁止发布",
                detail={"batch_id": report.batch_id, "metrics": report.metrics()},
            )

        rules = report.rules()
        expected_active_version = self._active_rule_version_id()
        current_entries = self._current_snapshot_entries()
        existing: dict[tuple[str, str, str, str, str, str], set[float]] = {}
        for rule in (Rule.model_validate(entry["rule"]) for entry in current_entries):
            key = (
                rule.academic_year,
                rule.college or "",
                rule.category,
                rule.level,
                rule.rank or "",
                rule.item_name or "",
            )
            existing.setdefault(key, set()).add(rule.score)
        conflicts: list[dict] = []
        for rule in rules:
            key = (
                rule.academic_year,
                rule.college or "",
                rule.category,
                rule.level,
                rule.rank or "",
                rule.item_name or "",
            )
            scores = existing.get(key, set())
            if scores and rule.score not in scores:
                conflicts.append({"key": list(key), "existing_scores": sorted(scores), "score": rule.score})
            existing.setdefault(key, set()).add(rule.score)
        if conflicts:
            self.record_extraction_report(report, model=model, status="blocked")
            raise VersionConflict(
                "规则冲突门禁阻止了抽取批次发布",
                detail={"batch_id": report.batch_id, "conflicts": conflicts},
            )

        new_entries = [
            self._snapshot_entry(rule, extraction_batch_id=report.batch_id) for rule in rules
        ]
        combined_by_id = {entry["rule"]["id"]: entry for entry in current_entries}
        for entry in new_entries:
            identity_id = self._stable_identity_for_entry(entry)
            matching_ids = [
                current_id
                for current_id, current in combined_by_id.items()
                if self._stable_identity_for_entry(current) == identity_id
            ]
            for current_id in matching_ids:
                combined_by_id.pop(current_id)
            combined_by_id[entry["rule"]["id"]] = entry
        combined = list(combined_by_id.values())
        Ruleset(rules=[Rule.model_validate(entry["rule"]) for entry in combined])
        combined.sort(key=lambda entry: str(entry["rule"]["id"]))
        with self._transaction():
            active_version = self._active_rule_version_id()
            if active_version != expected_active_version:
                raise VersionConflict(
                    "活动规则版本已变化，请基于最新版本重新执行抽取发布",
                    detail={
                        "expected_active_version": expected_active_version,
                        "active_version": active_version,
                        "batch_id": report.batch_id,
                    },
                )
            self._write_extraction_report(report, model=model, status="published")
            version_id, _ = self._ensure_rule_version(
                combined,
                parent_version_id=active_version,
                source_kind="llm_extraction",
                source_ref=report.batch_id,
                note=f"发布规则抽取批次 {report.batch_id}",
                actor="scoreproof:rule-extractor",
            )
            self._replace_live_rules(combined)
            self._before_rule_version_activation(version_id)
            self._set_active_rule_version_id(version_id)
            self._append_rule_version_event(
                action="publish",
                version_id=version_id,
                previous_version_id=active_version,
                actor="scoreproof:rule-extractor",
                note=f"规则抽取批次 {report.batch_id}",
            )
        return len(rules)

    def _write_extraction_report(
        self,
        report: GatewayReport,
        *,
        model: str | None,
        status: str,
    ) -> None:
        context = report.context
        self._conn.execute(
            """
            INSERT INTO extraction_batches
                (id, chunk_hash, academic_year, college, doc, model, status, secondary_used,
                 metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model=excluded.model, status=excluded.status,
                secondary_used=excluded.secondary_used, metrics=excluded.metrics
            """,
            (
                report.batch_id,
                report.chunk_hash,
                context.academic_year,
                context.college,
                context.doc,
                model,
                status,
                int(report.secondary_used),
                json.dumps(report.metrics(), ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        self._conn.executemany(
            """
            INSERT INTO extraction_audits
                (batch_id, item_index, status, raw_payload, issues, char_start, char_end,
                 ambiguity_flag, extract_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id, item_index) DO UPDATE SET
                status=excluded.status, raw_payload=excluded.raw_payload, issues=excluded.issues,
                char_start=excluded.char_start, char_end=excluded.char_end,
                ambiguity_flag=excluded.ambiguity_flag,
                extract_confidence=excluded.extract_confidence
            """,
            [
                (
                    report.batch_id,
                    item.index,
                    item.status,
                    json.dumps(item.raw_payload, ensure_ascii=False),
                    json.dumps([issue.model_dump(mode="json") for issue in item.issues], ensure_ascii=False),
                    item.char_start,
                    item.char_end,
                    int(item.ambiguity_flag),
                    item.extract_confidence,
                )
                for item in report.items
            ],
        )

    def list_extraction_audits(self, batch_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM extraction_audits WHERE batch_id=? ORDER BY item_index", (batch_id,)
        ).fetchall()
        return [
            {
                **dict(row),
                "raw_payload": json.loads(row["raw_payload"]),
                "issues": json.loads(row["issues"]),
                "ambiguity_flag": bool(row["ambiguity_flag"]),
                "manual_corrected": bool(row["manual_corrected"]),
                "corrected_payload": (
                    json.loads(row["corrected_payload"]) if row["corrected_payload"] else None
                ),
            }
            for row in rows
        ]

    def record_extraction_correction(
        self, batch_id: str, item_index: int, corrected_payload: dict
    ) -> bool:
        """记录人工修正；修正稿仍须重新运行网关，不能直接发布。"""
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE extraction_audits
                SET manual_corrected=1, corrected_payload=?
                WHERE batch_id=? AND item_index=?
                """,
                (json.dumps(corrected_payload, ensure_ascii=False), batch_id, item_index),
            )
        return cursor.rowcount > 0

    def extraction_audit_metrics(self, batch_id: str) -> dict:
        """汇总某批次的拦截原因与人工修正率。"""
        audits = self.list_extraction_audits(batch_id)
        corrections = sum(int(item["manual_corrected"]) for item in audits)
        reasons: dict[str, int] = {}
        for item in audits:
            for issue in item["issues"]:
                code = str(issue["code"])
                reasons[code] = reasons.get(code, 0) + 1
        total = len(audits)
        return {
            "batch_id": batch_id,
            "sample_size": total,
            "manual_corrections": corrections,
            "manual_correction_rate": corrections / total if total else None,
            "reasons": dict(sorted(reasons.items())),
        }


def export_json(ruleset: Ruleset, path: str | Path) -> Path:
    """导出为 JSON（人工校对 / 版本 diff 用）。"""
    return ruleset.to_json(path)


def import_json(path: str | Path) -> Ruleset:
    return Ruleset.from_json(path)


__all__ = [
    "SCHEMA_SQL",
    "RuleStore",
    "RuleVersionEvent",
    "RuleVersionRecord",
    "RuleVersionResult",
    "export_json",
    "import_json",
]
