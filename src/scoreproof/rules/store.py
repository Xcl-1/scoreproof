"""规则库持久化：SQLite + JSON 双向。

为什么是 SQLite：规则库是**结构化数据**，需要按学年/学院/类别/等级精确查表，
天然适合关系表；同时导出 JSON 便于人工校对与 diff（也是简历里的可验证产物）。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..errors import SchemaValidationError, VersionConflict
from ..schema import Rule, Ruleset

if TYPE_CHECKING:
    from .gateway import GatewayReport

SCHEMA_SQL = """
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
"""


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

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RuleStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- 写入 ----------

    def upsert_rules(self, rules: list[Rule]) -> int:
        rows = [
            (
                r.id,
                r.academic_year,
                r.college,
                r.category,
                r.level,
                r.rank,
                r.item_name,
                r.score,
                json.dumps(r.synonyms, ensure_ascii=False),
                r.constraints.model_dump_json(),
                r.source.model_dump_json(),
                r.priority,
                int(r.enabled),
                r.raw_text,
                None,
                r.created_at.isoformat(),
            )
            for r in rules
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO rules (id, academic_year, college, category, level, rank, item_name, score,
                                   synonyms, constraints, source, priority, enabled, raw_text,
                                   extraction_batch_id, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    academic_year=excluded.academic_year, college=excluded.college,
                    category=excluded.category, level=excluded.level, rank=excluded.rank,
                    item_name=excluded.item_name, score=excluded.score,
                    synonyms=excluded.synonyms, constraints=excluded.constraints,
                    source=excluded.source, priority=excluded.priority,
                    enabled=excluded.enabled, raw_text=excluded.raw_text,
                    extraction_batch_id=COALESCE(excluded.extraction_batch_id, rules.extraction_batch_id)
                """,
                rows,
            )
        return len(rows)

    def delete_rule(self, rule_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        return cur.rowcount > 0

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE rules SET enabled = ? WHERE id = ?", (int(enabled), rule_id)
            )
        return cur.rowcount > 0

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
        return Ruleset(rules=self.list_rules(**kwargs))

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
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM rules WHERE extraction_batch_id=?", (report.batch_id,)
                ).fetchone()[0]
            )
        if not report.publishable:
            self.record_extraction_report(report, model=model, status="blocked")
            raise SchemaValidationError(
                "抽取批次未通过五道验证或发布前冲突门禁，禁止发布",
                detail={"batch_id": report.batch_id, "metrics": report.metrics()},
            )

        rules = report.rules()
        existing: dict[tuple[str, str, str, str, str, str], set[float]] = {}
        for rule in self.list_rules(enabled_only=False):
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

        rows = [
            (
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
                report.batch_id,
                rule.created_at.isoformat(),
            )
            for rule in rules
        ]
        with self._conn:
            self._write_extraction_report(report, model=model, status="published")
            self._conn.executemany(
                """
                INSERT INTO rules (id, academic_year, college, category, level, rank, item_name, score,
                                   synonyms, constraints, source, priority, enabled, raw_text,
                                   extraction_batch_id, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

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


__all__ = ["SCHEMA_SQL", "RuleStore", "export_json", "import_json"]
