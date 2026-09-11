"""规则库持久化：SQLite + JSON 双向。

为什么是 SQLite：规则库是**结构化数据**，需要按学年/学院/类别/等级精确查表，
天然适合关系表；同时导出 JSON 便于人工校对与 diff（也是简历里的可验证产物）。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from ..errors import SchemaValidationError
from ..schema import Rule, Ruleset

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rules (
    id            TEXT PRIMARY KEY,
    academic_year TEXT NOT NULL,
    college       TEXT,
    category      TEXT NOT NULL,
    level         TEXT NOT NULL,
    score         REAL NOT NULL,
    synonyms      TEXT NOT NULL DEFAULT '[]',
    constraints   TEXT NOT NULL DEFAULT '{}',
    source        TEXT NOT NULL DEFAULT '{}',
    priority      INTEGER NOT NULL DEFAULT 0,
    enabled       INTEGER NOT NULL DEFAULT 1,
    raw_text      TEXT,
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
                r.score,
                json.dumps(r.synonyms, ensure_ascii=False),
                r.constraints.model_dump_json(),
                r.source.model_dump_json(),
                r.priority,
                int(r.enabled),
                r.raw_text,
                r.created_at.isoformat(),
            )
            for r in rules
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO rules (id, academic_year, college, category, level, score,
                                   synonyms, constraints, source, priority, enabled, raw_text, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    academic_year=excluded.academic_year, college=excluded.college,
                    category=excluded.category, level=excluded.level, score=excluded.score,
                    synonyms=excluded.synonyms, constraints=excluded.constraints,
                    source=excluded.source, priority=excluded.priority,
                    enabled=excluded.enabled, raw_text=excluded.raw_text
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


def export_json(ruleset: Ruleset, path: str | Path) -> Path:
    """导出为 JSON（人工校对 / 版本 diff 用）。"""
    return ruleset.to_json(path)


def import_json(path: str | Path) -> Ruleset:
    return Ruleset.from_json(path)


__all__ = ["SCHEMA_SQL", "RuleStore", "export_json", "import_json"]
