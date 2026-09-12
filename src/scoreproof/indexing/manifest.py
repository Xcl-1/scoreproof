"""文档/逻辑块双级 Hash 与可回滚的索引 manifest。

本模块只负责确定“哪些逻辑块需要重建”，并在 SQLite 中原子切换当前
manifest。BM25、Embedding 与向量库在后续工作包消费 ``rebuild_keys``；
未完成构建前不得调用 ``publish``，因此线上永远只读取完整快照。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..rules.gateway import chunk_hash
from ..schema import SourceRef

ChunkKind = Literal["page", "section", "table", "paragraph", "other"]
ManifestStatus = Literal["staged", "published", "superseded", "rolled_back"]

MANIFEST_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    source_path         TEXT NOT NULL,
    document_hash       TEXT NOT NULL,
    current_manifest_id TEXT,
    deleted             INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_manifests (
    id                   TEXT PRIMARY KEY,
    doc_id               TEXT NOT NULL,
    document_hash        TEXT NOT NULL,
    embedding_model      TEXT NOT NULL,
    status               TEXT NOT NULL,
    previous_manifest_id TEXT,
    created_at           TEXT NOT NULL,
    published_at         TEXT,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_index_manifests_doc
    ON index_manifests(doc_id, created_at);

CREATE TABLE IF NOT EXISTS document_chunks (
    manifest_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    chunk_hash  TEXT NOT NULL,
    index_key   TEXT NOT NULL,
    source      TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(manifest_id, logical_key),
    FOREIGN KEY(manifest_id) REFERENCES index_manifests(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_document_chunks_hash
    ON document_chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_document_chunks_index_key
    ON document_chunks(index_key);
"""


class DocumentChunk(BaseModel):
    """调用方已完成版面清洗后的稳定逻辑块。"""

    model_config = ConfigDict(extra="forbid")

    logical_key: str = Field(min_length=1, description="稳定定位键，如 page:6 或 table:2")
    text: str = Field(min_length=1)
    kind: ChunkKind = "other"
    source: SourceRef
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("logical_key", "text")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("逻辑块键与正文不能为空")
        return stripped


class StoredChunk(DocumentChunk):
    """某个 manifest 内不可变存在的逻辑块快照。"""

    ordinal: int = Field(ge=0)
    chunk_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestDiff(BaseModel):
    """相邻 manifest 的确定性差异；只有 ``rebuild_keys`` 消耗模型调用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    added_keys: list[str] = Field(default_factory=list)
    changed_keys: list[str] = Field(default_factory=list)
    metadata_only_keys: list[str] = Field(default_factory=list)
    removed_keys: list[str] = Field(default_factory=list)
    unchanged_keys: list[str] = Field(default_factory=list)

    @property
    def rebuild_keys(self) -> list[str]:
        return [*self.added_keys, *self.changed_keys]


class SyncResult(BaseModel):
    """一次同步或回滚的可审计结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["published", "unchanged", "rolled_back", "deleted"]
    doc_id: str
    manifest_id: str | None
    previous_manifest_id: str | None = None
    document_hash: str | None = None
    embedding_model: str | None = None
    diff: ManifestDiff = Field(default_factory=ManifestDiff)


def document_hash(content: bytes) -> str:
    """对原始文件字节计算 SHA-256；任何字节变化都会留下新版本。"""
    return hashlib.sha256(content).hexdigest()


def index_key(doc_id: str, digest: str, embedding_model: str) -> str:
    """按 V3.0 约定由 doc_id、chunk_hash、模型版本生成稳定索引键。"""
    raw = "\0".join((doc_id, digest, embedding_model)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class IndexManifestStore:
    """SQLite manifest 仓库；所有快照切换都在单事务中完成。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(MANIFEST_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IndexManifestStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def sync_document(
        self,
        *,
        doc_id: str,
        source_path: str | Path,
        document_bytes: bytes,
        chunks: Sequence[DocumentChunk],
        embedding_model: str,
    ) -> SyncResult:
        """建立完整快照并原子发布，返回需要重建索引的逻辑块。"""
        doc_id = doc_id.strip()
        embedding_model = embedding_model.strip()
        if not doc_id or not embedding_model:
            raise ValueError("doc_id 与 embedding_model 不能为空")
        if not chunks:
            raise ValueError("文档至少需要一个非空逻辑块")
        prepared = self._prepare_chunks(doc_id, chunks, embedding_model)
        digest = document_hash(document_bytes)
        current = self._current_document(doc_id)
        previous_id = current["current_manifest_id"] if current and not current["deleted"] else None
        previous = self._chunks_for_manifest(previous_id) if previous_id else []
        previous_model = self._manifest_model(previous_id) if previous_id else None
        diff = self._diff(previous, prepared, model_changed=previous_model != embedding_model)

        if (
            current is not None
            and not current["deleted"]
            and current["document_hash"] == digest
            and not diff.added_keys
            and not diff.changed_keys
            and not diff.metadata_only_keys
            and not diff.removed_keys
        ):
            return SyncResult(
                action="unchanged",
                doc_id=doc_id,
                manifest_id=previous_id,
                document_hash=digest,
                embedding_model=embedding_model,
                diff=diff,
            )

        manifest_id = f"im_{uuid.uuid4().hex[:16]}"
        now = datetime.now().isoformat()
        with self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if exists is None:
                self._conn.execute(
                    """
                    INSERT INTO documents
                        (doc_id, source_path, document_hash, current_manifest_id,
                         deleted, created_at, updated_at)
                    VALUES (?, ?, ?, NULL, 0, ?, ?)
                    """,
                    (doc_id, str(source_path), digest, now, now),
                )
            self._conn.execute(
                """
                INSERT INTO index_manifests
                    (id, doc_id, document_hash, embedding_model, status,
                     previous_manifest_id, created_at)
                VALUES (?, ?, ?, ?, 'staged', ?, ?)
                """,
                (manifest_id, doc_id, digest, embedding_model, previous_id, now),
            )
            self._conn.executemany(
                """
                INSERT INTO document_chunks
                    (manifest_id, logical_key, ordinal, kind, text, chunk_hash,
                     index_key, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        manifest_id,
                        chunk.logical_key,
                        chunk.ordinal,
                        chunk.kind,
                        chunk.text,
                        chunk.chunk_hash,
                        chunk.index_key,
                        chunk.source.model_dump_json(),
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    )
                    for chunk in prepared
                ],
            )
            self._before_publish(manifest_id)
            if previous_id:
                self._conn.execute(
                    "UPDATE index_manifests SET status='superseded' WHERE id=?",
                    (previous_id,),
                )
            self._conn.execute(
                "UPDATE index_manifests SET status='published', published_at=? WHERE id=?",
                (now, manifest_id),
            )
            self._conn.execute(
                """
                UPDATE documents
                SET source_path=?, document_hash=?, current_manifest_id=?,
                    deleted=0, updated_at=?
                WHERE doc_id=?
                """,
                (str(source_path), digest, manifest_id, now, doc_id),
            )
        return SyncResult(
            action="published",
            doc_id=doc_id,
            manifest_id=manifest_id,
            previous_manifest_id=previous_id,
            document_hash=digest,
            embedding_model=embedding_model,
            diff=diff,
        )

    def rollback(self, doc_id: str, *, manifest_id: str | None = None) -> SyncResult:
        """把当前指针原子切回同文档的历史完整快照。"""
        current = self._current_document(doc_id)
        if current is None or current["deleted"] or not current["current_manifest_id"]:
            raise ValueError(f"文档没有可回滚的当前 manifest：{doc_id}")
        current_id = str(current["current_manifest_id"])
        if manifest_id is None:
            row = self._conn.execute(
                "SELECT previous_manifest_id FROM index_manifests WHERE id=?", (current_id,)
            ).fetchone()
            manifest_id = row["previous_manifest_id"] if row else None
        if not manifest_id:
            raise ValueError(f"文档没有更早的 manifest：{doc_id}")
        target = self._conn.execute(
            "SELECT * FROM index_manifests WHERE id=? AND doc_id=?", (manifest_id, doc_id)
        ).fetchone()
        if target is None:
            raise ValueError("目标 manifest 不存在或不属于该文档")
        now = datetime.now().isoformat()
        with self._conn:
            self._conn.execute(
                "UPDATE index_manifests SET status='rolled_back' WHERE id=?", (current_id,)
            )
            self._conn.execute(
                "UPDATE index_manifests SET status='published', published_at=? WHERE id=?",
                (now, manifest_id),
            )
            self._conn.execute(
                """
                UPDATE documents
                SET document_hash=?, current_manifest_id=?, deleted=0, updated_at=?
                WHERE doc_id=?
                """,
                (target["document_hash"], manifest_id, now, doc_id),
            )
        return SyncResult(
            action="rolled_back",
            doc_id=doc_id,
            manifest_id=manifest_id,
            previous_manifest_id=current_id,
            document_hash=target["document_hash"],
            embedding_model=target["embedding_model"],
        )

    def delete_document(self, doc_id: str) -> SyncResult:
        """逻辑删除文档；历史快照保留，但不会再出现在活动索引中。"""
        current = self._current_document(doc_id)
        if current is None or current["deleted"]:
            raise ValueError(f"活动文档不存在：{doc_id}")
        current_id = current["current_manifest_id"]
        now = datetime.now().isoformat()
        with self._conn:
            if current_id:
                self._conn.execute(
                    "UPDATE index_manifests SET status='superseded' WHERE id=?", (current_id,)
                )
            self._conn.execute(
                """
                UPDATE documents
                SET current_manifest_id=NULL, deleted=1, updated_at=? WHERE doc_id=?
                """,
                (now, doc_id),
            )
        return SyncResult(
            action="deleted",
            doc_id=doc_id,
            manifest_id=None,
            previous_manifest_id=current_id,
            document_hash=current["document_hash"],
        )

    def active_chunks(self, *, doc_id: str | None = None) -> list[StoredChunk]:
        sql = """
            SELECT c.* FROM document_chunks c
            JOIN documents d ON d.current_manifest_id=c.manifest_id
            WHERE d.deleted=0
        """
        params: list[str] = []
        if doc_id is not None:
            sql += " AND d.doc_id=?"
            params.append(doc_id)
        sql += " ORDER BY d.doc_id, c.ordinal, c.logical_key"
        with closing(self._conn.execute(sql, params)) as cursor:
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def manifest_count(self, doc_id: str) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM index_manifests WHERE doc_id=?", (doc_id,)
            ).fetchone()[0]
        )

    def _before_publish(self, manifest_id: str) -> None:
        """供后续索引构建适配器扩展；异常会回滚整个 SQLite 发布事务。"""

    def _current_document(self, doc_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()

    def _manifest_model(self, manifest_id: str) -> str:
        row = self._conn.execute(
            "SELECT embedding_model FROM index_manifests WHERE id=?", (manifest_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"manifest 不存在：{manifest_id}")
        return str(row["embedding_model"])

    def _chunks_for_manifest(self, manifest_id: str) -> list[StoredChunk]:
        rows = self._conn.execute(
            "SELECT * FROM document_chunks WHERE manifest_id=? ORDER BY ordinal, logical_key",
            (manifest_id,),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> StoredChunk:
        return StoredChunk(
            logical_key=row["logical_key"],
            ordinal=row["ordinal"],
            kind=row["kind"],
            text=row["text"],
            chunk_hash=row["chunk_hash"],
            index_key=row["index_key"],
            source=json.loads(row["source"]),
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _prepare_chunks(
        doc_id: str, chunks: Sequence[DocumentChunk], embedding_model: str
    ) -> list[StoredChunk]:
        keys = [chunk.logical_key for chunk in chunks]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"logical_key 重复：{duplicates}")
        prepared: list[StoredChunk] = []
        for ordinal, chunk in enumerate(chunks):
            digest = chunk_hash(chunk.text)
            prepared.append(
                StoredChunk(
                    **chunk.model_dump(),
                    ordinal=ordinal,
                    chunk_hash=digest,
                    index_key=index_key(doc_id, digest, embedding_model),
                )
            )
        return prepared

    @staticmethod
    def _diff(
        previous: Sequence[StoredChunk],
        current: Sequence[StoredChunk],
        *,
        model_changed: bool,
    ) -> ManifestDiff:
        old = {chunk.logical_key: chunk for chunk in previous}
        new = {chunk.logical_key: chunk for chunk in current}
        added = sorted(new.keys() - old.keys())
        removed = sorted(old.keys() - new.keys())
        changed: list[str] = []
        metadata_only: list[str] = []
        unchanged: list[str] = []
        for key in sorted(old.keys() & new.keys()):
            before, after = old[key], new[key]
            if model_changed or before.chunk_hash != after.chunk_hash:
                changed.append(key)
            elif (
                before.ordinal != after.ordinal
                or before.kind != after.kind
                or before.source != after.source
                or before.metadata != after.metadata
            ):
                metadata_only.append(key)
            else:
                unchanged.append(key)
        return ManifestDiff(
            added_keys=added,
            changed_keys=changed,
            metadata_only_keys=metadata_only,
            removed_keys=removed,
            unchanged_keys=unchanged,
        )


__all__ = [
    "ChunkKind",
    "DocumentChunk",
    "IndexManifestStore",
    "MANIFEST_SCHEMA_SQL",
    "ManifestDiff",
    "ManifestStatus",
    "StoredChunk",
    "SyncResult",
    "document_hash",
    "index_key",
]
