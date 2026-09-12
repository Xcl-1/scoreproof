"""与 Manifest 同批发布的 BM25 文档和 Chroma 向量快照。"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..schema import SourceRef
from ..tokenize import tokenize_for_search
from .manifest import DocumentChunk, IndexManifestStore, StoredChunk, SyncResult

SEARCH_INDEX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS search_index_batches (
    manifest_id      TEXT PRIMARY KEY,
    doc_id           TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    chroma_collection TEXT NOT NULL,
    bm25_doc_count   INTEGER NOT NULL,
    vector_doc_count INTEGER NOT NULL,
    embedded_count   INTEGER NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    FOREIGN KEY(manifest_id) REFERENCES index_manifests(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bm25_index_documents (
    manifest_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    tokens      TEXT NOT NULL,
    source      TEXT NOT NULL,
    metadata    TEXT NOT NULL,
    chunk_hash  TEXT NOT NULL,
    index_key   TEXT NOT NULL,
    PRIMARY KEY(manifest_id, logical_key),
    FOREIGN KEY(manifest_id) REFERENCES search_index_batches(manifest_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bm25_batch ON bm25_index_documents(manifest_id, ordinal);
"""


class EmbeddingFunction(Protocol):
    """LangChain Embeddings 的最小兼容接口，保持可选依赖可延迟加载。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SearchIndexBatch(BaseModel):
    """一个 Manifest 对应的完整、不可变检索快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_id: str
    doc_id: str
    embedding_model: str
    chroma_collection: str
    bm25_doc_count: int = Field(ge=0)
    vector_doc_count: int = Field(ge=0)
    embedded_count: int = Field(ge=0)
    status: str
    created_at: datetime


class BM25IndexDocument(BaseModel):
    """持久化的 BM25 文档；只从活动 Manifest 批次读取。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_id: str
    logical_key: str
    doc_id: str
    ordinal: int = Field(ge=0)
    text: str
    tokens: list[str]
    source: SourceRef
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_hash: str
    index_key: str


class HashingEmbeddings:
    """离线可复现的稠密特征哈希 Embedding。

    它用于本地验收和无外部 Embedding 服务时的确定性降级，不宣称具备预训练
    语义模型的效果。生产评测可注入任意 LangChain ``Embeddings`` 实现。
    """

    def __init__(self, *, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("Embedding 维度不能小于 8")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        tokens = tokenize_for_search(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            values[slot] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return values


class HybridIndexManifestStore(IndexManifestStore):
    """先构建 BM25/Chroma 快照，再原子切换活动 Manifest 指针。"""

    def __init__(
        self,
        path: str | Path,
        *,
        vector_dir: str | Path,
        embeddings: EmbeddingFunction | None = None,
        embedding_model: str = "scoreproof-hash-v1",
    ) -> None:
        super().__init__(path)
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or HashingEmbeddings()
        self.embedding_model = embedding_model.strip()
        if not self.embedding_model:
            raise ValueError("embedding_model 不能为空")
        self._staged_collection: str | None = None
        with self._conn:
            self._conn.executescript(SEARCH_INDEX_SCHEMA_SQL)

    def __enter__(self) -> HybridIndexManifestStore:
        return self

    def sync_document(
        self,
        *,
        doc_id: str,
        source_path: str | Path,
        document_bytes: bytes,
        chunks: Sequence[DocumentChunk],
        embedding_model: str,
        force_publish: bool = False,
    ) -> SyncResult:
        if embedding_model != self.embedding_model:
            raise ValueError(
                f"同步模型 {embedding_model!r} 与索引器模型 {self.embedding_model!r} 不一致"
            )
        current = self._current_document(doc_id.strip())
        current_manifest_id = (
            str(current["current_manifest_id"])
            if current is not None and not current["deleted"] and current["current_manifest_id"]
            else None
        )
        force_publish = force_publish or bool(
            current_manifest_id and not self._batch_is_complete(current_manifest_id)
        )
        self._staged_collection = None
        try:
            result = super().sync_document(
                doc_id=doc_id,
                source_path=source_path,
                document_bytes=document_bytes,
                chunks=chunks,
                embedding_model=embedding_model,
                force_publish=force_publish,
            )
        except Exception:
            self._drop_staged_collection()
            raise
        self._staged_collection = None
        return result

    def _before_publish(self, manifest_id: str) -> None:
        manifest = self._conn.execute(
            "SELECT * FROM index_manifests WHERE id=?", (manifest_id,)
        ).fetchone()
        if manifest is None:  # pragma: no cover - 基表约束保证存在
            raise ValueError(f"manifest 不存在：{manifest_id}")
        chunks = self._chunks_for_manifest(manifest_id)
        previous_id = manifest["previous_manifest_id"]
        previous_chunks = (
            {chunk.logical_key: chunk for chunk in self._chunks_for_manifest(previous_id)}
            if previous_id
            else {}
        )
        collection_name = _collection_name(manifest_id)
        self._staged_collection = collection_name
        collection = self._new_collection(collection_name)

        reusable, to_embed = self._partition_embeddings(
            chunks=chunks,
            previous_manifest_id=previous_id,
            previous_chunks=previous_chunks,
        )
        generated: dict[str, list[float]] = {}
        if to_embed:
            vectors = self.embeddings.embed_documents([chunk.text for chunk in to_embed])
            if len(vectors) != len(to_embed):
                raise ValueError("Embedding 返回数量与待构建块数量不一致")
            generated = {
                chunk.logical_key: vector for chunk, vector in zip(to_embed, vectors, strict=True)
            }
        all_vectors = {**reusable, **generated}
        if len(all_vectors) != len(chunks):
            raise ValueError("向量快照不完整，拒绝发布")

        collection.add(
            ids=[_record_id(manifest_id, chunk.logical_key) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[all_vectors[chunk.logical_key] for chunk in chunks],
            metadatas=[
                _chroma_metadata(
                    manifest_id=manifest_id,
                    doc_id=str(manifest["doc_id"]),
                    chunk=chunk,
                )
                for chunk in chunks
            ],
        )
        if collection.count() != len(chunks):
            raise ValueError("Chroma 写入数量不完整，拒绝发布")

        now = datetime.now().isoformat()
        self._conn.execute(
            """
            INSERT INTO search_index_batches
                (manifest_id, doc_id, embedding_model, chroma_collection,
                 bm25_doc_count, vector_doc_count, embedded_count, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?)
            """,
            (
                manifest_id,
                manifest["doc_id"],
                manifest["embedding_model"],
                collection_name,
                len(chunks),
                len(chunks),
                len(to_embed),
                now,
            ),
        )
        self._conn.executemany(
            """
            INSERT INTO bm25_index_documents
                (manifest_id, logical_key, doc_id, ordinal, text, tokens,
                 source, metadata, chunk_hash, index_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    manifest_id,
                    chunk.logical_key,
                    manifest["doc_id"],
                    chunk.ordinal,
                    chunk.text,
                    json.dumps(tokenize_for_search(chunk.text), ensure_ascii=False),
                    chunk.source.model_dump_json(),
                    json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    chunk.chunk_hash,
                    chunk.index_key,
                )
                for chunk in chunks
            ],
        )

    def active_batches(self, *, doc_id: str | None = None) -> list[SearchIndexBatch]:
        sql = """
            SELECT b.* FROM search_index_batches b
            JOIN documents d ON d.current_manifest_id=b.manifest_id
            WHERE d.deleted=0 AND b.status='ready'
        """
        params: list[str] = []
        if doc_id is not None:
            sql += " AND d.doc_id=?"
            params.append(doc_id)
        sql += " ORDER BY d.doc_id"
        rows = self._conn.execute(sql, params).fetchall()
        return [SearchIndexBatch.model_validate(dict(row)) for row in rows]

    def search_batch(self, manifest_id: str) -> SearchIndexBatch | None:
        row = self._conn.execute(
            "SELECT * FROM search_index_batches WHERE manifest_id=?", (manifest_id,)
        ).fetchone()
        return SearchIndexBatch.model_validate(dict(row)) if row else None

    def active_bm25_documents(self, *, doc_id: str | None = None) -> list[BM25IndexDocument]:
        sql = """
            SELECT b.* FROM bm25_index_documents b
            JOIN documents d ON d.current_manifest_id=b.manifest_id
            JOIN search_index_batches s ON s.manifest_id=b.manifest_id
            WHERE d.deleted=0 AND s.status='ready'
        """
        params: list[str] = []
        if doc_id is not None:
            sql += " AND d.doc_id=?"
            params.append(doc_id)
        sql += " ORDER BY d.doc_id, b.ordinal, b.logical_key"
        return [self._row_to_bm25(row) for row in self._conn.execute(sql, params).fetchall()]

    def chroma_store(self, batch: SearchIndexBatch):
        """返回绑定批次的 LangChain Chroma VectorStore。"""
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:  # pragma: no cover - 最小安装环境
            raise RuntimeError("混合检索不可用：请安装 scoreproof[retrieval]") from exc

        return Chroma(
            client=self._chroma_client(),
            collection_name=batch.chroma_collection,
            embedding_function=self.embeddings,
            create_collection_if_not_exists=False,
        )

    def _partition_embeddings(
        self,
        *,
        chunks: Sequence[StoredChunk],
        previous_manifest_id: str | None,
        previous_chunks: dict[str, StoredChunk],
    ) -> tuple[dict[str, list[float]], list[StoredChunk]]:
        reusable_keys = [
            chunk.logical_key
            for chunk in chunks
            if chunk.logical_key in previous_chunks
            and previous_chunks[chunk.logical_key].index_key == chunk.index_key
        ]
        if not previous_manifest_id or not reusable_keys:
            return {}, list(chunks)
        batch = self.search_batch(previous_manifest_id)
        if batch is None:
            return {}, list(chunks)
        try:
            old_collection = self._chroma_client().get_collection(batch.chroma_collection)
            old_ids = [_record_id(previous_manifest_id, key) for key in reusable_keys]
            records = old_collection.get(ids=old_ids, include=["embeddings", "metadatas"])
        except Exception:
            return {}, list(chunks)
        ids = records.get("ids") or []
        embeddings = records.get("embeddings")
        if embeddings is None or len(ids) != len(reusable_keys):
            return {}, list(chunks)
        id_to_vector = {record_id: list(vector) for record_id, vector in zip(ids, embeddings, strict=True)}
        reused: dict[str, list[float]] = {}
        for key in reusable_keys:
            vector = id_to_vector.get(_record_id(previous_manifest_id, key))
            if vector is not None:
                reused[key] = vector
        to_embed = [chunk for chunk in chunks if chunk.logical_key not in reused]
        return reused, to_embed

    def _batch_is_complete(self, manifest_id: str) -> bool:
        batch = self.search_batch(manifest_id)
        if batch is None or batch.status != "ready":
            return False
        bm25_count = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM bm25_index_documents WHERE manifest_id=?",
                (manifest_id,),
            ).fetchone()[0]
        )
        if bm25_count != batch.bm25_doc_count:
            return False
        try:
            vector_count = self._chroma_client().get_collection(batch.chroma_collection).count()
        except Exception:
            return False
        return vector_count == batch.vector_doc_count == batch.bm25_doc_count

    def _new_collection(self, name: str):
        client = self._chroma_client()
        with suppress(Exception):
            client.delete_collection(name)
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _chroma_client(self):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 最小安装环境
            raise RuntimeError("向量索引不可用：请安装 scoreproof[retrieval]") from exc

        return chromadb.PersistentClient(path=str(self.vector_dir))

    def _drop_staged_collection(self) -> None:
        if not self._staged_collection:
            return
        try:
            self._chroma_client().delete_collection(self._staged_collection)
        except Exception:
            pass
        finally:
            self._staged_collection = None

    @staticmethod
    def _row_to_bm25(row: sqlite3.Row) -> BM25IndexDocument:
        return BM25IndexDocument(
            manifest_id=row["manifest_id"],
            logical_key=row["logical_key"],
            doc_id=row["doc_id"],
            ordinal=row["ordinal"],
            text=row["text"],
            tokens=json.loads(row["tokens"]),
            source=json.loads(row["source"]),
            metadata=json.loads(row["metadata"]),
            chunk_hash=row["chunk_hash"],
            index_key=row["index_key"],
        )


def _collection_name(manifest_id: str) -> str:
    return f"scoreproof_{manifest_id}"


def _record_id(manifest_id: str, logical_key: str) -> str:
    digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()[:16]
    return f"{manifest_id}_{digest}"


def _chroma_metadata(*, manifest_id: str, doc_id: str, chunk: StoredChunk) -> dict[str, Any]:
    return {
        "manifest_id": manifest_id,
        "logical_key": chunk.logical_key,
        "doc_id": doc_id,
        "ordinal": chunk.ordinal,
        "kind": chunk.kind,
        "index_key": chunk.index_key,
        "source": chunk.source.model_dump_json(),
        "metadata": json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
    }


__all__ = [
    "BM25IndexDocument",
    "EmbeddingFunction",
    "HashingEmbeddings",
    "HybridIndexManifestStore",
    "SEARCH_INDEX_SCHEMA_SQL",
    "SearchIndexBatch",
]
