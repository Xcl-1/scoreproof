"""双级 Hash、manifest 增量同步、删除与回滚。"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.errors import VersionConflict
from scoreproof.indexing import DocumentChunk, IndexManifestStore, document_hash, index_key
from scoreproof.ingest.pdf_loader import PDFPage
from scoreproof.schema import SourceRef


def chunk(key: str, text: str, *, page: int) -> DocumentChunk:
    return DocumentChunk(
        logical_key=key,
        text=text,
        kind="page",
        source=SourceRef(doc="rules.pdf", page=page, text=text),
    )


@pytest.fixture
def store(tmp_path: Path) -> IndexManifestStore:
    with IndexManifestStore(tmp_path / "index.sqlite") as value:
        yield value


def sync(
    store: IndexManifestStore,
    chunks: list[DocumentChunk],
    *,
    raw: bytes = b"document-v1",
    model: str = "embedding-v1",
):
    return store.sync_document(
        doc_id="rules-pdf",
        source_path="rules.pdf",
        document_bytes=raw,
        chunks=chunks,
        embedding_model=model,
    )


class TestHashes:
    def test_document_hash_uses_original_bytes(self) -> None:
        assert document_hash(b"a") == document_hash(b"a")
        assert document_hash(b"a") != document_hash(b"a ")

    def test_index_key_locks_document_chunk_and_model(self) -> None:
        digest = document_hash(b"chunk")
        assert index_key("doc", digest, "v1") == index_key("doc", digest, "v1")
        assert index_key("doc", digest, "v1") != index_key("doc", digest, "v2")


class TestManifestSync:
    def test_first_sync_publishes_complete_snapshot(self, store: IndexManifestStore) -> None:
        result = sync(store, [chunk("page:1", "第一页", page=1), chunk("page:2", "第二页", page=2)])

        assert result.action == "published"
        assert result.diff.rebuild_keys == ["page:1", "page:2"]
        assert [item.logical_key for item in store.active_chunks(doc_id="rules-pdf")] == [
            "page:1",
            "page:2",
        ]

    def test_same_document_is_idempotent(self, store: IndexManifestStore) -> None:
        chunks = [chunk("page:1", "第一页", page=1)]
        first = sync(store, chunks)
        second = sync(store, chunks)

        assert second.action == "unchanged"
        assert second.manifest_id == first.manifest_id
        assert second.diff.rebuild_keys == []
        assert store.manifest_count("rules-pdf") == 1

    def test_only_changed_page_is_rebuilt(self, store: IndexManifestStore) -> None:
        sync(store, [chunk("page:1", "第一页", page=1), chunk("page:2", "第二页", page=2)])
        result = sync(
            store,
            [chunk("page:1", "第一页", page=1), chunk("page:2", "第二页已修订", page=2)],
            raw=b"document-v2",
        )

        assert result.diff.changed_keys == ["page:2"]
        assert result.diff.unchanged_keys == ["page:1"]
        assert result.diff.rebuild_keys == ["page:2"]

    def test_removed_page_disappears_from_active_snapshot(self, store: IndexManifestStore) -> None:
        sync(store, [chunk("page:1", "第一页", page=1), chunk("page:2", "第二页", page=2)])
        result = sync(store, [chunk("page:1", "第一页", page=1)], raw=b"document-v2")

        assert result.diff.removed_keys == ["page:2"]
        assert [item.logical_key for item in store.active_chunks()] == ["page:1"]

    def test_raw_file_change_without_chunk_change_skips_rebuild(
        self, store: IndexManifestStore
    ) -> None:
        chunks = [chunk("page:1", "正文不变", page=1)]
        sync(store, chunks, raw=b"header-v1 + body")
        result = sync(store, chunks, raw=b"header-v2 + body")

        assert result.action == "published"
        assert result.diff.rebuild_keys == []
        assert result.diff.unchanged_keys == ["page:1"]

    def test_embedding_model_change_rebuilds_all_chunks(self, store: IndexManifestStore) -> None:
        chunks = [chunk("page:1", "第一页", page=1), chunk("page:2", "第二页", page=2)]
        sync(store, chunks, model="embedding-v1")
        result = sync(store, chunks, model="embedding-v2")

        assert result.diff.changed_keys == ["page:1", "page:2"]
        assert result.diff.rebuild_keys == ["page:1", "page:2"]

    def test_duplicate_logical_keys_are_rejected(self, store: IndexManifestStore) -> None:
        with pytest.raises(ValueError, match="logical_key 重复"):
            sync(store, [chunk("page:1", "甲", page=1), chunk("page:1", "乙", page=1)])

    def test_empty_document_is_rejected(self, store: IndexManifestStore) -> None:
        with pytest.raises(ValueError, match="至少需要一个"):
            sync(store, [])

    def test_delete_removes_document_from_active_index(self, store: IndexManifestStore) -> None:
        sync(store, [chunk("page:1", "第一页", page=1)])
        result = store.delete_document("rules-pdf")

        assert result.action == "deleted"
        assert store.active_chunks(doc_id="rules-pdf") == []

    def test_rollback_restores_previous_complete_snapshot(self, store: IndexManifestStore) -> None:
        first = sync(store, [chunk("page:1", "旧正文", page=1)])
        second = sync(store, [chunk("page:1", "新正文", page=1)], raw=b"document-v2")
        result = store.rollback("rules-pdf")

        assert result.action == "rolled_back"
        assert result.manifest_id == first.manifest_id
        assert result.previous_manifest_id == second.manifest_id
        assert store.active_chunks(doc_id="rules-pdf")[0].text == "旧正文"

    def test_failure_before_publish_leaves_no_half_snapshot(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        with IndexManifestStore(tmp_path / "atomic.sqlite") as store:
            monkeypatch.setattr(
                store,
                "_before_publish",
                lambda manifest_id: (_ for _ in ()).throw(RuntimeError(manifest_id)),
            )
            with pytest.raises(RuntimeError):
                sync(store, [chunk("page:1", "第一页", page=1)])

            assert store.active_chunks() == []
            assert store.manifest_count("rules-pdf") == 0

    def test_stale_expected_manifest_rejects_publish_and_delete(
        self, store: IndexManifestStore
    ) -> None:
        first = sync(store, [chunk("page:1", "第一版", page=1)])
        second = sync(store, [chunk("page:1", "第二版", page=1)], raw=b"document-v2")

        with pytest.raises(VersionConflict, match="manifest 已变化"):
            store.sync_document(
                doc_id="rules-pdf",
                source_path="rules.pdf",
                document_bytes=b"document-v3",
                chunks=[chunk("page:1", "第三版", page=1)],
                embedding_model="embedding-v1",
                expected_current_manifest=first.manifest_id,
            )
        with pytest.raises(VersionConflict, match="manifest 已变化"):
            store.delete_document(
                "rules-pdf", expected_current_manifest=first.manifest_id
            )

        assert store.manifest_count("rules-pdf") == 2
        assert store.active_chunks()[0].text == "第二版"
        assert second.manifest_id is not None

    def test_two_processes_publish_same_snapshot_once(self, tmp_path: Path) -> None:
        worker = textwrap.dedent(
            """
            import json
            import sys
            import time
            from scoreproof.indexing import DocumentChunk, IndexManifestStore
            from scoreproof.schema import SourceRef

            class SlowStore(IndexManifestStore):
                def _before_publish(self, manifest_id):
                    time.sleep(float(sys.argv[2]))

            chunk = DocumentChunk(
                logical_key="page:1",
                text="跨进程同源发布",
                kind="page",
                source=SourceRef(doc="rules.pdf", page=1, text="跨进程同源发布"),
            )
            with SlowStore(sys.argv[1]) as store:
                result = store.sync_document(
                    doc_id="rules-pdf",
                    source_path="rules.pdf",
                    document_bytes=b"same-document",
                    chunks=[chunk],
                    embedding_model="embedding-v1",
                )
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
            """
        )
        db = tmp_path / "concurrent.sqlite"
        first = subprocess.Popen(
            [sys.executable, "-c", worker, str(db), "0.6"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        second = subprocess.Popen(
            [sys.executable, "-c", worker, str(db), "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        first_out, first_err = first.communicate(timeout=30)
        second_out, second_err = second.communicate(timeout=30)
        assert first.returncode == 0, first_err
        assert second.returncode == 0, second_err
        actions = sorted(
            [json.loads(first_out)["action"], json.loads(second_out)["action"]]
        )
        assert actions == ["published", "unchanged"]
        with IndexManifestStore(db) as reopened:
            assert reopened.manifest_count("rules-pdf") == 1
            assert len(reopened.active_chunks(doc_id="rules-pdf")) == 1

    def test_abrupt_process_exit_rolls_back_sqlite_snapshot(self, tmp_path: Path) -> None:
        worker = textwrap.dedent(
            """
            import os
            import sys
            from scoreproof.indexing import DocumentChunk, IndexManifestStore
            from scoreproof.schema import SourceRef

            class CrashStore(IndexManifestStore):
                def _before_publish(self, manifest_id):
                    os._exit(91)

            chunk = DocumentChunk(
                logical_key="page:1",
                text="崩溃前不得发布",
                source=SourceRef(doc="rules.pdf", page=1),
            )
            with CrashStore(sys.argv[1]) as store:
                store.sync_document(
                    doc_id="rules-pdf",
                    source_path="rules.pdf",
                    document_bytes=b"crash-document",
                    chunks=[chunk],
                    embedding_model="embedding-v1",
                )
            """
        )
        db = tmp_path / "crash.sqlite"
        crashed = subprocess.run(
            [sys.executable, "-c", worker, str(db)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        assert crashed.returncode == 91, crashed.stderr
        with IndexManifestStore(db) as reopened:
            assert reopened.manifest_count("rules-pdf") == 0
            assert reopened.active_chunks() == []
            repaired = sync(reopened, [chunk("page:1", "恢复后发布", page=1)])
            assert repaired.action == "published"


class TestManifestCli:
    def test_sync_pdf_manifest_command(self, tmp_path: Path, monkeypatch) -> None:
        pdf = tmp_path / "rules.pdf"
        pdf.write_bytes(b"fake-pdf-v1")
        monkeypatch.setattr(
            "scoreproof.cli.load_pdf",
            lambda path: [
                PDFPage(
                    page=1,
                    text="省级二等奖加8分",
                    blocks=[{"text": "省级二等奖加8分", "bbox": (0, 0, 10, 10)}],
                    meta={"doc": "rules.pdf", "total_pages": 1},
                )
            ],
        )
        db = tmp_path / "index.sqlite"

        result = CliRunner().invoke(
            app,
            [
                "sync-pdf-manifest",
                str(pdf),
                "--doc-id",
                "rules-pdf",
                "--db",
                str(db),
            ],
        )

        assert result.exit_code == 0, result.output
        with IndexManifestStore(db) as store:
            assert store.active_chunks(doc_id="rules-pdf")[0].logical_key == "page:1"
            assert store.manifest_count("rules-pdf") == 1

        stale = CliRunner().invoke(
            app,
            [
                "sync-pdf-manifest",
                str(pdf),
                "--doc-id",
                "rules-pdf",
                "--db",
                str(db),
                "--expected-manifest",
                "im_stale",
            ],
        )
        assert stale.exit_code == 2
        assert "version_conflict" in stale.output
        with IndexManifestStore(db) as store:
            assert store.manifest_count("rules-pdf") == 1

    def test_sync_pdf_manifest_refuses_partial_scanned_snapshot(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pdf = tmp_path / "mixed.pdf"
        pdf.write_bytes(b"fake-pdf-v1")
        monkeypatch.setattr(
            "scoreproof.cli.load_pdf",
            lambda path: [
                PDFPage(page=1, text="有文本", meta={"doc": "mixed.pdf"}),
                PDFPage(page=2, text="", meta={"doc": "mixed.pdf"}),
            ],
        )
        db = tmp_path / "index.sqlite"

        result = CliRunner().invoke(
            app,
            ["sync-pdf-manifest", str(pdf), "--doc-id", "mixed", "--db", str(db)],
        )

        assert result.exit_code == 2
        assert not db.exists()
