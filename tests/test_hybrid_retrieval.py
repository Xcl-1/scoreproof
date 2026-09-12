"""同批次 BM25/Chroma 索引与 RRF 混合召回测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from typer.testing import CliRunner

from scoreproof.cli import _pdf_index_chunks, app
from scoreproof.indexing import (
    DocumentChunk,
    HashingEmbeddings,
    HybridIndexManifestStore,
    IndexManifestStore,
)
from scoreproof.ingest.pdf_loader import PDFPage
from scoreproof.retrieval import HybridRetriever, RetrievalHit, Router, reciprocal_rank_fusion
from scoreproof.retrieval.query import rewrite_retrieval_query
from scoreproof.retrieval.router import Clause
from scoreproof.schema import Ruleset, SourceRef

from .conftest import make_claim


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.delegate = HashingEmbeddings(dimensions=64)
        self.document_count = 0
        self.query_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_count += len(texts)
        return self.delegate.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_count += 1
        return self.delegate.embed_query(text)


def _chunks(*, changed: bool = False, college: str = "计算机学院") -> list[DocumentChunk]:
    texts = [
        "学科竞赛国家级一等奖计十五分" if not changed else "学科竞赛国家级一等奖计十六分",
        "志愿服务每满二十小时计一分，最高五分",
        "文体活动校级一等奖计三分",
        "知识产权第一专利人可申请创新加分",
    ]
    return [
        DocumentChunk(
            logical_key=f"page:{index}",
            text=text,
            kind="page",
            source=SourceRef(doc="细则.pdf", page=index, text=text),
            metadata={"academic_year": "2025-2026", "college": college},
        )
        for index, text in enumerate(texts, start=1)
    ]


def _store(tmp_path: Path, embeddings: Embeddings | None = None) -> HybridIndexManifestStore:
    return HybridIndexManifestStore(
        tmp_path / "index.sqlite",
        vector_dir=tmp_path / "chroma",
        embeddings=embeddings,
        embedding_model="test-hash-v1",
    )


def _sync(store: HybridIndexManifestStore, chunks: list[DocumentChunk], content: bytes = b"v1"):
    return store.sync_document(
        doc_id="rules",
        source_path="细则.pdf",
        document_bytes=content,
        chunks=chunks,
        embedding_model="test-hash-v1",
    )


def test_first_sync_builds_complete_same_batch(tmp_path: Path) -> None:
    embeddings = CountingEmbeddings()
    with _store(tmp_path, embeddings) as store:
        result = _sync(store, _chunks())
        batch = store.search_batch(result.manifest_id or "")
        assert batch is not None
        assert batch.status == "ready"
        assert batch.bm25_doc_count == batch.vector_doc_count == 4
        assert batch.embedded_count == 4
        assert len(store.active_bm25_documents()) == 4
        assert store.chroma_store(batch)._collection.count() == 4
    assert embeddings.document_count == 4


def test_unchanged_sync_builds_no_new_batch(tmp_path: Path) -> None:
    embeddings = CountingEmbeddings()
    with _store(tmp_path, embeddings) as store:
        first = _sync(store, _chunks())
        second = _sync(store, _chunks())
        assert second.action == "unchanged"
        assert second.manifest_id == first.manifest_id
        assert store.manifest_count("rules") == 1
        assert len(store.active_batches()) == 1
    assert embeddings.document_count == 4


def test_changed_chunk_only_calls_one_new_embedding(tmp_path: Path) -> None:
    embeddings = CountingEmbeddings()
    with _store(tmp_path, embeddings) as store:
        first = _sync(store, _chunks())
        second = _sync(store, _chunks(changed=True), content=b"v2")
        assert second.diff.changed_keys == ["page:1"]
        batch = store.search_batch(second.manifest_id or "")
        assert batch is not None and batch.embedded_count == 1
        assert store.search_batch(first.manifest_id or "") is not None
    assert embeddings.document_count == 5


def test_metadata_only_change_reuses_every_embedding(tmp_path: Path) -> None:
    embeddings = CountingEmbeddings()
    with _store(tmp_path, embeddings) as store:
        _sync(store, _chunks())
        second = _sync(store, _chunks(college="大数据学院"), content=b"v2")
        assert second.diff.metadata_only_keys == ["page:1", "page:2", "page:3", "page:4"]
        batch = store.search_batch(second.manifest_id or "")
        assert batch is not None and batch.embedded_count == 0
    assert embeddings.document_count == 4


def test_same_text_at_two_locations_has_two_vector_records(tmp_path: Path) -> None:
    chunks = _chunks()
    chunks[1] = chunks[1].model_copy(update={"text": chunks[0].text})
    with _store(tmp_path) as store:
        result = _sync(store, chunks)
        batch = store.search_batch(result.manifest_id or "")
        assert batch is not None
        assert store.chroma_store(batch)._collection.count() == 4


class BrokenEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError


def test_vector_failure_keeps_manifest_unpublished(tmp_path: Path) -> None:
    with _store(tmp_path, BrokenEmbeddings()) as store:
        with pytest.raises(RuntimeError, match="unavailable"):
            _sync(store, _chunks())
        assert store.manifest_count("rules") == 0
        assert store.active_batches() == []
        assert store._chroma_client().list_collections() == []


def test_delete_removes_batch_from_online_search(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _sync(store, _chunks())
        assert HybridRetriever(store).search("志愿服务")
        store.delete_document("rules")
        assert HybridRetriever(store).search("志愿服务") == []


def test_rollback_switches_bm25_and_vector_together(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        first = _sync(store, _chunks())
        _sync(store, _chunks(changed=True), content=b"v2")
        assert "十六分" in store.active_bm25_documents()[0].text
        result = store.rollback("rules", manifest_id=first.manifest_id)
        assert result.action == "rolled_back"
        assert "十五分" in store.active_bm25_documents()[0].text
        assert store.active_batches()[0].manifest_id == first.manifest_id


def test_hybrid_retriever_runs_both_channels_and_rrf(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _sync(store, _chunks())
        retriever = HybridRetriever(store)
        bm25 = retriever.search_bm25("志愿服务", top_k=3)
        vector = retriever.search_vector("志愿服务", top_k=3)
        fused = retriever.search("志愿服务", top_k=3)
        assert bm25 and bm25[0].channel == "bm25"
        assert vector and vector[0].channel == "vector"
        assert fused and fused[0].channel == "rrf"
        assert fused[0].clause.id.endswith("page:2")
        assert fused[0].component_ranks == {"bm25": 1, "vector": 1}


def test_query_rewrite_is_applied_only_to_bm25_channel(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            logical_key=f"page:{index}",
            text=text,
            kind="page",
            source=SourceRef(doc="rules.pdf", page=index),
        )
        for index, text in enumerate(
            ["推免资格审查办法", "学科竞赛评分办法", "志愿服务认定办法", "知识产权计分办法"],
            start=1,
        )
    ]
    with _store(tmp_path) as store:
        _sync(store, chunks)
        retriever = HybridRetriever(store, query_rewriter=rewrite_retrieval_query)
        assert retriever.search_bm25("免试研究生")[0].clause.id == "rules:page:1"


def test_hybrid_filter_excludes_other_college(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _sync(store, _chunks(college="计算机学院"))
        retriever = HybridRetriever(store, college="外国语学院")
        assert retriever.search_bm25("志愿服务") == []
        assert retriever.search_vector("志愿服务") == []


def test_rrf_rewards_cross_channel_hit_and_is_deterministic() -> None:
    source = SourceRef(doc="x.pdf", page=1)
    a = Clause(id="a", text="A", source=source)
    b = Clause(id="b", text="B", source=source)
    c = Clause(id="c", text="C", source=source)
    left = [
        RetrievalHit(a, 10, rank=1, channel="bm25"),
        RetrievalHit(b, 9, rank=2, channel="bm25"),
    ]
    right = [
        RetrievalHit(c, 1, rank=1, channel="vector"),
        RetrievalHit(b, 0.9, rank=2, channel="vector"),
    ]
    fused = reciprocal_rank_fusion([left, right], rrf_k=60, top_k=3)
    assert [hit.clause.id for hit in fused] == ["b", "a", "c"]
    assert fused[0].component_ranks == {"bm25": 2, "vector": 2}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_k": 0},
        {"top_k": 0},
        {"weights": [1.0]},
        {"weights": [1.0, 0.0]},
    ],
)
def test_rrf_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[], []], **kwargs)


def test_router_keeps_structured_channel_first(base_ruleset: Ruleset, tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _sync(store, _chunks())
        embeddings = store.embeddings
        router = Router(
            base_ruleset,
            academic_year="2025-2026",
            retriever=HybridRetriever(store),
        )
        result = router.route(make_claim("省二等奖", level="省级二等奖"))
        assert result.channel == "structured" and result.matched
        if isinstance(embeddings, CountingEmbeddings):  # pragma: no cover
            assert embeddings.query_count == 0


def test_sync_rejects_embedding_model_mismatch(tmp_path: Path) -> None:
    with _store(tmp_path) as store, pytest.raises(ValueError, match="不一致"):
        store.sync_document(
            doc_id="rules",
            source_path="x.pdf",
            document_bytes=b"x",
            chunks=_chunks(),
            embedding_model="another-model",
        )


def test_plain_manifest_is_backfilled_on_hybrid_upgrade(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    with IndexManifestStore(db) as plain:
        first = plain.sync_document(
            doc_id="rules",
            source_path="细则.pdf",
            document_bytes=b"v1",
            chunks=_chunks(),
            embedding_model="test-hash-v1",
        )
    embeddings = CountingEmbeddings()
    with _store(tmp_path, embeddings) as hybrid:
        upgraded = _sync(hybrid, _chunks())
        assert upgraded.action == "published"
        assert upgraded.manifest_id != first.manifest_id
        batch = hybrid.search_batch(upgraded.manifest_id or "")
        assert batch is not None and batch.vector_doc_count == 4
    assert embeddings.document_count == 4


def test_missing_chroma_snapshot_is_rebuilt_without_changing_source(tmp_path: Path) -> None:
    embeddings = CountingEmbeddings()
    with _store(tmp_path, embeddings) as store:
        first = _sync(store, _chunks())
        batch = store.search_batch(first.manifest_id or "")
        assert batch is not None
        store._chroma_client().delete_collection(batch.chroma_collection)
        repaired = _sync(store, _chunks())
        assert repaired.action == "published"
        repaired_batch = store.search_batch(repaired.manifest_id or "")
        assert repaired_batch is not None
        assert store.chroma_store(repaired_batch)._collection.count() == 4
    assert embeddings.document_count == 8


def test_hybrid_sync_and_search_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "rules.pdf"
    pdf.write_bytes(b"fake-pdf")
    monkeypatch.setattr(
        "scoreproof.cli.load_pdf",
        lambda path: [
            PDFPage(
                page=index,
                text=chunk.text,
                blocks=[{"text": chunk.text, "bbox": (0, 0, 10, 10)}],
                meta={"doc": "rules.pdf"},
            )
            for index, chunk in enumerate(_chunks(), start=1)
        ],
    )
    db = tmp_path / "index.sqlite"
    vectors = tmp_path / "vectors"
    runner = CliRunner()
    sync_result = runner.invoke(
        app,
        [
            "sync-pdf-hybrid",
            str(pdf),
            "--doc-id",
            "rules",
            "--year",
            "2025-2026",
            "--college",
            "计算机学院",
            "--embedding-model",
            "scoreproof-hash-v1",
            "--db",
            str(db),
            "--vector-dir",
            str(vectors),
        ],
    )
    assert sync_result.exit_code == 0, sync_result.output
    assert '"bm25_doc_count": 4' in sync_result.output

    search_result = runner.invoke(
        app,
        [
            "search-index",
            "志愿服务",
            "--db",
            str(db),
            "--vector-dir",
            str(vectors),
        ],
    )
    assert search_result.exit_code == 0, search_result.output
    assert "rules:page:2" in search_result.output

    class FakeReranker:
        model_version = "fake"

        def __init__(self, **kwargs) -> None:
            pass

        def score(self, query: str, documents: list[str]) -> list[float]:
            return [float(-index) for index in range(len(documents))]

    monkeypatch.setattr("scoreproof.cli.FastEmbedReranker", FakeReranker)
    rerank_result = runner.invoke(
        app,
        [
            "search-index",
            "志愿服务",
            "--rerank",
            "--db",
            str(db),
            "--vector-dir",
            str(vectors),
        ],
    )
    assert rerank_result.exit_code == 0, rerank_result.output
    assert '"channel": "rerank"' in rerank_result.output


def test_pdf_table_level_is_prefixed_to_following_rows(tmp_path: Path) -> None:
    page = PDFPage(
        page=5,
        text="国家级\n一等奖 4分 3分\n二等奖 3分 2分",
        blocks=[
            {"text": "国家级", "bbox": (0, 0, 1, 1)},
            {"text": "一等奖 4分 3分", "bbox": (0, 1, 1, 2)},
            {"text": "二等奖 3分 2分", "bbox": (0, 2, 1, 3)},
        ],
        meta={"doc": "rules.pdf"},
    )
    chunks = _pdf_index_chunks(
        pdf=tmp_path / "rules.pdf",
        pages=[page],
        chunk_mode="block",
        academic_year="2025-2026",
        college="计算机学院",
    )
    assert chunks[1].text == "国家级\n一等奖 4分 3分"
    assert chunks[1].source.text == "一等奖 4分 3分"
    assert chunks[1].metadata["original_text"] == "一等奖 4分 3分"
