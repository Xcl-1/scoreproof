"""复杂 PDF：多栏阅读顺序、跨页表拼接与正式评测门禁。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.eval.pdf_regression import (
    PDFRegressionCase,
    PDFRegressionDataset,
    PDFRegressionReport,
    PDFScenario,
    evaluate_pdf_regression,
)
from scoreproof.ingest.pdf_loader import (
    PDFDocument,
    PDFLogicalTable,
    PDFPage,
    PDFTableFragment,
    load_pdf_document,
    reorder_text_blocks,
    stitch_cross_page_tables,
)


def _block(text: str, bbox: tuple[float, float, float, float]) -> dict:
    return {"text": text, "bbox": bbox, "block_no": 0}


def _make_two_column_pdf(path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((40, 40), "DEMO HEADER", fontsize=12)
    page.insert_text((40, 100), "LEFT-ONE " + "A" * 28, fontsize=12)
    page.insert_text((40, 140), "LEFT-TWO " + "B" * 28, fontsize=12)
    page.insert_text((330, 100), "RIGHT-ONE " + "C" * 24, fontsize=12)
    page.insert_text((330, 140), "RIGHT-TWO " + "D" * 24, fontsize=12)
    page.insert_text((40, 760), "DEMO FOOTER", fontsize=12)
    document.save(path)
    document.close()


def test_two_column_blocks_are_reordered_left_then_right() -> None:
    blocks = [
        _block("HEADER", (20, 10, 580, 30)),
        _block("LEFT-ONE", (30, 80, 270, 110)),
        _block("RIGHT-ONE", (330, 80, 570, 110)),
        _block("LEFT-TWO", (30, 130, 270, 160)),
        _block("RIGHT-TWO", (330, 130, 570, 160)),
        _block("FOOTER", (20, 740, 580, 760)),
    ]
    ordered, detected = reorder_text_blocks(blocks, 600)
    assert detected is True
    assert [block["text"] for block in ordered] == [
        "HEADER",
        "LEFT-ONE",
        "LEFT-TWO",
        "RIGHT-ONE",
        "RIGHT-TWO",
        "FOOTER",
    ]


def test_narrow_table_cells_do_not_masquerade_as_two_columns() -> None:
    blocks = [
        _block("A", (30, 80, 80, 100)),
        _block("B", (130, 80, 180, 100)),
        _block("C", (330, 80, 380, 100)),
        _block("D", (430, 80, 480, 100)),
    ]
    _, detected = reorder_text_blocks(blocks, 600)
    assert detected is False


def test_cross_page_edge_fragments_are_stitched_and_keep_regions() -> None:
    pages = [
        PDFPage(page=1, text="表格第一页", meta={"page_height": 800}),
        PDFPage(page=2, text="表格第二页", meta={"page_height": 800}),
    ]
    fragments = {
        1: [
            PDFTableFragment(
                page=1,
                table_index=0,
                cells=[["项目", "分值"], ["论文 A", "5"]],
                bbox=(40, 650, 560, 790),
                page_width=600,
                page_height=800,
            )
        ],
        2: [
            PDFTableFragment(
                page=2,
                table_index=0,
                cells=[["论文 B", "3"]],
                bbox=(40, 20, 560, 180),
                page_width=600,
                page_height=800,
            )
        ],
    }
    tables, issues = stitch_cross_page_tables(pages, fragments)
    assert issues == []
    assert len(tables) == 1
    assert tables[0].spans_pages is True
    assert tables[0].rows[-1] == ["论文 B", "3"]
    assert tables[0].continuation_reasons == ["page_edge_continuation"]
    assert tables[0].fragments[1].bbox == (40, 20, 560, 180)


def test_ambiguous_adjacent_tables_remain_separate_and_visible() -> None:
    pages = [PDFPage(page=1, text="甲"), PDFPage(page=2, text="乙")]
    fragments = {
        1: [PDFTableFragment(page=1, table_index=0, cells=[["A", "B"]])],
        2: [PDFTableFragment(page=2, table_index=0, cells=[["C", "D"]])],
    }
    tables, issues = stitch_cross_page_tables(pages, fragments)
    assert len(tables) == 2
    assert issues and "人工复核" in issues[0]


def test_regression_schema_rejects_traversal_and_missing_expectations() -> None:
    with pytest.raises(ValidationError):
        PDFRegressionCase(
            case_id="bad",
            document="../private.pdf",
            scenario="cross_page_table",
            real_document=True,
        )


def test_blank_real_pdf_is_visible_as_scanned_and_requires_review(tmp_path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "blank-scan.pdf"
    source = pymupdf.open()
    source.new_page(width=600, height=800)
    source.save(path)
    source.close()
    document = load_pdf_document(path)
    assert document.document_kind == "scanned"
    assert document.scanned_pages == [1]
    assert document.review_issues and "OCR 或人工复核" in document.review_issues[0]


def test_real_generated_pdf_cli_is_explicitly_smoke_only(tmp_path: Path) -> None:
    pdf = tmp_path / "two-column.pdf"
    _make_two_column_pdf(pdf)
    document = load_pdf_document(pdf)
    assert document.multi_column_pages == [1]
    assert document.pages[0].text.index("LEFT-TWO") < document.pages[0].text.index("RIGHT-ONE")

    dataset = PDFRegressionDataset(
        dataset_version="synthetic-two-column-v1",
        independent_real_documents=False,
        cases=[
            PDFRegressionCase(
                case_id="two-column-1",
                document=pdf.name,
                scenario="multi_column",
                real_document=False,
                synthetic=True,
                expected_ordered_fragments=[
                    "LEFT-ONE",
                    "LEFT-TWO",
                    "RIGHT-ONE",
                    "RIGHT-TWO",
                ],
            )
        ],
    )
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "eval-complex-pdf",
            str(dataset_path),
            "--out",
            str(report_path),
            "--no-enforce",
        ],
    )
    assert result.exit_code == 0, result.output
    report = PDFRegressionReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.overall_success_rate == 1.0
    assert report.sample_size == 1
    assert report.formal_gate_eligible is False
    assert report.passed is False
    assert report.smoke_test_only is True


def test_same_document_across_scenarios_cannot_satisfy_formal_sample_gate(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = tmp_path / "same.pdf"
    pdf.write_bytes(b"placeholder")

    class _FakeDocument:
        path = pdf
        sha256 = "a" * 64
        pages = [PDFPage(page=1, text="LEFT RIGHT", raw_text="LEFT RIGHT")]
        logical_tables = []
        review_issues = []
        scanned_pages = [1]
        multi_column_pages = [1]

    monkeypatch.setattr(
        "scoreproof.eval.pdf_regression.load_pdf_document", lambda _: _FakeDocument()
    )
    dataset = PDFRegressionDataset(
        dataset_version="duplicate-document-v1",
        authorization_reference="public:test",
        independent_real_documents=True,
        cases=[
            PDFRegressionCase(
                case_id="multi",
                document=pdf.name,
                scenario="multi_column",
                real_document=True,
                expected_ordered_fragments=["LEFT", "RIGHT"],
            ),
            PDFRegressionCase(
                case_id="scan",
                document=pdf.name,
                scenario="scanned",
                real_document=True,
                expected_scanned_pages=[1],
            ),
        ],
    )
    report = evaluate_pdf_regression(dataset, base_dir=tmp_path)
    assert report.sample_size == 1
    assert report.formal_gate_eligible is False


def test_formal_gate_requires_and_accepts_ten_unique_documents_per_scenario(
    tmp_path: Path, monkeypatch
) -> None:
    cases: list[PDFRegressionCase] = []
    scenarios: tuple[PDFScenario, ...] = ("multi_column", "cross_page_table", "scanned")
    for scenario in scenarios:
        for index in range(10):
            name = f"{scenario}-{index}.pdf"
            kwargs: dict[str, object] = {}
            if scenario == "multi_column":
                kwargs["expected_ordered_fragments"] = ["LEFT", "RIGHT"]
            elif scenario == "cross_page_table":
                kwargs["expected_table_fragments"] = ["TABLE"]
                kwargs["expected_min_table_rows"] = 1
            else:
                kwargs["expected_scanned_pages"] = [1]
            cases.append(
                PDFRegressionCase(
                    case_id=f"{scenario}-{index}",
                    document=name,
                    scenario=scenario,
                    real_document=True,
                    **kwargs,
                )
            )

    def fake_load(path: Path) -> PDFDocument:
        scenario = path.name.rsplit("-", 1)[0]
        page = PDFPage(
            page=1,
            text="" if scenario == "scanned" else "LEFT RIGHT",
            raw_text="" if scenario == "scanned" else "LEFT RIGHT",
            meta={"layout": "two_column" if scenario == "multi_column" else "single_column"},
        )
        tables = (
            [PDFLogicalTable(id="t1", start_page=1, end_page=2, rows=[["TABLE", "VALUE"]])]
            if scenario == "cross_page_table"
            else []
        )
        return PDFDocument(
            path=path,
            sha256=hashlib.sha256(path.name.encode()).hexdigest(),
            pages=[page],
            logical_tables=tables,
        )

    monkeypatch.setattr("scoreproof.eval.pdf_regression.load_pdf_document", fake_load)
    dataset = PDFRegressionDataset(
        dataset_version="formal-shape-v1",
        authorization_reference="approval:formal-test",
        independent_real_documents=True,
        cases=cases,
    )
    report = evaluate_pdf_regression(dataset, base_dir=tmp_path)
    assert report.sample_size == 30
    assert report.formal_gate_eligible is True
    assert report.passed is True
    assert all(metric.sample_size == 10 for metric in report.scenario_metrics)
