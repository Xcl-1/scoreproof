"""复杂 PDF 回归：多栏、跨页表格、扫描页的真实文档门禁。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ingest.pdf_loader import PDFDocument, load_pdf_document
from .gateway import wilson_interval

PDFScenario = Literal["multi_column", "cross_page_table", "scanned"]


class PDFRegressionCase(BaseModel):
    """单份公开或已授权文档的可复核期望，不保存文档正文。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    scenario: PDFScenario
    real_document: bool
    synthetic: bool = False
    expected_ordered_fragments: list[str] = Field(default_factory=list)
    expected_table_fragments: list[str] = Field(default_factory=list)
    expected_min_table_rows: int | None = Field(default=None, ge=1)
    expected_scanned_pages: list[int] = Field(default_factory=list)

    @field_validator("document")
    @classmethod
    def _document_is_relative_and_safe(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("document 必须是数据集目录内的安全相对路径")
        if path.suffix.lower() != ".pdf":
            raise ValueError("document 必须指向 PDF")
        return path.as_posix()

    @model_validator(mode="after")
    def _scenario_expectation_is_present(self) -> Self:
        if self.scenario == "multi_column" and len(self.expected_ordered_fragments) < 2:
            raise ValueError("多栏用例至少需要两个按阅读顺序排列的文本片段")
        if self.scenario == "cross_page_table" and (
            not self.expected_table_fragments or self.expected_min_table_rows is None
        ):
            raise ValueError("跨页表用例必须给出表格片段和最小逻辑行数")
        if self.scenario == "scanned" and not self.expected_scanned_pages:
            raise ValueError("扫描用例必须给出预期扫描页")
        return self


class PDFRegressionDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_version: str = Field(min_length=1)
    authorization_reference: str | None = None
    independent_real_documents: bool = False
    cases: list[PDFRegressionCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_cases(self) -> Self:
        ids = [case.case_id for case in self.cases]
        pairs = [(case.document, case.scenario) for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id 不得重复")
        if len(pairs) != len(set(pairs)):
            raise ValueError("同一文档的同一场景不得重复计数")
        return self


class PDFCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str
    document_name: str
    document_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scenario: PDFScenario
    passed: bool
    page_count: int = Field(ge=0)
    detected_multi_column_pages: list[int]
    detected_scanned_pages: list[int]
    spanning_table_count: int = Field(ge=0)
    reasons: list[str]


class PDFScenarioMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scenario: PDFScenario
    sample_size: int = Field(ge=0)
    passed: int = Field(ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    ci95: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> Self:
        if self.passed > self.sample_size:
            raise ValueError("passed 不能大于 sample_size")
        if self.sample_size == 0 and (self.success_rate is not None or self.ci95 is not None):
            raise ValueError("空场景不能填写成功率或区间")
        if self.sample_size > 0 and (self.success_rate is None or self.ci95 is None):
            raise ValueError("非空场景必须填写成功率和区间")
        return self


class PDFRegressionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    dataset_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_size: int = Field(ge=0, description="去重 PDF 文档数")
    case_count: int = Field(ge=0)
    real_documents: bool
    authorization_verified: bool
    independent_real_documents: bool
    scenario_metrics: list[PDFScenarioMetric]
    overall_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_ci95: tuple[float, float] | None = None
    formal_gate_eligible: bool
    passed: bool
    smoke_test_only: bool
    results: list[PDFCaseResult]
    limitations: list[str]

    @model_validator(mode="after")
    def _formal_flags_and_results_are_consistent(self) -> Self:
        if self.smoke_test_only == self.formal_gate_eligible:
            raise ValueError("smoke_test_only 必须与 formal_gate_eligible 相反")
        if self.passed and not self.formal_gate_eligible:
            raise ValueError("非正式报告不能标记 passed")
        ids = [result.case_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("结果 case_id 不得重复")
        if self.case_count != len(self.results):
            raise ValueError("case_count 与结果数量不一致")
        return self


def load_pdf_regression_dataset(path: str | Path) -> PDFRegressionDataset:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 PDF 回归数据集：{exc}") from exc
    return PDFRegressionDataset.model_validate(payload)


def _ordered_fragments(text: str, fragments: list[str]) -> bool:
    cursor = 0
    for fragment in fragments:
        position = text.find(fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(fragment)
    return True


def _table_text(rows: list[list[str]]) -> str:
    return "\n".join("\t".join(row) for row in rows)


def _evaluate_case(case: PDFRegressionCase, document: PDFDocument) -> PDFCaseResult:
    reasons: list[str] = []
    spanning = [table for table in document.logical_tables if table.spans_pages]
    if case.scenario == "multi_column":
        text = "\n".join(page.text for page in document.pages)
        if not document.multi_column_pages:
            reasons.append("未检测到双栏页面")
        if not _ordered_fragments(text, case.expected_ordered_fragments):
            reasons.append("预期文本片段未按阅读顺序出现")
    elif case.scenario == "cross_page_table":
        matching = [
            table
            for table in spanning
            if len(table.rows) >= (case.expected_min_table_rows or 0)
            and all(fragment in _table_text(table.rows) for fragment in case.expected_table_fragments)
        ]
        if not matching:
            reasons.append("没有跨页逻辑表同时满足片段与最小行数要求")
    else:
        missing = sorted(set(case.expected_scanned_pages) - set(document.scanned_pages))
        if missing:
            reasons.append(f"未识别预期扫描页：{missing}")
    return PDFCaseResult(
        case_id=case.case_id,
        document_name=document.path.name,
        document_sha256=document.sha256,
        scenario=case.scenario,
        passed=not reasons,
        page_count=len(document.pages),
        detected_multi_column_pages=document.multi_column_pages,
        detected_scanned_pages=document.scanned_pages,
        spanning_table_count=len(spanning),
        reasons=reasons,
    )


def _scenario_metric(scenario: PDFScenario, results: list[PDFCaseResult]) -> PDFScenarioMetric:
    selected = [result for result in results if result.scenario == scenario]
    passed = sum(result.passed for result in selected)
    return PDFScenarioMetric(
        scenario=scenario,
        sample_size=len(selected),
        passed=passed,
        success_rate=round(passed / len(selected), 4) if selected else None,
        ci95=wilson_interval(passed, len(selected)) if selected else None,
    )


def evaluate_pdf_regression(
    dataset: PDFRegressionDataset,
    *,
    base_dir: str | Path,
) -> PDFRegressionReport:
    """真实打开每份 PDF；同一文件 Hash 不会被重复计入正式样本量。"""

    root = Path(base_dir).resolve()
    results: list[PDFCaseResult] = []
    document_cache: dict[Path, PDFDocument] = {}
    input_hashes: dict[str, str] = {}
    for case in dataset.cases:
        document_path = (root / case.document).resolve()
        try:
            document_path.relative_to(root)
        except ValueError:
            results.append(
                PDFCaseResult(
                    case_id=case.case_id,
                    document_name=Path(case.document).name,
                    scenario=case.scenario,
                    passed=False,
                    page_count=0,
                    detected_multi_column_pages=[],
                    detected_scanned_pages=[],
                    spanning_table_count=0,
                    reasons=["文档路径越出数据集目录"],
                )
            )
            continue
        try:
            if document_path not in document_cache:
                document_cache[document_path] = load_pdf_document(document_path)
            document = document_cache[document_path]
            input_hashes[case.document] = document.sha256
            results.append(_evaluate_case(case, document))
        except Exception as exc:
            results.append(
                PDFCaseResult(
                    case_id=case.case_id,
                    document_name=document_path.name,
                    scenario=case.scenario,
                    passed=False,
                    page_count=0,
                    detected_multi_column_pages=[],
                    detected_scanned_pages=[],
                    spanning_table_count=0,
                    reasons=[f"解析失败：{type(exc).__name__}: {exc}"],
                )
            )

    scenario_metrics = [
        _scenario_metric(scenario, results)
        for scenario in ("multi_column", "cross_page_table", "scanned")
    ]
    passed_cases = sum(result.passed for result in results)
    unique_hashes = set(input_hashes.values())
    hashes_by_scenario: dict[str, set[str]] = {
        "multi_column": set(),
        "cross_page_table": set(),
        "scanned": set(),
    }
    for case in dataset.cases:
        if document_hash := input_hashes.get(case.document):
            hashes_by_scenario[case.scenario].add(document_hash)
    unique_per_scenario = {
        scenario: len(document_hashes)
        for scenario, document_hashes in hashes_by_scenario.items()
    }
    real_documents = bool(dataset.cases) and all(
        case.real_document and not case.synthetic for case in dataset.cases
    )
    authorization_verified = bool(dataset.authorization_reference)
    formal = bool(
        len(unique_hashes) >= 30
        and all(unique_per_scenario[scenario] >= 10 for scenario in unique_per_scenario)
        and real_documents
        and authorization_verified
        and dataset.independent_real_documents
    )
    rates_pass = bool(
        results
        and passed_cases / len(results) >= 0.95
        and all(
            metric.success_rate is not None and metric.success_rate >= 0.95
            for metric in scenario_metrics
        )
    )
    canonical = {
        "dataset": dataset.model_dump(mode="json"),
        "document_sha256": dict(sorted(input_hashes.items())),
    }
    limitations: list[str] = []
    if len(unique_hashes) < 30:
        limitations.append(f"只有 {len(unique_hashes)} 份去重文档；正式验收要求至少 30 份。")
    for scenario, count in unique_per_scenario.items():
        if count < 10:
            limitations.append(f"{scenario} 只有 {count} 份去重文档；要求至少 10 份。")
    if not real_documents:
        limitations.append("含合成或未声明为真实的文档，只能作为烟雾测试。")
    if not authorization_verified:
        limitations.append("缺少公开来源或授权引用。")
    if not dataset.independent_real_documents:
        limitations.append("未声明为独立真实文档集。")
    return PDFRegressionReport(
        dataset_version=dataset.dataset_version,
        source_sha256=hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        sample_size=len(unique_hashes),
        case_count=len(results),
        real_documents=real_documents,
        authorization_verified=authorization_verified,
        independent_real_documents=dataset.independent_real_documents,
        scenario_metrics=scenario_metrics,
        overall_success_rate=round(passed_cases / len(results), 4) if results else None,
        overall_ci95=wilson_interval(passed_cases, len(results)),
        formal_gate_eligible=formal,
        passed=formal and rates_pass,
        smoke_test_only=not formal,
        results=results,
        limitations=limitations,
    )


__all__ = [
    "PDFCaseResult",
    "PDFRegressionCase",
    "PDFRegressionDataset",
    "PDFRegressionReport",
    "PDFScenarioMetric",
    "PDFScenario",
    "evaluate_pdf_regression",
    "load_pdf_regression_dataset",
]
