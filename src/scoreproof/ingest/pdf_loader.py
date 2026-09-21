"""PDF 接入：细则文本 + 表格抽取，保留页码与坐标用于溯源。

选型：PyMuPDF 抽文本（快、坐标精确），表格复杂时回落到 pdfplumber。
两者都是可选依赖，缺失时给出可执行的提示而不是 ImportError 崩溃。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DataSourceError

try:  # pragma: no cover - 取决于环境
    import pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None


@dataclass
class PDFPage:
    """一页的抽取结果，``source`` 可直接挂到 Rule.source 上。"""

    page: int  # 1-based
    text: str = ""
    raw_text: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def is_probably_scanned(self) -> bool:
        """几乎没文字 => 大概率是扫描件，需走 OCR + 人工校对。"""
        return self.char_count < 20

    def find(self, *keywords: str) -> list[dict[str, Any]]:
        """按关键词定位段落，返回带坐标的块（用于条款级引用）。"""
        hits: list[dict[str, Any]] = []
        for b in self.blocks:
            text = b.get("text", "")
            if any(k in text for k in keywords):
                hits.append(b)
        return hits


@dataclass
class PDFTableFragment:
    """pdfplumber 检出的单页表格片段，保留页码、序号与区域。"""

    page: int
    table_index: int
    cells: list[list[str]]
    bbox: tuple[float, float, float, float] | None = None
    page_width: float | None = None
    page_height: float | None = None

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.cells), default=0)


@dataclass
class PDFLogicalTable:
    """一个可能由多个相邻页面片段拼接而成的逻辑表。"""

    id: str
    start_page: int
    end_page: int
    rows: list[list[str]] = field(default_factory=list)
    fragments: list[PDFTableFragment] = field(default_factory=list)
    repeated_headers_removed: int = 0
    continuation_reasons: list[str] = field(default_factory=list)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    @property
    def spans_pages(self) -> bool:
        return self.end_page > self.start_page


@dataclass
class PDFDocument:
    """PDF 解析结果：原文、版面顺序、逻辑表与显式复核问题。"""

    path: Path
    sha256: str
    pages: list[PDFPage]
    logical_tables: list[PDFLogicalTable] = field(default_factory=list)
    review_issues: list[str] = field(default_factory=list)

    @property
    def scanned_pages(self) -> list[int]:
        return [page.page for page in self.pages if page.is_probably_scanned]

    @property
    def multi_column_pages(self) -> list[int]:
        return [page.page for page in self.pages if page.meta.get("layout") == "two_column"]

    @property
    def document_kind(self) -> str:
        if not self.scanned_pages:
            return "text"
        if len(self.scanned_pages) == len(self.pages):
            return "scanned"
        return "mixed"


def _vertical_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    ly0, ly1 = float(left["bbox"][1]), float(left["bbox"][3])
    ry0, ry1 = float(right["bbox"][1]), float(right["bbox"][3])
    return max(0.0, min(ly1, ry1) - max(ly0, ry0))


def reorder_text_blocks(
    blocks: list[dict[str, Any]], page_width: float
) -> tuple[list[dict[str, Any]], bool]:
    """保守识别双栏正文，并按“左栏后右栏”重排；表格窄单元格不会触发。"""

    if page_width <= 0 or len(blocks) < 4:
        stable_order = sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return stable_order, False
    middle = page_width / 2
    margin = page_width * 0.06
    min_prose_width = page_width * 0.20
    max_column_width = page_width * 0.58
    left = [
        block
        for block in blocks
        if min_prose_width <= block["bbox"][2] - block["bbox"][0] <= max_column_width
        and block["bbox"][2] <= middle + margin
    ]
    right = [
        block
        for block in blocks
        if min_prose_width <= block["bbox"][2] - block["bbox"][0] <= max_column_width
        and block["bbox"][0] >= middle - margin
    ]
    has_parallel_rows = any(_vertical_overlap(a, b) > 2 for a in left for b in right)
    if len(left) < 2 or len(right) < 2 or not has_parallel_rows:
        stable_order = sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return stable_order, False

    left_ids = {id(block) for block in left}
    right_ids = {id(block) for block in right}
    spanning = [block for block in blocks if id(block) not in left_ids | right_ids]
    spanning.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    boundaries = [-float("inf"), *(float(item["bbox"][1]) for item in spanning), float("inf")]
    reordered: list[dict[str, Any]] = []
    for index in range(len(boundaries) - 1):
        lo, hi = boundaries[index], boundaries[index + 1]
        section_left = [
            item for item in left if lo <= (item["bbox"][1] + item["bbox"][3]) / 2 < hi
        ]
        section_right = [
            item for item in right if lo <= (item["bbox"][1] + item["bbox"][3]) / 2 < hi
        ]
        reordered.extend(sorted(section_left, key=lambda item: (item["bbox"][1], item["bbox"][0])))
        reordered.extend(sorted(section_right, key=lambda item: (item["bbox"][1], item["bbox"][0])))
        if index < len(spanning):
            reordered.append(spanning[index])
    if len(reordered) != len(blocks):  # 防御：任何坐标异常都退回稳定顺序
        return sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])), False
    return reordered, True


def _as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _rounded_bbox(value: Any) -> tuple[float, float, float, float] | None:
    bbox = _as_bbox(value)
    if bbox is None:
        return None
    return (round(bbox[0], 2), round(bbox[1], 2), round(bbox[2], 2), round(bbox[3], 2))


def _require_pymupdf() -> Any:
    if pymupdf is None:  # pragma: no cover
        raise DataSourceError(
            "需要 pymupdf 才能读取 PDF",
            detail={"hint": "uv sync --extra multimodal 或 uv add pymupdf"},
        )
    return pymupdf


def load_pdf(
    path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
    with_tables: bool = False,
) -> list[PDFPage]:
    """读取 PDF 每页的文本与文本块坐标。

    Args:
        start: 起始页（1-based，含）。
        end: 结束页（1-based，含）。
        with_tables: 是否顺带用 pdfplumber 抽表格。
    """
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"PDF 文件不存在：{p}", detail={"path": str(p)})
    fitz = _require_pymupdf()

    pages: list[PDFPage] = []
    with fitz.open(p) as doc:
        total = doc.page_count
        lo = max(1, start or 1)
        hi = min(total, end or total)
        for i in range(lo - 1, hi):
            page = doc.load_page(i)
            raw_text = page.get_text("text")
            blocks = [
                {
                    "text": b[4].strip(),
                    "bbox": tuple(round(v, 2) for v in b[:4]),
                    "block_no": b[5] if len(b) > 5 else None,
                }
                for b in page.get_text("blocks")
                if str(b[4]).strip()
            ]
            ordered_blocks, is_multi_column = reorder_text_blocks(blocks, float(page.rect.width))
            for reading_order, block in enumerate(ordered_blocks):
                block["reading_order"] = reading_order
            text = (
                "\n".join(str(block["text"]).strip() for block in ordered_blocks)
                if is_multi_column
                else raw_text
            )
            pages.append(
                PDFPage(
                    page=i + 1,
                    text=text,
                    raw_text=raw_text,
                    blocks=ordered_blocks,
                    meta={
                        "doc": p.name,
                        "total_pages": total,
                        "page_width": round(float(page.rect.width), 2),
                        "page_height": round(float(page.rect.height), 2),
                        "layout": "two_column" if is_multi_column else "single_column",
                        "raw_order_changed": is_multi_column,
                    },
                )
            )
    if with_tables:
        fragments = load_pdf_table_fragments(p, start=start, end=end)
        for pg in pages:
            page_fragments = fragments.get(pg.page, [])
            pg.tables = [fragment.cells for fragment in page_fragments]
            pg.meta["table_regions"] = [
                {
                    "table_index": fragment.table_index,
                    "bbox": fragment.bbox,
                }
                for fragment in page_fragments
            ]
    return pages


def _clean_table(raw: list[list[Any]]) -> list[list[str]]:
    return [["" if cell is None else str(cell).strip() for cell in row] for row in raw]


def load_pdf_table_fragments(
    path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
) -> dict[int, list[PDFTableFragment]]:
    """抽取带 bbox 的单页表格片段，供跨页逻辑拼接与人工复核。"""

    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"PDF 文件不存在：{p}", detail={"path": str(p)})
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError(
            "抽表格需要 pdfplumber",
            detail={"hint": "uv add pdfplumber 或 uv sync --extra tables"},
        ) from exc

    out: dict[int, list[PDFTableFragment]] = {}
    with pdfplumber.open(p) as pdf:
        lo = max(1, start or 1)
        hi = min(len(pdf.pages), end or len(pdf.pages))
        for i in range(lo - 1, hi):
            page = pdf.pages[i]
            found = page.find_tables()
            fragments: list[PDFTableFragment] = []
            for table_index, table in enumerate(found):
                raw = table.extract() or []
                if not raw:
                    continue
                fragments.append(
                    PDFTableFragment(
                        page=i + 1,
                        table_index=table_index,
                        cells=_clean_table(raw),
                        bbox=_rounded_bbox(table.bbox),
                        page_width=float(page.width),
                        page_height=float(page.height),
                    )
                )
            if fragments:
                out[i + 1] = fragments
    return out


def load_pdf_tables(
    path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
) -> dict[int, list[list[list[str]]]]:
    """用 pdfplumber 抽表格：``{页码: [表1, 表2, ...]}``。

    细则里的"等级×类别 加分矩阵"就在这里；第一版允许"抽取 + 人工校对一遍"。
    """
    fragments = load_pdf_table_fragments(path, start=start, end=end)
    return {
        page: [fragment.cells for fragment in page_fragments]
        for page, page_fragments in fragments.items()
    }


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("|：:，,；;")


def _first_meaningful_row(rows: list[list[str]]) -> tuple[int, tuple[str, ...]] | None:
    for index, row in enumerate(rows):
        normalized = tuple(_compact(cell) for cell in row)
        if sum(bool(cell) for cell in normalized) >= 2:
            return index, normalized
    return None


def _header_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 0.0
    padded_left = (*left, *("" for _ in range(width - len(left))))
    padded_right = (*right, *("" for _ in range(width - len(right))))
    matches = sum(
        1
        for a, b in zip(padded_left, padded_right, strict=True)
        if a and b and (a == b or a in b or b in a)
    )
    return matches / width


def _near_page_edge(fragment: PDFTableFragment, *, bottom: bool) -> bool:
    if fragment.bbox is None or not fragment.page_height:
        return False
    coordinate = fragment.bbox[3] if bottom else fragment.bbox[1]
    ratio = coordinate / fragment.page_height
    return ratio >= 0.80 if bottom else ratio <= 0.22


def stitch_cross_page_tables(
    pages: list[PDFPage], fragments: dict[int, list[PDFTableFragment]]
) -> tuple[list[PDFLogicalTable], list[str]]:
    """按相邻页边缘、列结构、重复表头/续表词拼接逻辑表。"""

    logical: list[PDFLogicalTable] = []
    issues: list[str] = []
    page_by_number = {page.page: page for page in pages}
    for page_number in sorted(fragments):
        current_fragments = fragments[page_number]
        for position, fragment in enumerate(current_fragments):
            previous = logical[-1] if logical else None
            can_compare = bool(
                previous
                and previous.end_page == page_number - 1
                and position == 0
                and previous.fragments[-1].table_index
                == len(fragments.get(page_number - 1, [])) - 1
                and previous.column_count == fragment.column_count
                and fragment.column_count > 1
            )
            reasons: list[str] = []
            repeated_index: int | None = None
            if can_compare and previous is not None:
                previous_header = _first_meaningful_row(previous.rows)
                current_header = _first_meaningful_row(fragment.cells)
                if previous_header and current_header:
                    similarity = _header_similarity(previous_header[1], current_header[1])
                    if similarity >= 0.6:
                        reasons.append("repeated_header")
                        repeated_index = current_header[0]
                if _near_page_edge(previous.fragments[-1], bottom=True) and _near_page_edge(
                    fragment, bottom=False
                ):
                    reasons.append("page_edge_continuation")
                adjoining_text = "\n".join(
                    page_by_number[number].text
                    for number in (page_number - 1, page_number)
                    if number in page_by_number
                )
                if re.search(r"(?:续表|接上表|续上表)", adjoining_text):
                    reasons.append("continuation_marker")
            if previous is not None and reasons:
                rows = list(fragment.cells)
                if repeated_index is not None:
                    rows.pop(repeated_index)
                    previous.repeated_headers_removed += 1
                previous.rows.extend(rows)
                previous.fragments.append(fragment)
                previous.end_page = page_number
                previous.continuation_reasons.extend(reasons)
                continue
            logical.append(
                PDFLogicalTable(
                    id=f"table-{len(logical) + 1}",
                    start_page=page_number,
                    end_page=page_number,
                    rows=list(fragment.cells),
                    fragments=[fragment],
                )
            )
            if can_compare and not reasons:
                issues.append(
                    f"第 {page_number - 1}～{page_number} 页存在同列数相邻表格，"
                    "但没有足够续表证据，已保持分离并要求人工复核。"
                )
    return logical, issues


def load_pdf_document(path: str | Path) -> PDFDocument:
    """完整加载 PDF，并输出复杂版面元数据、跨页逻辑表和显式复核项。"""

    p = Path(path)
    pages = load_pdf(p, with_tables=True)
    fragments: dict[int, list[PDFTableFragment]] = {}
    for page in pages:
        regions = page.meta.get("table_regions", [])
        page_fragments = [
            PDFTableFragment(
                page=page.page,
                table_index=index,
                cells=table,
                bbox=_as_bbox(regions[index].get("bbox")) if index < len(regions) else None,
                page_width=float(page.meta["page_width"]),
                page_height=float(page.meta["page_height"]),
            )
            for index, table in enumerate(page.tables)
        ]
        if page_fragments:
            fragments[page.page] = page_fragments
    logical_tables, issues = stitch_cross_page_tables(pages, fragments)
    scanned_pages = [page.page for page in pages if page.is_probably_scanned]
    if scanned_pages:
        issues.append(f"疑似扫描/低文本页 {scanned_pages}：需要 OCR 或人工复核。")
    return PDFDocument(
        path=p,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        pages=pages,
        logical_tables=logical_tables,
        review_issues=issues,
    )


def pdf_page_to_source(pg: PDFPage, *, table: str | None = None, clause: str | None = None,
                       text: str | None = None, bbox: tuple[float, float, float, float] | None = None):
    """PDFPage -> SourceRef（延迟导入，避免循环依赖）。"""
    from ..schema import SourceRef

    return SourceRef(doc=pg.meta.get("doc", ""), page=pg.page, table=table, clause=clause,
                     text=text, bbox=bbox)


__all__ = [
    "PDFDocument",
    "PDFLogicalTable",
    "PDFPage",
    "PDFTableFragment",
    "load_pdf",
    "load_pdf_document",
    "load_pdf_table_fragments",
    "load_pdf_tables",
    "pdf_page_to_source",
    "reorder_text_blocks",
    "stitch_cross_page_tables",
]
