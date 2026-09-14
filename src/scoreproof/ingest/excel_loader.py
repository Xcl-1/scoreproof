"""Excel 接入：综测表（结构化申报数据）与规则表。

**易翻车点 1（项目总结第 6 节）**：Excel 合并单元格必须 fill-down，
否则整表"类别"列会错位，导致所有条目挂到错误的组上。
本模块把"读文件"和"处理合并单元格"分开暴露，便于单元测试。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DataSourceError
from ..normalize import level_aliases, normalize_academic_year, normalize_category, normalize_level
from ..schema import Claim, ConstraintSpec, Rule, SourceRef

# ======================================================================
# 通用工具
# ======================================================================


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip() in ("", "nan", "NaN", "None")


def normalize_header(raw: Any) -> str:
    """表头清洗：去空白、换行、括号注释，保留中文语义。"""
    if raw is None:
        return ""
    t = str(raw)
    t = re.sub(r"[\s\u3000]+", "", t)
    t = t.replace("\n", "")
    t = re.sub(r"[（(].*?[)）]$", "", t)
    return t.strip()


def find_overlapping_merges(
    merged_ranges: Iterable[tuple[int, int, int, int]],
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """找出互相重叠的合并区域。

    重叠的合并区域是**数据损坏的强信号**（真实 Excel 里 openpyxl 会直接报错，
    但从其他工具/手工拼出来的表可能带着重叠范围），填充结果会互相覆盖，
    导致整列类别错位 —— 必须显式告警而不是静默算错。
    """
    ranges = list(merged_ranges)
    clashes: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i, a in enumerate(ranges):
        for b in ranges[i + 1 :]:
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                clashes.append((a, b))
    return clashes


def fill_merged_cells(
    df: pd.DataFrame,
    merged_ranges: Iterable[tuple[int, int, int, int]],
    *,
    fill_down: bool = True,
    fill_right: bool = True,
) -> pd.DataFrame:
    """按合并区域填充空值。

    Args:
        df: 原始网格（``header=None`` 读入）。
        merged_ranges: ``(min_row, min_col, max_row, max_col)``，1-based、闭区间（openpyxl 语义）。
        fill_down: 纵向填充（同一类别跨多行时必需）。
        fill_right: 横向填充（表头跨多列时必需）。

    Raises:
        DataSourceError: 合并区域互相重叠（数据可能已损坏，拒绝猜测）。

    Note:
        只填充**合并区域内部**，不做全表 ffill —— 全表 ffill 会把某一列的值
        污染到整行（表头行尤其致命），这是 Excel 解析最常见的翻车来源。
        普通空单元格（未合并）一律保持为空，由上层决定怎么处理。
    """
    ranges = list(merged_ranges)
    clashes = find_overlapping_merges(ranges)
    if clashes:
        raise DataSourceError(
            f"检测到 {len(clashes)} 处重叠的合并单元格，表格结构可能已损坏，拒绝继续解析",
            detail={"overlapping": [list(c) for c in clashes[:10]]},
        )
    out = df.copy()
    for min_row, min_col, max_row, max_col in ranges:
        r0, r1 = min_row - 1, max_row - 1
        c0, c1 = min_col - 1, max_col - 1
        if r0 < 0 or c0 < 0 or r0 >= len(out) or c0 >= out.shape[1]:
            continue
        r1 = min(r1, len(out) - 1)
        c1 = min(c1, out.shape[1] - 1)
        block = out.iloc[r0 : r1 + 1, c0 : c1 + 1]
        value = None
        for cell in block.to_numpy().ravel():
            if not _is_blank(cell):
                value = cell
                break
        if value is None:
            continue
        # 先横后纵：保证表头式"L 形"合并区域彻底填满
        target = out.iloc[r0 : r1 + 1, c0 : c1 + 1]
        if fill_right:
            target = target.ffill(axis=1)
        if fill_down:
            target = target.ffill(axis=0)
        if not fill_right and not fill_down:
            target = pd.DataFrame(value, index=target.index, columns=target.columns)
        out.iloc[r0 : r1 + 1, c0 : c1 + 1] = target
    return out


def fill_down_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """对单列做 fill-down（类别列常见于"只写一次，下面留空"的表）。"""
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = out[column].ffill()
    return out


def detect_header_row(df: pd.DataFrame, *, max_scan: int = 10, min_fill: float = 0.5) -> int:
    """猜测表头行：取值最多、且文本占比最高的一行。"""
    best_idx, best_score = 0, -1.0
    for i in range(min(max_scan, len(df))):
        row = df.iloc[i]
        non_null = row.notna().sum()
        if non_null == 0:
            continue
        fill = non_null / max(1, df.shape[1])
        if fill < min_fill:
            continue
        text_ratio = sum(1 for v in row.dropna() if isinstance(v, str) and not _num(v)) / non_null
        score = fill + text_ratio
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _num(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def to_number(v: Any, *, default: float | None = None) -> float | None:
    """稳健解析分值：支持 ``8`` / ``"8分"`` / ``"8.0"`` / 空白。"""
    if _is_blank(v):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else default


@dataclass
class ExcelGrid:
    """一个已清洗的工作表 + 出处信息。"""

    df: pd.DataFrame
    path: Path
    sheet: str
    header_row: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def source_doc(self) -> str:
        return self.path.name

    def source_ref(self, row: int, *, table: str | None = None, text: str | None = None) -> SourceRef:
        """行号 -> 出处（Excel 用行号定位，注意 1-based 且含表头偏移）。"""
        return SourceRef(
            doc=self.source_doc,
            table=table or self.sheet,
            row=int(row) + self.header_row + 2,
            text=text,
        )

    def column(self, *candidates: str) -> str | None:
        """按候选表头名找列。

        三级匹配，逐级放宽（宁可稳一点也不要错列）：
        1. 完全相同；
        2. 互为子串（``分值`` vs ``加分数值``）；
        3. 字符集合重叠度 >= 0.6（``加分分值`` vs ``分值``），仍取最高分。
        """
        cols = list(self.df.columns)
        normalized = {c: normalize_header(c) for c in cols}
        for cand in candidates:
            cand_n = normalize_header(cand)
            if not cand_n:
                continue
            for c, col_n in normalized.items():
                if cand_n == col_n:
                    return c
            for c, col_n in normalized.items():
                if col_n and (cand_n in col_n or col_n in cand_n):
                    return c
            best, best_score = None, 0.0
            for c, col_n in normalized.items():
                if not col_n:
                    continue
                overlap = len(set(cand_n) & set(col_n))
                score = overlap / min(len(cand_n), len(col_n))
                if score > best_score:
                    best, best_score = c, score
            if best is not None and best_score >= 0.6:
                return best
        return None

    def __len__(self) -> int:
        return len(self.df)


# ======================================================================
# 读文件
# ======================================================================


def load_workbook_grid(path: str | Path) -> dict[str, tuple[pd.DataFrame, list[tuple[int, int, int, int]]]]:
    """读整本 Excel，返回 ``sheet -> (原始网格, 合并区域)``。"""
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"Excel 文件不存在：{p}", detail={"path": str(p)})
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError("需要 openpyxl 才能读取 .xlsx", detail={"hint": "uv sync"}) from exc

    wb = openpyxl.load_workbook(p, data_only=True)
    out: dict[str, tuple[pd.DataFrame, list[tuple[int, int, int, int]]]] = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        ranges = [
            (rng.min_row, rng.min_col, rng.max_row, rng.max_col)
            for rng in ws.merged_cells.ranges
        ]
        out[ws.title] = (df, ranges)
    wb.close()
    return out


def load_sheet(
    path: str | Path,
    sheet: str | int = 0,
    *,
    header_row: int | None = None,
    fill_down: bool = True,
    fill_right: bool = True,
    drop_empty_rows: bool = True,
) -> ExcelGrid:
    """读单个工作表：填充合并单元格 -> 定位表头 -> 返回 DataFrame。"""
    grids = load_workbook_grid(path)
    if not grids:
        raise DataSourceError(f"Excel 中没有工作表：{path}")
    if isinstance(sheet, int):
        name = list(grids)[sheet]
    else:
        if sheet not in grids:
            raise DataSourceError(
                f"工作表 {sheet!r} 不存在", detail={"sheets": list(grids)}
            )
        name = sheet

    raw, ranges = grids[name]
    df = fill_merged_cells(raw, ranges, fill_down=fill_down, fill_right=fill_right)
    if df.empty:
        return ExcelGrid(df=df, path=Path(path), sheet=name, header_row=0)

    hrow = detect_header_row(df) if header_row is None else header_row
    header = [normalize_header(v) if not _is_blank(v) else f"col_{i}" for i, v in enumerate(df.iloc[hrow])]
    body = df.iloc[hrow + 1 :].reset_index(drop=True)
    body.columns = _dedupe_columns(header)
    if drop_empty_rows:
        body = body.dropna(how="all").reset_index(drop=True)
    return ExcelGrid(df=body, path=Path(path), sheet=name, header_row=hrow)


def _dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        base = c or "col"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out


def read_excel(path: str | Path, sheet: str | int = 0, **kwargs) -> pd.DataFrame:
    """最常用入口：直接要清洗后的 DataFrame。"""
    return load_sheet(path, sheet, **kwargs).df


# ======================================================================
# 规则表 -> Rule
# ======================================================================

_RULE_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "category": ("类别", "项目类别", "加分项目", "项目", "分类"),
    "level": ("等级", "级别", "名次", "获奖等级", "加分等级", "获奖级别"),
    "score": ("分值", "加分", "分数", "加分数值", "得分"),
    "cap": ("封顶", "上限", "最高分", "单项上限", "限额"),
    "dedup_group": ("互斥组", "去重组", "同类组", "分组"),
    "team_factor": ("团队系数", "团队折算", "折算系数"),
    "synonyms": ("同义词", "别名", "其他表述"),
    "clause": ("条款", "依据", "页码", "备注"),
}


def load_rules(
    path: str | Path,
    *,
    sheet: str | int = 0,
    academic_year: str,
    college: str | None = None,
    priority: int = 0,
    header_row: int | None = None,
    extra_level_aliases: bool = True,
) -> list[Rule]:
    """把"等级×类别 加分矩阵表"读成 Rule 列表。

    第一版策略（项目总结第 6 节）：**程序抽取 + 人工校对一遍**，
    所以这里只做结构化映射，不做语义猜测；无法确定分值的行会被跳过。
    """
    grid = load_sheet(path, sheet, header_row=header_row)
    df = grid.df
    if df.empty:
        raise DataSourceError(f"工作表 {grid.sheet!r} 没有数据行", detail={"path": str(path)})

    col = {key: grid.column(*cands) for key, cands in _RULE_COLUMN_CANDIDATES.items()}
    missing = [k for k in ("category", "level", "score") if not col[k]]
    if missing:
        raise DataSourceError(
            f"规则表缺少必要列：{missing}",
            detail={"found_columns": list(df.columns), "path": str(path)},
        )

    year = normalize_academic_year(academic_year)
    rules: list[Rule] = []
    for idx, row in df.iterrows():
        level_raw = row.get(col["level"])
        category_raw = row.get(col["category"])
        score = to_number(row.get(col["score"]))
        if _is_blank(level_raw) or _is_blank(category_raw) or score is None:
            continue

        res = normalize_level(str(level_raw))
        category = normalize_category(str(category_raw))
        synonyms = [str(level_raw).strip()]
        if col["synonyms"] and not _is_blank(row.get(col["synonyms"])):
            synonyms += [s.strip() for s in re.split(r"[/、,，;；|]", str(row.get(col["synonyms"]))) if s.strip()]
        if extra_level_aliases:
            synonyms += level_aliases(res.canonical)

        cap = to_number(row.get(col["cap"])) if col["cap"] else None
        team_factor = to_number(row.get(col["team_factor"]), default=1.0) if col["team_factor"] else 1.0
        if team_factor is not None and team_factor > 1:  # 表里常写成百分数或"0.5"
            team_factor = team_factor / 100 if team_factor >= 10 else team_factor

        dedup_group = (
            str(row.get(col["dedup_group"])).strip()
            if col["dedup_group"] and not _is_blank(row.get(col["dedup_group"]))
            else category
        )

        rules.append(
            Rule(
                academic_year=year,
                college=college,
                category=category,
                level=res.canonical,
                score=score,
                synonyms=sorted({s for s in synonyms if s and s != res.canonical}),
                constraints=ConstraintSpec(
                    dedup_group=dedup_group,
                    cap=cap,
                    team_factor=team_factor if team_factor else 1.0,
                    note=str(row.get(col["clause"])).strip()
                    if col["clause"] and not _is_blank(row.get(col["clause"]))
                    else None,
                ),
                source=grid.source_ref(idx, table=grid.sheet, text=str(level_raw).strip()),
                priority=priority,
                raw_text=str(level_raw).strip(),
            )
        )
    return rules


# ======================================================================
# 综测表 -> Claim
# ======================================================================

_CLAIM_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "item_key": ("申报项标识", "申报项ID", "申报项编号"),
    "student_id": ("学号", "学生学号", "编号", "工号"),
    "student_name": ("姓名", "学生姓名", "名字"),
    "college": ("学院", "院系", "所在学院"),
    "category": ("类别", "项目类别", "加分项目", "项目分类"),
    "raw_text": ("申报内容", "申报条目", "加分事由", "事由", "内容", "项目名称", "获奖名称"),
    "level": ("等级", "级别", "名次", "获奖等级"),
    "team": ("团队", "是否团队", "集体"),
    "reason": ("备注", "说明", "备注说明"),
}


def load_claims(
    path: str | Path,
    *,
    sheet: str | int = 0,
    academic_year: str | None = None,
    header_row: int | None = None,
) -> list[Claim]:
    """把综测表读成 Claim 列表。

    注意：Excel **不进向量库**（结构化数据），这里只做字段映射与归一化。
    """
    grid = load_sheet(path, sheet, header_row=header_row)
    df = grid.df
    if df.empty:
        raise DataSourceError(f"工作表 {grid.sheet!r} 没有数据行", detail={"path": str(path)})

    col = {key: grid.column(*cands) for key, cands in _CLAIM_COLUMN_CANDIDATES.items()}
    if not col["student_id"] and not col["student_name"]:
        raise DataSourceError(
            "综测表必须至少包含学号或姓名列",
            detail={"found_columns": list(df.columns)},
        )

    year = normalize_academic_year(academic_year) if academic_year else None
    claims: list[Claim] = []
    for idx, row in df.iterrows():
        sid = str(row.get(col["student_id"])).strip() if col["student_id"] and not _is_blank(row.get(col["student_id"])) else ""
        name = str(row.get(col["student_name"])).strip() if col["student_name"] and not _is_blank(row.get(col["student_name"])) else None
        raw_text = (
            str(row.get(col["raw_text"])).strip()
            if col["raw_text"] and not _is_blank(row.get(col["raw_text"]))
            else ""
        )
        level_raw = (
            str(row.get(col["level"])).strip()
            if col["level"] and not _is_blank(row.get(col["level"]))
            else raw_text
        )
        if _is_blank(sid) and not name and not raw_text:
            continue
        if not sid:
            sid = f"{name or '未知'}#{idx}"

        level_res = normalize_level(level_raw) if level_raw else None
        source_ref = grid.source_ref(idx, table=grid.sheet, text=raw_text or level_raw)
        stable_id_seed = f"{grid.sheet}|{source_ref.row}|{sid}"
        stable_claim_id = f"c_{hashlib.sha256(stable_id_seed.encode('utf-8')).hexdigest()[:12]}"
        claims.append(
            Claim(
                id=stable_claim_id,
                student_id=sid,
                student_name=name,
                academic_year=year,
                college=str(row.get(col["college"])).strip()
                if col["college"] and not _is_blank(row.get(col["college"]))
                else None,
                category=normalize_category(str(row.get(col["category"])))
                if col["category"] and not _is_blank(row.get(col["category"]))
                else "未分类",
                raw_text=raw_text,
                level=level_res.canonical if level_res and level_res.matched else None,
                team=_as_bool(row.get(col["team"])) if col["team"] else False,
                source_ref=source_ref,
                extra={
                    "level_matched": bool(level_res and level_res.matched),
                    **(
                        {"item_key": str(row.get(col["item_key"])).strip()}
                        if col["item_key"] and not _is_blank(row.get(col["item_key"]))
                        else {}
                    ),
                }
                if level_res or col["item_key"]
                else {},
            )
        )
    return claims


def _as_bool(v: Any) -> bool:
    if _is_blank(v):
        return False
    s = str(v).strip().lower()
    return s in ("是", "y", "yes", "true", "1", "团队", "集体", "√")


def group_claims_by_student(claims: Iterable[Claim]) -> dict[str, list[Claim]]:
    acc: dict[str, list[Claim]] = {}
    for c in claims:
        acc.setdefault(c.student_id, []).append(c)
    return acc


__all__ = [
    "ExcelGrid",
    "detect_header_row",
    "fill_down_column",
    "fill_merged_cells",
    "find_overlapping_merges",
    "group_claims_by_student",
    "load_claims",
    "load_rules",
    "load_sheet",
    "load_workbook_grid",
    "normalize_header",
    "read_excel",
    "to_number",
]
