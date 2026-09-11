"""测试公共 fixture：构造规则、申报、Excel 样例。

所有测试都用**合成数据**，绝不引用真实同学材料（合规红线）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from scoreproof.normalize import level_aliases
from scoreproof.schema import Claim, ConstraintSpec, Rule, Ruleset, SourceRef

# 工作区内的临时目录根：不使用系统临时目录（沙箱下不可写）
_TMP_ROOT = Path(__file__).resolve().parent.parent / ".tmp"
_RUN_ID = f"run-{int(time.time())}-{os.getpid()}"
_CASE_COUNTER = [1]  # 用 list 保存可变计数，保证同一次会话内目录不重名


def pytest_configure(config: pytest.Config) -> None:
    """避免使用 pytest 自带的临时目录工厂。

    受限环境（沙箱 / 只读系统临时目录）下，pytest 会以 ``mode=0o700`` 创建
    临时目录，随后自己的清理钩子又去 ``iterdir()`` 它 —— 在 DSH 沙箱里这会被
    直接拒绝（WinError 5）。这里把 basetemp 指到工作区内的普通目录，并关闭清理；
    真正的临时目录由下面的 ``tmp_path`` fixture 提供（普通权限、位于工作区内）。
    """
    config.option.basetemp = str(_TMP_ROOT / _RUN_ID)
    config.option.keep_temp_dir = True
    config.option.keep_tempfiles = True
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@pytest.fixture
def tmp_path() -> Path:
    """替代 pytest 内置 ``tmp_path``：工作区内、普通权限、每次唯一。"""
    current = _CASE_COUNTER[0]
    _CASE_COUNTER[0] += 1
    path = _TMP_ROOT / _RUN_ID / f"case-{current}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_rule(
    level: str,
    score: float,
    *,
    category: str = "学科竞赛",
    group: str | None = None,
    cap: float | None = None,
    team_factor: float = 1.0,
    synonyms: list[str] | None = None,
    academic_year: str = "2025-2026",
    college: str | None = None,
    priority: int = 0,
    require_catalog: str | None = None,
    exclusive_with: list[str] | None = None,
    rule_id: str | None = None,
) -> Rule:
    kwargs = {}
    if rule_id:
        kwargs["id"] = rule_id
    return Rule(
        academic_year=academic_year,
        college=college,
        category=category,
        level=level,
        score=score,
        synonyms=list(synonyms or []),
        constraints=ConstraintSpec(
            dedup_group=group or category,
            cap=cap,
            team_factor=team_factor,
            require_catalog=require_catalog,
            exclusive_with=list(exclusive_with or []),
        ),
        source=SourceRef(doc="合成细则.pdf", page=4, table="加分标准表", clause="第三章第7条",
                         text=f"{level} {score:g}分"),
        priority=priority,
        raw_text=f"{level} {score:g}分",
        **kwargs,
    )


def make_claim(
    raw_text: str = "省二等奖",
    *,
    level: str | None = None,
    category: str = "学科竞赛",
    student_id: str = "2023001",
    academic_year: str = "2025-2026",
    college: str | None = None,
    team: bool = False,
    catalog_listed: bool = True,
    claim_id: str | None = None,
) -> Claim:
    kwargs = {}
    if claim_id:
        kwargs["id"] = claim_id
    return Claim(
        student_id=student_id,
        academic_year=academic_year,
        college=college,
        category=category,
        raw_text=raw_text,
        level=level,
        team=team,
        catalog_listed=catalog_listed,
        **kwargs,
    )


@pytest.fixture
def base_ruleset() -> Ruleset:
    """一套典型的学科竞赛规则（含同义词、封顶、团队折算）。"""
    return Ruleset(
        rules=[
            make_rule("国家级一等奖", 15, group="学科竞赛", cap=30,
                      synonyms=level_aliases("国家级一等奖")),
            make_rule("国家级二等奖", 12, group="学科竞赛", cap=30,
                      synonyms=level_aliases("国家级二等奖")),
            make_rule("省级一等奖", 10, group="学科竞赛", cap=20,
                      synonyms=level_aliases("省级一等奖")),
            make_rule("省级二等奖", 8, group="学科竞赛", cap=20, team_factor=0.5,
                      synonyms=level_aliases("省级二等奖")),
            make_rule("省级三等奖", 5, group="学科竞赛", cap=20,
                      synonyms=level_aliases("省级三等奖")),
            make_rule("校级一等奖", 3, group="学科竞赛", cap=10,
                      synonyms=level_aliases("校级一等奖")),
            make_rule("一等奖", 6, category="文体活动", group="文体活动", cap=10),
            make_rule("二等奖", 4, category="文体活动", group="文体活动", cap=10),
            make_rule("国家奖学金", 10, category="荣誉称号", group="荣誉称号", cap=10),
            make_rule("省级三好学生", 5, category="荣誉称号", group="荣誉称号", cap=10),
        ]
    )


@pytest.fixture
def rules_excel(tmp_path: Path) -> Path:
    """合成规则表：含**合并单元格**（类别跨行），用于验证 fill-down。"""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "加分标准表"
    ws.append(["类别", "等级", "分值", "封顶"])
    ws.append(["学科竞赛", "国家级一等奖", 15, 30])
    ws.append([None, "省级一等奖", 10, 20])  # 依赖合并/留空 -> 类别应保持"学科竞赛"
    ws.merge_cells("A2:A3")
    ws.append(["文体活动", "一等奖", 6, 10])
    ws.append([None, "二等奖", 4, 10])
    ws.merge_cells("A4:A5")
    ws.append(["荣誉称号", "国家奖学金", 10, 10])
    path = tmp_path / "2025综测加分细则.xlsx"
    wb.save(path)
    return path


@pytest.fixture
def claims_excel(tmp_path: Path) -> Path:
    """合成综测表：学号/姓名列存在合并单元格（3 行同一个人）。"""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "综测汇总"
    ws.append(["学号", "姓名", "类别", "申报内容", "等级", "是否团队"])
    ws.append(["2023001", "张三", "学科竞赛", "省二等奖", "省级二等奖", "否"])
    ws.append(["2023001", "张三", "文体活动", "校运会一等奖", "校级一等奖", "否"])
    ws.append(["2023001", "张三", "学科竞赛", "国赛二等奖", "国家级二等奖", "否"])
    ws.append(["2023002", "李四", "学科竞赛", "省赛三等奖", "省级三等奖", "是"])
    ws.append(["2023002", "李四", "荣誉称号", "国家奖学金", "国家奖学金", "否"])
    ws.merge_cells("A2:A4")
    ws.merge_cells("B2:B4")
    path = tmp_path / "2024综测汇总.xlsx"
    wb.save(path)
    return path
