"""scoreproof · 综测加分智能核算系统。

一句话：把散落在 PDF/Word/Excel/图片中的加分规则结构化为可执行规则库，
用**确定性代码**完成算分，每条分值都能回溯到原文出处。

    LLM 只负责抽取与条款定位，所有数值计算与约束判断由 Python 完成。

包内分层（依赖方向自上而下，禁止反向 import）::

    api / cli      <- 入口
    eval           <- 回测（往年综测表 = 天然 ground truth）
    retrieval      <- 双通道调度（结构化主 + 原文兜底 + 拒答）
    calc           <- 确定性计算引擎（纯函数）
    rules          <- schema / store(SQLite) / extractor
    ingest         <- excel / pdf / docx / image 分流
    normalize      <- 等级、学年、类别归一化（脏活集中处）
    config/errors  <- 基础设施
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import (
    AmbiguousRule,
    DataSourceError,
    RuleNotFound,
    SchemaValidationError,
    ScoreProofError,
    UnsupportedModality,
    VersionConflict,
)
from .normalize import canonical_level, normalize_academic_year, normalize_level
from .schema import (
    Claim,
    ConstraintSpec,
    Evidence,
    GroupBreakdown,
    Rule,
    RuleMatch,
    Ruleset,
    ScoreBreakdown,
    SourceRef,
)

__all__ = [
    "AmbiguousRule",
    "Claim",
    "ConstraintSpec",
    "DataSourceError",
    "Evidence",
    "GroupBreakdown",
    "Rule",
    "RuleMatch",
    "RuleNotFound",
    "Ruleset",
    "SchemaValidationError",
    "ScoreBreakdown",
    "ScoreProofError",
    "SourceRef",
    "UnsupportedModality",
    "VersionConflict",
    "__version__",
    "canonical_level",
    "normalize_academic_year",
    "normalize_level",
]
