"""计算层：规则引擎（纯函数 + 单元测试）。

核心承诺：**所有数值计算与约束判断都在这里用 Python 代码完成**，
LLM 只负责抽取与条款定位，绝不参与算分。
"""

from .engine import (
    EngineConfig,
    MatchOutcome,
    RuleIndex,
    apply_cap,
    apply_team_factor,
    compute_all,
    compute_claims,
    compute_item_score,
    compute_student,
    dedup_take_max,
    default_config,
    match_claim,
    resolve_exclusive_groups,
)

__all__ = [
    "EngineConfig",
    "MatchOutcome",
    "RuleIndex",
    "apply_cap",
    "apply_team_factor",
    "compute_all",
    "compute_claims",
    "compute_item_score",
    "compute_student",
    "dedup_take_max",
    "default_config",
    "match_claim",
    "resolve_exclusive_groups",
]
