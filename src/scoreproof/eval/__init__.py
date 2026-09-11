"""评测层：往年综测表回测（天然 ground truth）。"""

from .backtest import (
    BacktestReport,
    ItemDiff,
    StudentResult,
    compare_students,
    load_ground_truth,
    run_backtest,
)

__all__ = [
    "BacktestReport",
    "ItemDiff",
    "StudentResult",
    "compare_students",
    "load_ground_truth",
    "run_backtest",
]
