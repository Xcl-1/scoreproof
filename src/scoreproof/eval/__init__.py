"""评测层：往年综测表回测（天然 ground truth）。"""

from .backtest import (
    BacktestReport,
    ItemDiff,
    StudentResult,
    compare_students,
    load_ground_truth,
    run_backtest,
)
from .gateway import (
    GatewayEvaluationReport,
    GatewayNegativeCase,
    GatewayTargetResult,
    evaluate_gateway_negatives,
    wilson_interval,
)

__all__ = [
    "BacktestReport",
    "GatewayEvaluationReport",
    "GatewayNegativeCase",
    "GatewayTargetResult",
    "ItemDiff",
    "StudentResult",
    "compare_students",
    "evaluate_gateway_negatives",
    "load_ground_truth",
    "run_backtest",
    "wilson_interval",
]
from .retrieval import (
    RetrievalAblationReport,
    RetrievalCase,
    RetrievalVariantReport,
    build_ablation_report,
    evaluate_retriever,
    load_retrieval_cases,
)

__all__ = [
    "RetrievalAblationReport",
    "RetrievalCase",
    "RetrievalVariantReport",
    "build_ablation_report",
    "evaluate_retriever",
    "load_retrieval_cases",
]
