"""评测层：往年综测表回测（天然 ground truth）。"""

from .backtest import (
    BacktestReport,
    ItemDiff,
    StudentResult,
    compare_students,
    load_ground_truth,
    run_backtest,
)
from .citation import (
    CitationRefusalReport,
    RefusalCase,
    evaluate_citation_refusal,
    load_refusal_cases,
)
from .gateway import (
    GatewayEvaluationReport,
    GatewayNegativeCase,
    GatewayTargetResult,
    evaluate_gateway_negatives,
    wilson_interval,
)
from .retrieval import (
    RetrievalAblationReport,
    RetrievalCase,
    RetrievalVariantReport,
    build_ablation_report,
    evaluate_retriever,
    load_retrieval_cases,
)

__all__ = [
    "BacktestReport",
    "CitationRefusalReport",
    "GatewayEvaluationReport",
    "GatewayNegativeCase",
    "GatewayTargetResult",
    "ItemDiff",
    "RefusalCase",
    "RetrievalAblationReport",
    "RetrievalCase",
    "RetrievalVariantReport",
    "StudentResult",
    "build_ablation_report",
    "compare_students",
    "evaluate_citation_refusal",
    "evaluate_gateway_negatives",
    "evaluate_retriever",
    "load_ground_truth",
    "load_refusal_cases",
    "load_retrieval_cases",
    "run_backtest",
    "wilson_interval",
]
