"""评测层：往年综测表回测（天然 ground truth）。"""

from .backtest import (
    BacktestMode,
    BacktestReport,
    ItemDiff,
    ItemExpectation,
    StudentResult,
    claim_item_key,
    compare_items,
    compare_students,
    item_reference_template,
    load_ground_truth,
    load_item_expectations,
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
    "BacktestMode",
    "BacktestReport",
    "CitationRefusalReport",
    "GatewayEvaluationReport",
    "GatewayNegativeCase",
    "GatewayTargetResult",
    "ItemDiff",
    "ItemExpectation",
    "RefusalCase",
    "RetrievalAblationReport",
    "RetrievalCase",
    "RetrievalVariantReport",
    "StudentResult",
    "build_ablation_report",
    "claim_item_key",
    "compare_items",
    "compare_students",
    "evaluate_citation_refusal",
    "evaluate_gateway_negatives",
    "evaluate_retriever",
    "item_reference_template",
    "load_ground_truth",
    "load_item_expectations",
    "load_refusal_cases",
    "load_retrieval_cases",
    "run_backtest",
    "wilson_interval",
]
