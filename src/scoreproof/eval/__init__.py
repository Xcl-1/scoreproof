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
from .certificate import (
    CertificateEvaluationReport,
    FieldMetric,
    evaluate_certificate_fields,
    load_jsonl,
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
    "CertificateEvaluationReport",
    "GatewayEvaluationReport",
    "GatewayNegativeCase",
    "GatewayTargetResult",
    "ItemDiff",
    "ItemExpectation",
    "FieldMetric",
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
    "evaluate_certificate_fields",
    "evaluate_gateway_negatives",
    "evaluate_retriever",
    "item_reference_template",
    "load_ground_truth",
    "load_jsonl",
    "load_item_expectations",
    "load_refusal_cases",
    "load_retrieval_cases",
    "run_backtest",
    "wilson_interval",
]
