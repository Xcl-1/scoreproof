"""规则层：schema（数据模型）+ store（SQLite 持久化）+ extractor（抽取）。"""

from .extractor import (
    RULE_BATCH_SCHEMA,
    RULE_DRAFT_SCHEMA,
    Extractor,
    HeuristicExtractor,
    LLMExtractor,
    RuleDraft,
    drafts_to_rules,
)
from .gateway import (
    ExtractionGateway,
    GatewayContext,
    GatewayIssue,
    GatewayItem,
    GatewayReport,
    RuleDraftInput,
    chunk_hash,
)
from .store import RuleStore, export_json, import_json

__all__ = [
    "RULE_BATCH_SCHEMA",
    "RULE_DRAFT_SCHEMA",
    "Extractor",
    "ExtractionGateway",
    "HeuristicExtractor",
    "LLMExtractor",
    "RuleDraft",
    "RuleDraftInput",
    "RuleStore",
    "GatewayContext",
    "GatewayIssue",
    "GatewayItem",
    "GatewayReport",
    "chunk_hash",
    "drafts_to_rules",
    "export_json",
    "import_json",
]
