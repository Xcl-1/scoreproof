"""运行配置：从环境变量 / .env 读取，集中一处便于审计。

红线：所有密钥只从环境变量来，绝不写进代码或提交进仓库。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取（避免额外依赖）：已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ and value:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    """全局设置。"""

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SCOREPROOF_DATA_DIR", "data")))
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("SCOREPROOF_DB_PATH", "data/rules/rules.sqlite"))
    )
    index_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("SCOREPROOF_INDEX_DB_PATH", "data/index/index.sqlite")
        )
    )
    vector_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SCOREPROOF_VECTOR_DIR", "data/index/chroma"))
    )
    model_cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SCOREPROOF_MODEL_CACHE", "data/models/fastembed"))
    )
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("SCOREPROOF_EMBEDDING_BACKEND", "hash")
    )
    embedding_model: str | None = field(
        default_factory=lambda: os.getenv("SCOREPROOF_EMBEDDING_MODEL") or None
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("SCOREPROOF_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    rerank_candidate_k: int = field(
        default_factory=lambda: int(os.getenv("SCOREPROOF_RERANK_CANDIDATE_K", "20"))
    )
    rerank_base_weight: float = field(
        default_factory=lambda: float(os.getenv("SCOREPROOF_RERANK_BASE_WEIGHT", "4"))
    )
    rerank_model_weight: float = field(
        default_factory=lambda: float(os.getenv("SCOREPROOF_RERANK_MODEL_WEIGHT", "1"))
    )
    log_level: str = field(default_factory=lambda: os.getenv("SCOREPROOF_LOG_LEVEL", "INFO"))

    llm_base_url: str = field(
        default_factory=lambda: os.getenv("SCOREPROOF_LLM_BASE_URL", "https://api.deepseek.com")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("SCOREPROOF_LLM_MODEL", "deepseek-v4-flash")
    )
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY") or None)

    vlm_provider: str | None = field(
        default_factory=lambda: os.getenv("SCOREPROOF_VLM_PROVIDER") or None
    )
    dashscope_api_key: str | None = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY") or None
    )
    zhipuai_api_key: str | None = field(
        default_factory=lambda: os.getenv("ZHIPUAI_API_KEY") or None
    )

    # ---------- 派生路径 ----------
    @property
    def raw_dir(self) -> Path:
        """原始材料目录：**绝不提交**。"""
        return self.data_dir / "raw"

    @property
    def rules_dir(self) -> Path:
        """结构化规则库（JSON/SQLite）。"""
        return self.data_dir / "rules"

    @property
    def eval_dir(self) -> Path:
        """评测集 + 往年综测表（脱敏）。"""
        return self.data_dir / "eval"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def vlm_configured(self) -> bool:
        provider = (self.vlm_provider or "").strip().lower()
        return (provider == "qwen-vl-plus" and bool(self.dashscope_api_key)) or (
            provider == "glm-4v" and bool(self.zhipuai_api_key)
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.rules_dir,
            self.eval_dir,
            self.index_db_path.parent,
            self.vector_dir,
            self.model_cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def safe_repr(self) -> dict:
        """可安全打印的配置摘要（不含密钥）。"""
        return {
            "data_dir": str(self.data_dir),
            "db_path": str(self.db_path),
            "index_db_path": str(self.index_db_path),
            "vector_dir": str(self.vector_dir),
            "model_cache_dir": str(self.model_cache_dir),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "rerank_candidate_k": self.rerank_candidate_k,
            "rerank_base_weight": self.rerank_base_weight,
            "rerank_model_weight": self.rerank_model_weight,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_configured": self.llm_configured,
            "vlm_provider": self.vlm_provider,
            "vlm_configured": self.vlm_configured,
        }


@lru_cache(maxsize=1)
def get_settings(*, reload: bool = False) -> Settings:
    """单例配置。``reload=True`` 可重新读取 .env（测试用）。"""
    if reload:
        get_settings.cache_clear()
    _load_dotenv(PROJECT_ROOT / ".env")
    return Settings()


__all__ = ["PROJECT_ROOT", "Settings", "get_settings"]
