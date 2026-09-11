"""服务层：FastAPI 应用（引用面板 + SSE 所需数据结构）。"""

from .app import AppState, app, create_app, get_state

__all__ = ["AppState", "app", "create_app", "get_state"]
