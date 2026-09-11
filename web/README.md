# web/ · 前端（P3 待实现）

后端接口已经就绪，前端只需要消费它们。计划包含两个入口：

| 页面 | 使用者 | 依赖接口 |
|---|---|---|
| 同学自助查询 | 学生 | `POST /api/calc`、`POST /api/explain` |
| 班委批量核算 + 引用面板 | 班委 | `POST /api/calc/stream`（SSE）、`POST /api/rules/import` |
| 材料校对（P2） | 班委 | `POST /api/evidence`、`GET /api/evidence/duplicates` |

## 接口约定（可直接对着 `/docs` 调）

- **引用面板必需字段**：`breakdown.matches[].source`（`doc` / `page` / `clause` / `text` / `bbox`）
  —— 这是"每条分值可回溯原文"的落地方式，前端务必把它渲染成可点击的出处。
- **`counted=false`** 的命中要在界面上显式划掉（同类取最高 / 互斥组未计入），并展示 `reason`。
- **`needs_review=true`** 与 `breakdown.review_claims` 必须标红进入复核队列（人机协同，不做全自动）。
- **拒答**：`channel=none` 时显示 `reason`（"未在细则中找到，建议咨询辅导员"），不要显示 0 分账单。
- **兜底候选**：`channel=vector` 时 `score_candidates` 只是候选值，必须由人确认后才可入规则库。

## 本地联调

```bash
# 1. 生成合成数据并入库，然后起后端（CORS 已放开给本地开发）
uv run python scripts/make_sample_data.py
uv run scoreproof import-rules data/sample/rules_sample.xlsx --year 2025-2026
uv run scoreproof serve --port 8000 --reload
```

启动后打开 <http://127.0.0.1:8000/docs> 可直接试调；前端开发服务器请把 `/api` 代理到 `http://127.0.0.1:8000`。

> 上公网前务必收敛 `api/app.py` 里的 `allow_origins=["*"]`，并加上鉴权 —— 综测数据涉及隐私。
