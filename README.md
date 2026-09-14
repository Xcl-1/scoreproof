# scoreproof · 综测加分智能核算系统

> 面向高校综测场景的**异构文档核算系统**：把散落在 PDF / Word / Excel / 图片中的加分规则结构化为可执行规则库，
> 用**确定性代码**完成算分，并对奖状图片做字段抽取、一致性与重复申报核对，**每条分值可回溯至原文出处**。

**核心承诺（也是最值得写在简历上的一句）**：

> LLM 只负责**抽取与条款定位**，所有**数值计算与约束判断由 Python 代码完成**。
> 结果可复现、可单元测试、可追溯 —— 杜绝模型心算。

---

## 三个差异化亮点

1. **异构文档 → 结构化规则库**：不是问答，是把非结构化规则变成可执行规则（带学年/学院/版本/出处）。
2. **双通道检索 + 确定性计算**：结构化查表优先 → 原文检索兜底 → 未命中**拒答**（不瞎给分）。
3. **LangChain 工具编排 + 代码级护栏**：五个 `@tool` 通过 `bind_tools` 暴露；空结果强制澄清，账本外数字强制拦截，模型不可用仍可查表算分。
4. **多模态材料核对**（P2）：图片抽字段（字段级置信度 + 人工校对闭环）、pHash + 字段指纹查重、申报与证据一致性比对。

## 快速开始

```bash
# 1. 安装（Windows 下若默认缓存不可写，指定工作区内缓存目录）
uv sync --all-extras

# 2. 环境自检：依赖 / 配置 / 目录 / 规则库
uv run scoreproof doctor

# 3. 演示数据：生成合成综测表 + 规则表（不含任何真实同学数据）
uv run python scripts/make_sample_data.py

# 4. 导入规则 -> 核算 -> 回测
uv run scoreproof import-rules data/sample/rules_sample.xlsx --year 2025-2026
uv run scoreproof calc data/sample/claims_sample.xlsx --year 2025-2026
uv run scoreproof backtest data/sample/claims_sample.xlsx --truth data/sample/truth_sample.xlsx --year 2025-2026

# 5. 起服务（/docs 可交互调试）
uv run scoreproof serve --port 8000
```

## 架构

```
┌─ 接入层（按格式分流）──────────────────────────────┐
│ excel_loader  openpyxl/pandas → 综测表 + 规则表     │
│ pdf_loader    pymupdf / pdfplumber → 细则文本+表格  │
│ docx_loader   python-docx → 正文 + 表格（保序）     │
│ image_loader  预处理 + OCR（RapidOCR）               │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 结构化层（一次性，人工校对）──────────────────────┐
│ extractor  LangChain/程序抽取 → 严格草稿 schema      │
│ gateway    五道验证 + 独立冲突门禁 → 字符偏移/审计    │
│ normalize  归一化：省部级→省级、第二名→二等奖        │
│ verify     一致性核对 + pHash 查重（P2）              │
│ store      SQLite 规则库（学年/学院/版本/出处）     │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 检索层 ──────────────────────────────────────────┐
│ structured  精确查表（主通道，给确定分值）           │
│ fallback    查询改写 → BM25 + BGE 向量 → RRF → 精排 │
│ router      双通道调度 + 置信度 + 未命中拒答         │
│ manifest    双级 Hash + BM25/向量同批快照 + 原子切换  │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 计算层（纯 Python，单元测试覆盖）─────────────────┐
│ 查值 → 同类取最高（不累加）→ 封顶 → 学年过滤 → 折算   │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 编排层（LangChain 工具 + 显式状态机）──────────────┐
│ @tool/bind_tools → 空结果分支 → 数字校验 → 降级路由   │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 服务层  FastAPI + SSE 流式 + 引用面板 + 校对界面 ──┐
└───────────────────────────────────────────────────┘
```

## 目录结构

```
scoreproof/
├── pyproject.toml            # 依赖与工具配置（可选 extras：agent / multimodal / retrieval / llm）
├── .env.example              # 配置模板（密钥只从环境变量读，绝不入库）
├── data/                     # 原始材料与规则库（.gitignore 已隔离）
│   ├── raw/                  # 原始文档：绝不提交
│   ├── rules/                # 结构化规则库（SQLite / JSON）
│   └── eval/                 # 评测集 + 往年综测表（脱敏）
├── scripts/
│   └── make_sample_data.py   # 生成合成演示数据（可安全提交）
├── src/scoreproof/
│   ├── schema.py             # Rule / Claim / Evidence / ScoreBreakdown（pydantic v2）
│   ├── normalize.py          # 等级、学年、类别归一化（脏活集中处）
│   ├── config.py             # 环境配置 + 安全摘要
│   ├── errors.py             # 带 code 的领域异常
│   ├── ingest/               # excel / pdf / docx 分流（含合并单元格 fill-down）
│   ├── rules/                # store(SQLite) + extractor(规则抽取)
│   ├── indexing/             # 双级 Hash + manifest + BM25/Chroma 同批发布
│   ├── calc/                 # 计算引擎（纯函数 + 可解释账本）
│   ├── retrieval/            # structured 主 + 查询改写 + BM25/BGE + RRF/Rerank
│   ├── agent/                # 五个 @tool + 显式状态机 + 数字护栏 + 降级路由
│   ├── eval/                 # backtest + 检索三档消融评测
│   ├── api/                  # FastAPI + SSE
│   └── cli.py                # typer 命令行
├── reports/                  # 可复跑评测报告（样本量、版本、置信区间）
├── tests/                    # 317 项自动化测试（合成/公开数据，无隐私）
└── web/                      # 前端占位（V3.0：P2 延后）
```

## 数据模型（三张核心表）

```jsonc
// 1) 规则库 rule —— 确定性核心
{ "academic_year": "2025-2026", "college": "计算机学院",
  "category": "学科竞赛", "level": "省级二等奖", "score": 8,
  "synonyms": ["省二等奖","省级二等","省赛第二名"],
  "constraints": { "dedup_group": "学科竞赛", "cap": 15,
                   "require_catalog": "认可竞赛目录", "team_factor": 0.5 },
  "source": { "doc": "2025综测细则.pdf", "page": 4, "table": "加分标准表", "clause": "第三章第7条" } }

// 2) 申报条目 claim
{ "student_id": "2023xxxx", "raw_text": "省二等奖", "category": "学科竞赛",
  "level": "省级二等奖", "evidence_ids": ["ev_001"], "status": "待核对" }

// 3) 证据 evidence（多模态核心）
{ "id": "ev_001", "type": "image", "path": "data/raw/xxx.jpg",
  "ocr_text": "...", "fields": { }, "field_confidence": { },
  "phash": "a1b2c3...", "extractor": "ocr+llm", "manual_corrected": true }
```

## 计算语义（全部可测）

| 语义 | 实现位置 | 说明 |
|---|---|---|
| 同类取最高，不累加 | `calc.engine.dedup_take_max` | 按 `dedup_group` 分组，只保留一条 |
| 单项 / 单类封顶 | `calc.engine.apply_cap` | `cap=None` 表示"不设额度" |
| 团队奖折算 | `calc.engine.apply_team_factor` | 先折算后封顶 |
| 学年 / 学院过滤 | `calc.engine.RuleIndex` | `college=None` 视为校级通用 |
| 互斥组裁决 | `calc.engine.resolve_exclusive_groups` | 保留总分更高的组合，完全确定性 |
| 未命中拒答 | `retrieval.router.REFUSAL_MESSAGE` | 兜底候选分值**不自动计分** |

## 命令速查

| 命令 | 作用 |
|---|---|
| `scoreproof doctor` | 环境自检（依赖/配置/规则库） |
| `scoreproof parse-pdf 细则.pdf --tables` | 抽 PDF 文本与表格，标记疑似扫描页 |
| `scoreproof sync-pdf-manifest 细则.pdf --doc-id school-rules` | 计算文档/页块 Hash，原子发布增量 manifest |
| `scoreproof sync-pdf-hybrid 细则.pdf --doc-id school-rules --chunk-mode block --embedding-backend fastembed --embedding-model BAAI/bge-small-zh-v1.5` | 用预训练 BGE 建立 BM25/Chroma 同批 manifest；表格行保留级别上下文 |
| `scoreproof search-index "第一专利人如何加分" --embedding-backend fastembed --embedding-model BAAI/bge-small-zh-v1.5 --rerank` | 查询改写后混合召回，并用 BGE Reranker 精排；输出各通道名次与分数 |
| `scoreproof eval-retrieval tests/fixtures/retrieval_test_v2.json --out reports/retrieval-ablation-v1.json` | 复跑 A=BM25、B=+BGE/RRF、C=+Rerank 的 100 条冻结集评测 |
| `scoreproof rollback-index-manifest school-rules` | 将活动索引回滚到上一份完整 manifest |
| `scoreproof delete-index-document school-rules` | 从活动索引删除文档并保留历史快照 |
| `scoreproof parse-image 奖状.png --ocr` | 检查图片质量、计算 pHash 并运行 RapidOCR |
| `scoreproof extract-rules-llm 规则文本.txt -y 2025-2026 --double-check --allowed-level 第一专利人` | LLM 抽取经过五道验证及独立冲突门禁；自定义等级参数可重复；默认只审计不发布 |
| `scoreproof eval-extraction-gateway tests/fixtures/gateway_negative_cases.json -y 2025-2026` | 复跑 100 条分层负例，报告各道网关及冲突门禁的 Wilson 95% 区间 |
| `scoreproof parse-claims 综测表.xlsx` | 解析申报条目（含合并单元格 fill-down） |
| `scoreproof import-rules 规则表.xlsx -y 2025-2026` | 规则入库（建议人工校对一遍） |
| `scoreproof list-rules -y 2025-2026` | 查看规则库 |
| `scoreproof ask-score "省级二等奖能加多少分" --student-id TEST-USER -y 2025-2026 --category 学科竞赛 --level 省级二等奖` | 真实工具编排入口；DeepSeek 不可用时自动降级，输出账本、状态轨迹与数字校验结果 |
| `scoreproof calc 综测表.xlsx -y 2025-2026` | 批量核算，输出可回溯账目 |
| `scoreproof explain 省二等奖` | 解释单条申报走哪个通道、引用哪段原文 |
| `scoreproof route-pdf 细则.pdf --text 省二等奖` | 演示兜底检索与拒答 |
| `scoreproof backtest 明细.xlsx --truth 汇总.xlsx` | 回测计算准确率 |
| `scoreproof serve` | 启动 FastAPI（`/docs`） |

## 红线与合规

1. **隐私**：同学姓名、学号、成绩、证书照片**绝不进公开仓库**；`data/raw/`、`data/eval/` 已被 `.gitignore` 隔离，本地处理 + 脱敏。
2. **不编数字**：没测出来的指标不写进 README / 简历（回测数字必须来自 `scoreproof backtest`）。
3. **人机协同**：不追求全自动，保留人工校对闭环（低置信度一律标红进复核队列）。
4. **密钥**：只从环境变量 / `.env` 读取（`.env` 不入库），`.env.example` 只放变量名。

## 里程碑

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 0～1 | 口径、Schema、数据库与工程基线 | ✅ 已完成 |
| 阶段 2 | 异构解析与 LangChain 抽取 | 🟡 DeepSeek 真实 API 烟雾测试已通过；复杂版面回归集仍待验收 |
| 阶段 3 | 五道抽取验证 + 独立发布冲突门禁 | ✅ 代码链路与 100 条分层冻结负例完成 |
| 阶段 4 | 增量索引、混合检索、LangChain 工具编排 | 🟡 4.1～4.4 已完成：增量索引、BGE + RRF + Rerank、五工具状态机及真实 CLI/API/DeepSeek 验收通过；4.5 引用核查与拒答待做 |
| 阶段 5 | 确定性计算与 52 人回测 | 🟡 计算核心与 41 项边界测试完成；真实回测待做 |
| 阶段 6 | OCR + LLM/VLM + 查重 | 🟡 RapidOCR、预处理、pHash 完成；字段链路与评测待做 |
| 阶段 7 | 消融、全量评测与结项 | ⏳ 待做；Web 三端按 V3.0 延后至 P2 |

## 测试

检索冻结集（基于一份真实公开细则人工整理，**不是生产用户日志**）的最终报告见 `reports/retrieval-ablation-v1.json`：A/B/C 的 Hit@5 分别为 0.96/0.98/0.98，MRR@10 为 0.863/0.915/0.915；C 档 P95 为 1.139 秒、实际处理 2,000 个候选对。Rerank 相对 B 的 MRR 增量为 0，nDCG@10 增量为 +0.000336，按实验纪律如实披露。真实 API 双请求测试为冷启动 4.84 秒、模型缓存后的热请求 1.02 秒；P95 门槛按稳定运行口径统计，部署时应预热模型。

编排护栏报告见 `reports/orchestration-guardrails-v1.json`：100/100 个账本外伪造数字被拦截；真实 CLI 降级、真实 Uvicorn HTTP 入口与真实 DeepSeek `bind_tools` 均通过。模型侧学生身份统一替换为 `CURRENT_STUDENT`，复测使用本地合成身份与仓库示例规则，不含真实学生数据；这次仅覆盖一条字段完整的核算主链路，阶段 4.5 的引用核查与成对拒答评测尚未包含。

```bash
uv run pytest              # 全部单元测试
uv run pytest -k calc      # 只跑计算引擎
uv run ruff check src tests
```

---

**许可**：MIT（个人项目）。真实材料与评测数据不在本仓库内。
