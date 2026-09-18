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
3. **LangChain 工具编排 + 代码级护栏**：五个 `@tool` 通过 `bind_tools` 暴露；空结果强制澄清，账本外数字与错误引用强制拦截，模型不可用仍可查表算分。
4. **多模态材料核对**：RapidOCR → DeepSeek 严格字段草稿 → 代码校验/多信号置信度 → 低置信字段局部 VLM 决策；无视觉模型配置时明确转人工复核。

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
uv run scoreproof backtest data/sample/claims_sample.xlsx --truth data/sample/truth_sample.xlsx \
  --item-reference data/sample/item_reference_sample.csv --required-students 5 --year 2025-2026

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
│ router      双通道调度 + 引用核查 + 成对拒答门禁     │
│ manifest    双级 Hash + BM25/向量同批快照 + 原子切换  │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 计算层（纯 Python，单元测试覆盖）─────────────────┐
│ 查值 → 同类取最高（不累加）→ 封顶 → 学年过滤 → 折算   │
└──────────────────┬─────────────────────────────────┘
                   ↓
┌─ 编排层（LangChain 工具 + 显式状态机）──────────────┐
│ @tool/bind_tools → 空结果 → 引用/数字校验 → 降级路由  │
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
│   ├── retrieval/            # structured 主 + BM25/BGE + RRF/Rerank + 引用核查
│   ├── agent/                # 五个 @tool + 显式状态机 + 数字/引用护栏 + 降级路由
│   ├── eval/                 # backtest + 检索消融 + 引用/拒答成对评测
│   ├── evidence/             # 奖状字段 Schema、代码校验、置信度与 VLM 决策
│   ├── api/                  # FastAPI + SSE
│   └── cli.py                # typer 命令行
├── reports/                  # 可复跑评测报告（样本量、版本、置信区间）
├── tests/                    # 391 项自动化测试（合成/公开数据，无隐私）
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
| 引用门禁 | `retrieval.citation` | 结构化账本逐项核对；文本命中只进人工确认 |

## 命令速查

| 命令 | 作用 |
|---|---|
| `scoreproof doctor` | 环境自检（依赖/配置/规则库） |
| `scoreproof parse-pdf 细则.pdf --tables` | 抽 PDF 文本与表格，标记疑似扫描页 |
| `scoreproof sync-pdf-manifest 细则.pdf --doc-id school-rules` | 计算文档/页块 Hash，原子发布增量 manifest |
| `scoreproof sync-pdf-hybrid 细则.pdf --doc-id school-rules --chunk-mode block --embedding-backend fastembed --embedding-model BAAI/bge-small-zh-v1.5` | 用预训练 BGE 建立 BM25/Chroma 同批 manifest；表格行保留级别上下文 |
| `scoreproof search-index "第一专利人如何加分" --embedding-backend fastembed --embedding-model BAAI/bge-small-zh-v1.5 --rerank` | 查询改写后混合召回，并用 BGE Reranker 精排；输出各通道名次与分数 |
| `scoreproof eval-retrieval tests/fixtures/retrieval_test_v2.json --out reports/retrieval-ablation-v1.json` | 复跑 A=BM25、B=+BGE/RRF、C=+Rerank 的 100 条冻结集评测 |
| `scoreproof eval-citation-refusal tests/fixtures/retrieval_test_v2.json tests/fixtures/refusal_cases_v1.json --out reports/citation-refusal-v1.json` | 成对复跑引用定位、应拒答与误拒答指标 |
| `scoreproof rollback-index-manifest school-rules` | 将活动索引回滚到上一份完整 manifest |
| `scoreproof delete-index-document school-rules` | 从活动索引删除文档并保留历史快照 |
| `scoreproof parse-image 奖状.png --ocr` | 检查图片质量、计算 pHash 并运行 RapidOCR |
| `scoreproof extract-certificate 奖状.png --out result.json` | RapidOCR + DeepSeek 文本结构化；输出逐字段原值/规范值/证据/bbox/置信度、VLM 原因和人工复核状态 |
| `scoreproof eval-certificate-fields labels.jsonl --predictions predictions.jsonl --out report.json` | 输出规范值/原始值字段 F1、整证正确率、VLM 触发/调用率与样本量；合成或 n<30 自动标记为仅烟雾测试 |
| `scoreproof compare-evidence 左图.png 右图.jpg --out decision.json` | 用文件 SHA-256、pHash 汉明距离和结构化事实联合查重；只拦截/送审，不自动删除 |
| `scoreproof check-evidence-consistency claim.json evidence.json --policy policy.json` | 逐字段核对姓名、赛事别名、等级奖项、学年、团队、单位/目录和类别；信息不足进入人工复核 |
| `scoreproof eval-evidence-dedup pairs.json --out report.json` | 输出查重 Recall、Precision、F1、Wilson 区间与混淆矩阵；n<50 或非独立真实样本自动标记为仅烟雾测试 |
| `scoreproof quality-gates --out reports/quality-gates-v1.json` | 真实运行 pytest、Ruff、mypy、离线锁文件与 diff 检查，并记录 Git HEAD、耗时和工作树冻结状态 |
| `scoreproof release-readiness --candidate-version <commit>` | 汇总 RC 门禁、报告 SHA-256、成本调用量和阻塞项；烟雾报告永远不能使正式门禁通过 |
| `scoreproof extract-rules-llm 规则文本.txt -y 2025-2026 --double-check --allowed-level 第一专利人` | LLM 抽取经过五道验证及独立冲突门禁；自定义等级参数可重复；默认只审计不发布 |
| `scoreproof eval-extraction-gateway tests/fixtures/gateway_negative_cases.json -y 2025-2026` | 复跑 100 条分层负例，报告各道网关及冲突门禁的 Wilson 95% 区间 |
| `scoreproof parse-claims 综测表.xlsx` | 解析申报条目（含合并单元格 fill-down） |
| `scoreproof import-rules 规则表.xlsx -y 2025-2026` | 规则入库（建议人工校对一遍） |
| `scoreproof list-rules -y 2025-2026` | 查看规则库 |
| `scoreproof ask-score "省级二等奖能加多少分" --student-id TEST-USER -y 2025-2026 --category 学科竞赛 --level 省级二等奖` | 真实工具编排入口；DeepSeek 不可用时自动降级，输出账本、状态轨迹与数字校验结果 |
| `scoreproof calc 综测表.xlsx -y 2025-2026` | 批量核算，输出可回溯账目 |
| `scoreproof explain 省二等奖` | 解释单条申报走哪个通道、引用哪段原文 |
| `scoreproof route-pdf 细则.pdf --text 省二等奖` | 演示兜底检索与拒答 |
| `scoreproof export-backtest-template 明细.xlsx --out 逐项参照.xlsx` | 从申报明细生成稳定的逐项标注模板 |
| `scoreproof backtest 明细.xlsx --truth 汇总.xlsx --item-reference 逐项参照.xlsx --required-students 52` | 逐人/逐项回测、完整差异与 52 人数据门禁 |
| `scoreproof serve` | 启动 FastAPI（`/docs`） |

### 52 人真实回测数据准备

真实脱敏数据统一放在不会提交的 `data/eval/backtest-2025-2026/`：申报明细
`claims.xlsx`、人工历史汇总 `totals.xlsx`，以及由下列命令生成并填写的 `items.xlsx`。

```bash
uv run scoreproof export-backtest-template data/eval/backtest-2025-2026/claims.xlsx \
  --out data/eval/backtest-2025-2026/items.xlsx --year 2025-2026

# 没有业务裁决：只能报告“与历史人工结果的一致率/差异率”
uv run scoreproof backtest data/eval/backtest-2025-2026/claims.xlsx \
  --truth data/eval/backtest-2025-2026/totals.xlsx \
  --item-reference data/eval/backtest-2025-2026/items.xlsx \
  --mode historical-reference --required-students 52 --year 2025-2026 \
  --out reports/backtest-52-v1.json --diff-out reports/backtest-52-diffs-v1.csv
```

`items.xlsx` 的“历史/裁决得分”必须是最终计入总分的逐项贡献（去重、互斥、封顶后），
每人的逐项合计应等于 `totals.xlsx`。所有差异必须填写“差异归因”；只有每项都经业务
确认并把“已裁决”设为“是”后，才可改用 `--mode adjudicated-truth` 并称为准确率。

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
| 阶段 4 | 增量索引、混合检索、LangChain 工具编排 | ✅ 4.1～4.5 已完成：索引、混合检索、五工具状态机、代码级引用门禁与成对拒答评测均通过真实 CLI/API 验收 |
| 阶段 5 | 确定性计算与 52 人回测 | 🟡 逐人/逐项回测、双口径、完整差异与 52 人门禁已落地；5 人合成文件真实 CLI 通过，52 人脱敏历史数据待提供 |
| 阶段 6 | OCR + LLM/VLM + 查重 | 🟡 **6.1～6.3 工具链已落地但阶段未完成**：真实 RapidOCR + DeepSeek CLI/Uvicorn 上传 API、联合查重 CLI/API 与一致性入口已跑通；VLM 未调用，n≥30 真实脱敏字段集与 n≥50 对独立真实查重集仍缺 |
| 阶段 7 | 消融、全量评测与结项 | 🟡 **候选冻结与统一评测门禁已落地，阶段未完成**：质量命令和只读 CLI/API 已实跑；正式阻塞项被如实保留，Web 三端仍按 V3.0 延后至 P2 |

## 测试

检索冻结集（基于一份真实公开细则人工整理，**不是生产用户日志**）的最终报告见 `reports/retrieval-ablation-v1.json`：A/B/C 的 Hit@5 分别为 0.96/0.98/0.98，MRR@10 为 0.863/0.915/0.915；C 档 P95 为 1.139 秒、实际处理 2,000 个候选对。Rerank 相对 B 的 MRR 增量为 0，nDCG@10 增量为 +0.000336，按实验纪律如实披露。真实 API 双请求测试为冷启动 4.84 秒、模型缓存后的热请求 1.02 秒；P95 门槛按稳定运行口径统计，部署时应预热模型。

编排护栏报告见 `reports/orchestration-guardrails-v1.json`：100/100 个账本外伪造数字被拦截；真实 CLI 降级、真实 Uvicorn HTTP 入口与真实 DeepSeek `bind_tools` 均通过。模型侧学生身份统一替换为 `CURRENT_STUDENT`，复测使用本地合成身份与仓库示例规则，不含真实学生数据。

引用与拒答成对报告见 `reports/citation-refusal-v1.json`：在同一份真实公开细则 PDF 上，可回答问法 n=100 的引用定位正确率为 98%（95% CI 93.0%–99.4%），人工构造域外负例 n=50 的正确拒答率为 100%（95% CI 92.9%–100%），可回答问法误拒答率为 0%（95% CI 0%–3.7%）。两条引用定位失败已保留在报告中；集合不是生产用户日志。真实 Uvicorn 复测同时覆盖了文本候选人工确认、域外拒答和结构化规则表行级自动核算；真实 DeepSeek 在结构化参数锁定后完成 `lookup_rule → calc_score`，最终文档名、表名与行号均通过代码校验且未降级。

奖状字段烟雾报告见 `reports/certificate-fields-smoke-v1.json`：5 张合成图片均实际经过 RapidOCR 与 DeepSeek 文本 API；规范值 micro-F1 为 0.8788，整证完全正确 1/5，VLM 决策触发 4/5、实际调用 0/5。失败主要来自两张赛事名漏掉级别前缀，以及 4 张证书没有“个人”原文、系统按“不猜测”原则将团队属性置空。**该结果仅验证代码、CLI/API 与外部文本服务主链路，不是正式业务评测，不能用于简历；原始值标签尚未提供，raw F1 为 null。**

查重烟雾报告见 `reports/evidence-dedup-smoke-v1.json`：真实 CLI 读取合成奖状文件及其完全相同、JPEG 压缩、亮度变化、裁剪缩放变体，n=5（重复正例 4、不同奖状负例 1）的烟雾结果为 TP=4、FP=0、TN=1、FN=0；Recall/Precision 点估计虽均为 1.0，但各自 Wilson 95% CI 下界仅 0.5101。**该集合规模小、类别不充分且图片为合成，不具备 n≥50 对正式验收资格，数字不得写入简历。**

阶段 7 就绪审计见 `reports/quality-gates-v1.json` 与 `reports/release-readiness-v1.json`：固定质量命令真实执行后 391 项测试、Ruff、mypy、`uv lock --offline --check` 和 `git diff --check` 均通过；由于当前变更尚未提交，候选冻结门禁仍阻塞。统一审计认可网关、A/B/C 三档消融、引用/拒答和编排护栏报告，但继续阻塞复杂 PDF 正式回归、n≥30 字段集、真实 VLM、n≥50 查重对、52 人回测和真实用户试用，并把缺少统一 token/货币成本记录列为警告。**这只是阶段 7 工具链与真实入口验证，不代表 RC 或项目结项。**

真正启用视觉模型前必须在 `.env` 二选一配置：`SCOREPROOF_VLM_PROVIDER=qwen-vl-plus` + `DASHSCOPE_API_KEY`，或 `SCOREPROOF_VLM_PROVIDER=glm-4v` + `ZHIPUAI_API_KEY`。当前 DeepSeek 是文本模型，不会被当作 VLM；未配置时低置信字段只进入人工复核。

```bash
uv run pytest              # 全部单元测试
uv run pytest -k calc      # 只跑计算引擎
uv run ruff check src tests
```

---

**许可**：MIT（个人项目）。真实材料与评测数据不在本仓库内。
