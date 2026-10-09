# 阶段 7.10：正式验收数据盘点与准备清单

盘点时间：2026-10-09（Asia/Shanghai）

盘点版本：`bfb671a1eb1156c3bfcfd2ed19eed16b13f2a9f2`

机器可读结果：`reports/formal-data-inventory-v1.json`

## 当前结论

`data/eval/` 只有 `.gitkeep`，四个优先正式门禁的清单均不存在；当前没有可执行的足量正式数据。本阶段选择停止条件 B。预检通过仅说明输入完整，正式指标仍须由对应真实 CLI 评测产生。

| 类型 | 仓库现状 | 正式资格 |
|---|---|---|
| 真实授权业务数据 | `data/eval/` 无文件 | 0 |
| 公开真实数据 | 官网 6 页 PDF 1 份，SHA-256 `1c1fc3c899017e641f7a92430dd342634c4f50df5d67ebab6f5547e3d4b840c5` | 仅跨页表工程烟雾 n=1 |
| 脱敏真实数据 | 未发现可核对授权和原件的业务图片、金标或历史申报 | 0 |
| 合成数据 | 5 张奖状、示例 Excel/CSV、合成规则和各种变体 | 只能烟雾 |
| Mock/单测夹具 | `tests/fixtures/`、`.tmp/run-*/case-*/` 下的大量生成文件 | 不能正式计数 |
| 公开细则派生工作簿 | `outputs/scoreproof-real-acceptance-seed/`；细则真实，学生申报与标准答案为仿真 | 不可称 52 人回测 |

`.tmp/` 中虽有 248 份 PDF 路径、41 个不同字节哈希，主要是测试生成与反复运行产物。哈希不同不等于独立真实授权文档；所有这些文件均排除在正式盘点之外。不得用复制、改名、重新压缩、重新导出或不同裁剪凑样本量。

真实入口复测：已安装 `scoreproof.exe eval-complex-pdf` 读取上述公开 PDF，跨页表 1/1，`formal_gate_eligible=false`；已安装 `scoreproof.exe audit-formal-data` 在隔离清单中实际读取同一公开 PDF，识别 SHA-256 与 `real_public` 类型，按 n=1、三类不足以退出码 2 拦截。该隔离副本只用于入口验证，不计入正式数据集。

重新运行 `release-readiness` 后仍阻塞：`quality`、`complex_pdf`、`rule_extraction`、`certificate_fields`、`evidence_dedup`、`backtest_52`、`user_trial`。缺少五份正式报告：规则抽取、奖状字段、查重、52 人回测、真实用户试用。质量项还需要在本批变更提交后重新冻结候选 HEAD；本阶段未经授权不提交。

## 四个优先门禁的数据合同

所有正式文件放在本地 `data/eval/`，该目录已被 Git 忽略。清单只使用安全相对路径；原图、未脱敏标签、授权原件及身份映射不得写入 `reports/`、`tests/fixtures/` 或公开仓库。授权编号或公开来源 URL 保存在私有清单中；公开报告只保留其存在性和输入哈希。

### 1. 复杂 PDF：先收集

目录：`data/eval/complex-pdf/dataset.json` 和 `data/eval/complex-pdf/documents/*.pdf`。复制 `data/sample/complex_pdf_regression_template.json` 作为清单起点。

- 至少 30 份**独立原始 PDF**；多栏、跨页表、扫描件各至少 10 份。每份原始文件按 SHA-256 去重，允许一份文档有多个场景标签，但总独立文件数仍须 ≥30，三类各自的独立文件数仍须 ≥10。
- 顶层填写 `dataset_version`、`authorization_reference`（官网来源或授权记录编号）、`independent_real_documents=true`、`cases`。每例填写唯一 `case_id`、相对 `document`、`scenario`、`real_document=true`、`synthetic=false`。
- 多栏例填至少两个按正确阅读顺序的 `expected_ordered_fragments`；跨页表填 `expected_table_fragments` 与 `expected_min_table_rows`；扫描例填 1 起始的 `expected_scanned_pages`。片段需能人工复核，不应含学生个人信息。
- 正式指标：总体和三类成功率都 ≥95%，各报 Wilson 95% CI。若每类恰好 n=10，则该类必须 10/10 成功；总 n=30 至少 29/30 才达到 95% 点估计。
- 当前可用 1 份公开真实跨页表文档；还缺至少 29 份独立 PDF，且多栏与扫描各缺 10 份、跨页表至少缺 9 份。扫描识别成功仅指识别扫描页并显式转复核，不等同 OCR 内容全对。

### 2. 规则抽取：其次

目录：`data/eval/rule-extraction/dataset.json`。复制 `data/sample/rule_extraction_template.json`。至少 50 个不同来源、不同原文哈希的独立真实授权金标；覆盖正文、矩阵表、跨页表与补充通知。不能把合成规则或同一原文块改 ID 后重复计入。

顶层需 `dataset_version`、`authorization_reference`、`independent_real_samples=true`。每例需唯一 `case_id`、不含原始路径的 `source_id`、已脱敏 `source_text`、`academic_year`、可选 `college/page/table/category_hint/allowed_levels/risk_level`、`real_source=true`、`synthetic=false`，以及人工确认的完整 `expected_rules`。每条金标使用 `RuleDraftInput` 字段（类别、等级、分值、同义词、封顶、团队系数、条款、证据原文、名次、项目名、生效日期）；金标必须先通过同一五道网关。按计划书 §8.3 隔日双次独立标注并记录分歧处理。要求完整规则正确率 ≥96%（n=50 时至少 48 条），同时报告逐字段 P/R/F1 与 95% CI。需真实文本模型调用。

### 3. 奖状字段：第三

目录：`data/eval/certificate-fields/labels.jsonl` 与 `images/匿名编号.png|jpg|webp`。复制 `data/sample/certificate_fields_formal_template.jsonl` 后逐图填写，模板默认 `synthetic=true`、`real_source=false`、`redacted=false`，不得原样送验。

至少 30 张不同原件、不同文件哈希的真实授权**脱敏**图片。每行需要唯一 `evidence_id`、相对 `image_path`、真实文件 `sha256`、`synthetic=false`、`real_source=true`、`redacted=true`、`authorization_reference`。规范值七字段：`name`、`event_name`、`tier`、`award`、`award_date`、`issuer`、`team_attribute`；原始值 `raw_fields` 必须以“姓名、赛事名称、级别、奖项/名次、获奖日期、颁发单位、团队属性”七个中文键逐一标注。看不清时填 `null` 并记录人工复核，不得猜测。脱敏时遮蔽姓名、学号、二维码、证书编号和无关第三方信息，保留赛事、奖项、日期等评测必需内容；移除 EXIF，并在同一匿名身份的图片与标签中使用一致代号。

正式报告需分别给出原始值/规范值 micro-F1、七字段各自 F1、整证正确率、VLM 触发率及实际调用率、VLM 对 F1 的增益，并为比例指标报 95% CI。目标规范值 micro-F1 ≥91%，VLM 触发率 ≤15%（n=30 时最多 4 张触发）。真实图片发往外部 LLM/VLM 前还需逐批确认数据传输授权和必要裁剪范围；目前评测命令不会自动把 VLM 建议覆盖为正确证据，因此无真实图片时不得回填增益数字。

### 4. 证据查重：第四

目录：`data/eval/evidence-dedup/pairs.json` 与 `images/匿名编号.png|jpg|webp`。复制 `data/sample/evidence_dedup_formal_template.json`。至少 50 对独立真实脱敏样本，建议正负各 ≥25 对；各对使用独立来源图片，正例可在**同一对内**使用同一原图或其合法变体，但同一图片不得跨对反复出现。保留 exact、压缩、旋转、裁剪、截图、同一获奖事实不同图片等正例，以及同赛事不同人、同人不同届、同等级不同赛事等困难负例。

顶层需 `dataset_version`、`authorization_reference`、`independent_real_pairs=true`、`pairs`。每对需唯一 ASCII 匿名 `id`、人工标注的布尔 `duplicate`（是否同一获奖事实）、`real_source=true`、`redacted=true`。正例 `transformation` 取 `exact/compressed/rotated/cropped/screenshot/same_fact` 之一；负例 `hard_negative` 取 `same_event_different_person/same_person_different_year/same_level_different_event/other` 之一。两侧 `left/right` Evidence 各需 ASCII 匿名 `id`、`type=image`、安全相对 `path`、七个规范事实 `fields`；`phash` 由真实 CLI 从文件重算，不信任人工填写值。报告需 Recall ≥96%、Precision ≥95%，并列 F1、TP/FP/TN/FN、阈值及 Recall/Precision 的 Wilson 95% CI。报告中的原始字段值会被脱敏占位符替换；原图和标签只存本地。

## 其他发布阻塞

- **52 人回测**：`data/eval/backtest-2025-2026/claims.xlsx`、`totals.xlsx`、逐项 `items.xlsx`，完整 52 名不同匿名学生，逐项最终计入分值、差异归因、规则版本与同口径端到端计时。无业务裁决时仅称“与人工历史结果的差异分析”，不得称准确率。模板由 `export-backtest-template` 从真实脱敏申报生成。
- **真实用户试用**：至少有真实独立用户与实际任务记录；`data/sample/user_trial_template.json` 只提供空协议。每位用户随机令牌、授权引用、知情同意版本、入口/任务、起止时间、完成/核验/人工复核状态、问题码及可选满意度。不得写姓名、学号、申报文本或图片路径。
- **质量/候选冻结**：本阶段变更审查、全量自动化与真实入口复测后仍保持未提交；`quality` 门禁只有在用户明确要求提交并按提交后的 HEAD 重跑时才能真正解除。

## 下一步命令（PowerShell）

```powershell
New-Item -ItemType Directory -Force data/eval/complex-pdf, data/eval/rule-extraction, data/eval/certificate-fields, data/eval/evidence-dedup
Copy-Item data/sample/complex_pdf_regression_template.json data/eval/complex-pdf/dataset.json
Copy-Item data/sample/rule_extraction_template.json data/eval/rule-extraction/dataset.json
Copy-Item data/sample/certificate_fields_formal_template.jsonl data/eval/certificate-fields/labels.jsonl
Copy-Item data/sample/evidence_dedup_formal_template.json data/eval/evidence-dedup/pairs.json

# 先人工填写授权、脱敏、标注与真实文件，再预检；数据不足时预期退出码 2。
.venv\Scripts\scoreproof.exe audit-formal-data data/eval --out reports/formal-data-inventory-v1.json

# 只执行预检显示 eligible 的第一个门禁；正式失败时保留报告与退出码 2。
.venv\Scripts\scoreproof.exe eval-complex-pdf data/eval/complex-pdf/dataset.json --base-dir data/eval/complex-pdf --out reports/complex-pdf-regression-v1.json
.venv\Scripts\scoreproof.exe eval-rule-extraction data/eval/rule-extraction/dataset.json --out reports/rule-extraction-formal-v1.json
.venv\Scripts\scoreproof.exe eval-certificate-fields data/eval/certificate-fields/labels.jsonl --call-vlm --vlm-provider qwen-vl-plus --out reports/certificate-fields-formal-v1.json
.venv\Scripts\scoreproof.exe eval-evidence-dedup data/eval/evidence-dedup/pairs.json --out reports/evidence-dedup-formal-v1.json
```

以上四个评测命令按优先级**只运行具备数据条件的一项**，不可用空模板或合成集生成同名正式报告。当前没有任何一项具备数据条件，故未运行正式评测、未解除业务门禁。
