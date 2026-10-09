# 阶段 7.11：公开复杂 PDF 候选来源

核查日期：2026-10-09。以下仅是从高校官方站点在线核查过 PDF 类型和页数的**候选来源**，尚未取得本地原始文件，也未人工确认多栏、跨页表格或扫描版式。不得计入 `data/eval/complex-pdf/` 的正式样本量。

| 来源 | 在线确认 | 待核查事项 |
|---|---|---|
| [上海交通大学机械与动力工程学院 2025 年细则](https://me.sjtu.edu.cn/xsgz/upload/ueditor/file/20250824/2025%E5%B9%B4%E6%9C%AC%E7%A7%91%E7%94%9F%E7%BB%BC%E5%90%88%E7%B4%A0%E8%B4%A8%E6%B5%8B%E8%AF%84%E7%BB%86%E5%88%99.pdf) | PDF，17 页 | 原件 SHA-256、版式分类、人工金标、隐私信息 |
| [华南理工大学机械与汽车工程学院 2025 年细则](https://www2.scut.edu.cn/_upload/article/files/d7/57/81d4873d47ef9be5155bb07375d2/47a33b2f-2a73-4364-ac31-fb47f9c7f799.pdf) | PDF，8 页 | 同上 |
| [华南理工大学化学与化工学院 2025 年细则](https://www2.scut.edu.cn/_upload/article/files/69/43/229ac54c4adeb63e5c47bec6124a/f65fce75-966e-475c-bcdd-30a0c13c7798.pdf) | PDF，22 页；在线文本显示表格跨页候选 | 核对原件版面和表格人工金标 |
| [中国农业大学植物保护学院测评细则](https://cpp.cau.edu.cn/module/download/downfile.jsp?classid=0&filename=324cb8016de74f62b2c0eb1a525e60cf.pdf) | PDF，11 页；在线文本显示第 8～9 页附近有表格 | 核对原件版面和表格人工金标 |
| [中国石油大学（北京）克拉玛依校区 2026 年细则](https://www.cupk.edu.cn/wlxy/upload/resources/file/2025/12/26/110067.pdf) | PDF，14 页 | 原件 SHA-256、版式分类、人工金标、隐私信息 |

本环境对官网 PDF 的直接下载返回 `EACCES`，因此没有将在线文本或截图重新制成 PDF。在线资料不能代替独立原始文件的本地 Hash、去重和真实 CLI 评测。获取原件后，放入 `data/eval/complex-pdf/documents/`，按 `reports/phase-7-10-data-requirements.md` 标注；先运行 `scoreproof audit-formal-data`，再对足量且合格的数据运行正式回归。扫描件类别目前没有经原件核实的候选，仍需独立收集。

工程回归另在 `.tmp/stage7-11-pdf-content-audit/` 对仓库已有公开真实 PDF 进行仅改写元数据的派生测试：原件 260526 字节，改写后 312498 字节，字节 SHA-256 不同；已安装 `scoreproof.exe audit-formal-data` 将二者按页面内容识别为 **1** 份独立文档，报告 `formal_gate_eligible=false`，退出码 2。该派生副本仅用于去重回归，绝不计入正式集。低分辨率页面内容指纹可能把视觉上极其相似但真实独立的文件保守地标为重复；人工复核仍是正式入集的必要步骤。
