# TTA 项目交接材料

P3 完成后的接手入口为 [P3 实验交接报告](P3交接报告.md)，包含单卡实验结论、归档文件和 CPU 复查路径。

直接发送 `TTA阶段交接报告.pdf` 即可。文中三个统计图均为 PDF 内部的文字和矢量图形；四个案例有 28 张不同的 224 × 224 视图，封面再次引用 003 的两张原图。正文和图表布局可在 [compose.py](compose.py) 中编辑，打印样式在 [print.css](print.css)。

## 重建 PDF

进入本目录，运行 `node build_pdf.cjs`。脚本需要 Python 3、Node.js、Playwright、本机 Chromium 或 Chrome，以及微软雅黑或 Noto CJK 字体。可用 `PYTHON` 指定 Python 解释器，用 `CHROME_PATH` 指定浏览器。已有这些依赖时，编辑和重建均可离线完成。本机使用的 PowerShell 命令如下。

```powershell
$env:NODE_PATH='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1123-915-0_26.1123.915.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules'
node build_pdf.cjs
```

从缓存更新数据时，在本目录依次运行以下命令。排版数据在内存中直接传给 PDF 打印程序。

```bash
python -B audit.py
node build_pdf.cjs
python -B verify_pdf.py
```

`audit.py` 需要 NumPy 与仓库中的 P1、P2 实验目录。它从旧、新原始评分重算分差、共同分量、严格跨提示词期望、主要策略的每套搭配、按图对配对区间及所引案例，并与原逐对 CSV 对账。来源文件摘要记录在 `audit.json`。`compose.py` 读 `audit.json`、`assets/`、`case_inputs.csv` 和 `print.css`。PDF 校验需要 pypdf、Pillow、NumPy 和 Poppler 的 pdftoppm。它只把 PDF 复制到临时空目录，再逐页渲染、核验字体嵌入、核对图像绘制顺序和像素。8 页的 PNG、文件校验值及每页资源清单见 `qa/`。

## 表、图与案例的来源

| 页码 | 引用材料 | 实际输入 |
|---|---|---|
| 1 | 003 的 A/B 原图、同时涨分的示例 | `P1/baseline/pairs.csv`、`scores.csv` 的 P0 × L0/L1；图片沿用 `P2/translation/old_cases/assets/` |
| 2 | 六阶段设计、005/053 的两分均值 | 各目录的 README 和 report；`P1/prompts/old_prompt_audit.csv`、`P1/baseline/pair_results.csv` 与 pilot 原始分数 |
| 3 | 共同分量、图 1 随机重配参照 | `P1/prompts/scores.csv`、`P1/decomposition/pair_decomposition.csv`、`summary.csv`；`P1/pairing/summary.csv` 与 `null_distribution.csv` 中 all × derangement |
| 4 | 图 2 纠正与改错、准确率与平局表、图 3 原始分差四分组 | `P2/translation/scores.csv`、`transitions.csv`、`strategy_summary.csv`、`endpoint.json`、`margin_groups.csv` 与 `pair_contributions.csv` |
| 5 | 003、008 的三列图像及已观察到的搭配 | `P1/baseline/pairs.csv`、`P2/scores.csv`、`combination_results.csv` 和 `P2/translation/old_cases/assets/` |
| 6–7 | N032、N141 的原图、三个版本、带符号分差、方向表 | `P2/translation/pairs_frozen.csv`、`scores.csv`、`combination_results.csv`、`direction_counts.csv`；冻结路径标识的验证集原始图像 |
| 8 | 复查路径、命令、结论边界与后续问题 | 各实验 README；短提示词实验及 P2 两批样本的汇总和逐对缓存 |

## 案例输入核验

旧例优先复用以前渲染的 O/T PNG；新例与旧例缺少的亮度图均从原始 WebP 重建。使用原 P2 的 `original`、`make_view`、`final_rgb` 和同一官方验证变换，逐一核读原始图像路径、P0、A/B 映射以及增强后 RGB 像素 SHA-256。每个显示文件的来源与 SHA-256 见 `case_inputs.csv`。如需在已有验证集缓存的服务器重建图片，可将本轮目录放在旧实验目录旁运行下列命令。此步骤只做输入预处理，全程 CPU。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 ../P1/baseline/.venv/bin/python -B case_inputs.py
```

