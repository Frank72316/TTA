# P2 · 固定三次评分预算的增强互补性

[中文报告](report.md) · [策略对比图](figures/strategies.png) · [策略汇总](strategy_summary.csv) · [混合与单类的配对比较](mix_comparisons.csv)

从仓库根目录进入 `P2`。本实验读取 P1/baseline 的冻结抽样清单、验证集图像及 P0 分数，使用其公开 HPS 权重和已验证的推理环境。研究对象为原始提示词下人对原图对的偏好。模型在生成的变换视图上评分，策略最后仍在两张原图间选择。

## 输出

| 文件 | 用途 |
|---|---|
| [scores.csv](scores.csv) | 128 张图各 17 个版本，共 2,176 条唯一原始分数。含 P0、有效 token 哈希、原图路径、增强参数、像素哈希、最终输入尺寸、相对原图的模型输入变化、原生 dtype 和评分来源 |
| cache/ | 每张图一个原子写入的 JSON 断点，包括该图全部 17 条分数和协议哈希 |
| [combination_results.csv](combination_results.csv) | 每对每个具体参数组合的三项均值、分差、选择与人工对错。共 7,680 个三次决策和 64 条原图参考 |
| [pair_strategy_results.csv](pair_strategy_results.csv) | 每对每策略的组合数、选对/选错/平局的计数和概率、分差范围。原图判错时 correction_rate 为纠正概率，原图判对时 harm_rate 为改错概率，其余不适用处留空 |
| [strategy_summary.csv](strategy_summary.csv) | 原图参考与 10 种策略的期望计数、严格准确率、全部九类转移、相对原图的配对差和区间 |
| [mix_comparisons.csv](mix_comparisons.csv) | 六种混合各对两个组成单类及较强单类的严格准确率差和 95% 配对 bootstrap 区间，单位为百分点 |
| [score_shifts.csv](score_shifts.csv) | 四种增强族及各个参数相对原图的单图评分位移与图对分差变化 |
| [mechanism.csv](mechanism.csv)、[cases.csv](cases.csv) | 发生纠正和改错的图对、独有事件的几种口径、案例实际版本与分数。集合为空用“—”，重叠不可判定用 NA |
| [environment.json](environment.json)、[runs.json](runs.json)、[验收日志](logs/verification.txt) | 权重、软件、源代码与验证集校验和、实际 GPU、分阶段耗时和缓存命中数 |

识别图对时按字符串读取 pair_id，以保留 003、008 等前导零。分差均为 A 减 B。同分视为独立结果，严格选对率分母包含同分图对。

## 协议与预设

1. 用冻结的 64 条原始 P0、随机 A/B 和 `human_winner_P0`。先核对 rank 较小者为赢家，并从旧 P0 原始分数独立复现 43/64。保留原文首尾空格；有效输入由官方 tokenizer 决定。
2. O 为 RGB 原图。B090/B095/B105/B110 和相应 C 版本的系数为 0.90/0.95/1.05/1.10。J85/J89/J93/J97 在内存中用 Pillow 从同一 RGB 原图分别做 JPEG 编解码，固定 4:2:0、optimize=False、progressive=False。Tleft/Tright/Tup/Tdown 分别平移对应边长的 3%，正位移表示画面右移或下移；取最近整像素，半数向上。反射填充采用 NumPy reflect 边界，末端像素不重复。使用各图自己的宽高将同一相对位移换算为像素，本轮实际 dx/dy 已存入 scores.csv。
3. 对每个版本均调用相同的官方验证预处理。实际为 RGB、张量化、ResizeMaxSize 等比缩放并补边、归一化。预览为最终归一化张量反归一化后截取到显示范围的 224×224 图像。末行专门放大 JPEG 残差。input_mean_abs_RGB255 是最终张量反归一化后相对原图的通道平均绝对差乘 255，统计时保留 bicubic 可能产生的微小越界值。尺度用于描述输入差异。
4. 单类策略从四个版本中无放回取两个，不计次序，共六种。混合策略从每个族各取一个，共十六种。每张图用 O 和两个指定版本的三个分数等权平均。候选两侧共用同一组参数；先对每个版本组合独立比较 A/B，再对判定等权平均。浮点推理输出可精确提升到 float64，比较三项同分母之和再计算展示分差。
5. BC、BJ、BT 的两两单类对照为主要比较。CJ、CT、JT 为探索性比较。用 NumPy default_rng(42) 进行 10,000 次完整图对 bootstrap。每个重采样向量在各策略和参数上共用；“相对较强者”在每次重采样中重新取两个组成单类准确率的最大值。95% 区间采用线性插值百分位法。只有 64 个图对作为独立单位。
6. 判断某图对被“某策略纠正/改错”要求至少有一种版本产生该事件；另报告发生概率和期望计数。报告分别检验相对组成单类、全部四种单类及其余全部策略的独有事件。案例优先展示排他性最强的真实事件；同等级依次比较出现频率、主要比较优先、图对 ID 和策略名。具体组合取该事件分差最接近其中位数的一种，按版本名处理并列。这个规则只用于展示。

## 运行

GPU 使用 P1/baseline/.venv，Python 3.10.20、Torch 2.5.1+cu124、torchvision 0.20.1+cu124、Pillow 12.2.0、libjpeg-turbo 3.1.4.1。模型为 HPS v2.1，官方调用显式传入 `hps_version="v2.1"`；使用 eval、inference_mode 和官方 CUDA AMP。原生输出 float16，无损转成 Python float 保存。系统 Python 3.12 的 NumPy、Matplotlib、Pillow 用于独立的离线步骤。

```bash
cd P2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/runtime/mpl"
GPY=../P1/baseline/.venv/bin/python
CPU=/usr/bin/python3
mkdir -p logs
# 首次建立新运行的输入保护基线；已有文件时沿用首次快照。
test -f logs/pilot-before.sha256 || sha256sum ../P1/baseline/*.csv ../P1/baseline/*.json ../P1/baseline/*.py ../P1/baseline/*.md > logs/pilot-before.sha256
"$GPY" -B study.py
"$GPY" -Bu score_p2.py --limit 2
"$GPY" -B -c 'from study import *; r=[x for x in read_csv(ROOT/"scores.csv") if x["pair_id"] in ("001","002")]; assert len(r)==68 and all(float(x["input_mean_abs_RGB255"])>0 for x in r if x["view"]!="O")'
"$GPY" -Bu score_p2.py --limit 64
"$CPU" -B analyze_p2.py
"$CPU" -B display_p2.py
"$CPU" -B make_report.py
"$CPU" -B verify_p2.py
```

从现成缓存重新计算策略和区间只需运行 analyze_p2.py。display_p2.py 为案例示意另读原图 ZIP，不进行模型评分。已附案例图和 cases.csv 时，可直接运行 make_report.py 重建本批实验报告。

## 缓存与复验

旧 P0 的原图与 B110 是仅有的旧分数候选。先验证原始图像 ZIP、P0 清单、Pillow/Torch 版本、HPS/CLIP 权重、官方模型/预处理/tokenizer 代码与词表、驱动、返回精度和旧评分脚本。随后对 001、002 的全部 68 条重新推理，其中 8 条可比旧分数逐项精确匹配。其余图中 O、B110 共 248 条读自旧缓存，其他版本全部实际推理。64 对阶段续用本轮冒烟的 68 条记录。最终唯一记录 = 1,928 条新增推理 + 248 条旧评分。已完成输入的像素和文本身份保存在每条记录内。

评分中断时重跑原 --limit 命令；已提交的逐图断点会校验协议哈希并复用。全新复验可在同一父目录另建目录，复制本轮 Python 脚本再执行上列命令，保留原始案例文档和缓存。在新机器上需先按 P1/baseline/README.md 恢复原环境、权重和公开验证集。缓存兼容性检查绑定本次旧环境与设备描述。

