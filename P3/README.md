# P3 · 推理次数的边际收益受共同偏差限制

**状态：完成。** 已在单张 NVIDIA L4 上完成预算、幅度、留出策略、第二评分器和米制深度复验。原始评分、计时、深度误差矩与校验和均已归档，现有结果可在 CPU 上复算。[实验报告](report.md) · [预算曲线](results/budget_curves.png) · [冻结协议](protocol.md)

固定增强类型和幅度，比较每图 **1、2、4、8 次推理**的任务表现和耗时，再固定次数改变幅度。核心问题是多算一次还能消除多少误差，以及何时值得换一种增强。

方案提出估计共同偏差、单视角误差方差和视角间误差相关性，将预测的预算曲线与实测曲线对照，估计有效视角数及边际收益拐点，并在留出集比较不同采样策略。

## 从归档结果复算

从仓库根目录运行。分析需要 NumPy，图表另需 Matplotlib，读深度误差矩另需 Pillow。

```bash
OUT=P3/runtime/reproduced
mkdir -p "$OUT/depth"
for run in development-3 development-1.5 development-6 holdout-hps holdout-clip; do
  SRC="P3/results/$run"
  PAIRS=P2/translation/pairs_frozen.csv
  if [[ "$run" == holdout-* ]]; then
    PAIRS=P3/results/holdout/pairs.csv
  fi
  python -B P3/analyze.py --pairs "$PAIRS" --scores "$SRC/scores.csv" \
    --timings "$SRC/timings.csv" --environment "$SRC/environment.json" --out "$OUT/$run"
done
python -B P3/analyze_study.py --runs P3/results --out "$OUT/summary" \
  --legacy-scores P2/translation/scores.csv
cp P3/results/depth/depth_{moments,environment}.json "$OUT/depth/"
python -B P3/depth.py --out "$OUT/depth"
python -B P3/plot_budget.py --runs "$OUT" --depth "$OUT/depth" --out "$OUT/budget_curves.png"
```

## 缓存摸底

使用 [P2 平移复验](../P2/translation/report.md) 的 256 对原图与四个 3% 平移方向，评分来自 Keqing 的 RTX 4080 SUPER。每对图枚举所有等概率的方向子集，先按单个子集选赢家，再对各子集的结果取均值。按完整图对做 10,000 次配对 bootstrap。

| 每图评分次数 | 严格选对率 |
|---:|---:|
| 1 | 68.36% |
| 2 | 68.95% |
| 4 | 68.85% |

原图始终计入预算。四个平移方向最多支持每图五次评分。缓存摸底没有本轮实测耗时，完整预算比较使用同一台 GPU 实例的新评分和计时。

从 1 到 2 次增加 0.586 个百分点，95% 配对区间为 [−1.660, +2.832]。从 2 到 4 次变化 −0.098 个百分点，区间为 [−1.270, +0.977]。两段的区间均跨零。

从仓库根目录运行，CPU 环境需要 NumPy。

```bash
python -B P3/analyze.py --out P3/runtime/p2-cache
```

## 单 GPU 复测

首轮复用 P2 冻结的 256 对原图和原始提示词，从头评分。原图加八个相对位移幅度为 3% 的方向，四个轴向沿用 P2 的整数位移和反射填充，四个斜向在以图像宽高归一化的坐标系中保持相同幅度。A/B 使用相同方向。预算 K 包括原图和 K−1 个不同方向，K=1、2、4、8 分别枚举 1、8、56、8 个子集。完整视图库每图记录九个分数，预算曲线仍只统计相应预算下会用到的视图。

计时按图对使用可复现的随机方向顺序，记录同一推理在每个预算点的累计墙钟耗时。时间包含两张原图的 ZIP 读图、增强、预处理、一次文本编码、逐图推理与分数回传。模型加载、预热、输入哈希和文件写出在计时区间外。准确率对所有方向子集等权平均，每个图对只实测一个预算嵌套方向顺序的耗时。

在 GPU 实例上按 [P1 的 CUDA 环境说明](../P1/baseline/README.md#在新的-linux-cuda-环境从头复现) 安装依赖与固定版本的 HPS 源码。Ubuntu 22.04 上需要 `python3.10-venv` 和 `python3-tk`。HPS 上游 wheel 缺少 tokenizer 数据文件。保留相同提交的源码和词表，并将源码目录加入 `PYTHONPATH`。从 P1/baseline 运行 `fetch.py` 下载并校验原图 ZIP 和权重。下面以 L4 为例，使用其他 GPU 时传入 `nvidia-smi` 显示的实际型号和所在可用区。

```bash
REV=866735ecaae999fa714bd9edfa05aa2672669ee3
mkdir -p P1/baseline/vendor
curl -fsSL "https://codeload.github.com/tgxs002/HPSv2/tar.gz/$REV" |
  tar -xz -C P1/baseline/vendor
export PYTHONPATH="$PWD/P1/baseline/vendor/HPSv2-$REV"
bash P3/run_gpu.sh asia-southeast1-c "NVIDIA L4"
```

- `environment.json` 记录 GPU、驱动、操作系统、CUDA、PyTorch、HPS、模型权重和输入文件校验和。评分器用预热图对逐项对照官方 HPS v2.1 接口。
- `budget_summary.csv` 汇报严格选对率、纠正、改错、同分与延迟均值、P50、P95。延迟单位为每对图的毫秒。
- `marginal_gains.csv` 汇报相邻主预算和相对于原图基线的选对率差、配对区间及每额外一次单图评分的百分点收益。
- `pair_results.csv`、`scores.csv`、`timings.csv` 和 `analysis.json` 支持逐图复核与复算。

完整实验的样本、幅度、误差模型和留出规则见[冻结协议](protocol.md)。脚本依次运行开发集 3%、1.5%、6% 平移、留出集 HPS、留出集 CLIP。下列命令生成幅度对照、Brier 预算曲线、同计算预算的采样策略对照，以及旧缓存与 L4 在相同视图上的差异。只做离线复算可用任意装有 NumPy 的 Python。

```bash
python -B P3/analyze_study.py --runs P3/runtime/gpu --out P3/runtime/gpu/summary \
  --legacy-scores P2/translation/scores.csv
```

## 米制深度复验

使用 [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) 的 RGB 与米制深度，读取[已打包的 Eigen 测试划分](https://docs.ultralytics.com/datasets/depth/nyu-depth-v2)。Depth Anything V2 Metric Indoor Small 模型权重固定至脚本中的版本。每次基于对齐后的整帧共同可见区域测算，误差矩独立保存，可在 CPU 上复算预算曲线，无需保存或重跑预测图。

```bash
mkdir -p P3/runtime/depth
curl -fL https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth-png.zip \
  -o P3/runtime/depth/nyu-depth-png.zip
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 P1/baseline/.venv/bin/python -Bu P3/depth.py \
  --archive P3/runtime/depth/nyu-depth-png.zip --out P3/runtime/depth/results
# 取回 depth_moments.json 与 depth_environment.json 后只做 CPU 复算
python -B P3/depth.py --out P3/runtime/depth/results
```
