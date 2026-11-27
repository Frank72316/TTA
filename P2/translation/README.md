# P2 独立复验 · 轻微平移与人工偏好排序

阅读 [中文结果](report.md) 或在浏览器直接打开 [五例旧样本](old_cases/index.html)。独立样本的主比较是 BT 与 BC 的严格选对率差。两种策略均对每张图使用原图分数及两个独立增强的分数。

## 目录与环境

从仓库根目录进入 `P2/translation`。脚本以自身目录解析路径，使用仓库根目录下的 `P1/baseline/` 和 `P2/`。旧项目作为只读输入。

- 推理使用已有的 `../../P1/baseline/.venv/bin/python`、HPS v2.1、ViT-H-14、CUDA 12.4、torch 2.5.1+cu124、torchvision 0.20.1、hpsv2 1.2.0、Pillow 12.2.0、NumPy 1.26.4。完整版本、源码指纹、官方预处理和两个权重 SHA-256 收录在 [environment.json](environment.json)。
- 抽样需要上述环境中的 pyarrow。CPU 分析使用服务器的系统 Python 3、NumPy、Pillow、Matplotlib 和 Noto CJK 字体。环境已就绪，无需联网下载。
- 主要输入为 `../../P1/baseline/pairs.csv`、`manifest.json`、`data/metadata-validation.parquet`、`data/*.zip`、`data/model-cache/`、`vendor/` 以及 `../scores.csv`、`environment.json`、`study.py`、`views.py`。用原始提示词 P0 原样评分。validation 数据 revision 为 `c493721e19e296eb615420036f2a2eed08412bb4`，HPS 权重 revision 为 `697403c78157020a1ae59d23f111aa58ced35b0a`。
- RGB 亮度与对比度使用 Pillow 0.90、0.95、1.05、1.10。平移的像素数按 `(3*边长+50)//100` 算出，先反射填充、再按整像素截取，无平移插值。官方管线将 RGB 转张量、以双三次插值把长边缩到 224、补边并归一化。案例 PNG 按归一化的通道均值和标准差逆变换后才量化到 8 位，仅供显示。每个增强从 RGB 原图单独生成。
- [protocol.md](protocol.md) 在抽样与独立样本推理前冻结，时间及哈希见 [freeze.json](freeze.json)。原始缓存均保留原生输出的完整精度。把每个三分均值的**选择**等概率平均，等分保持同分。

## 实际运行与复现

从 Windows 登录，再在服务器 bash 中运行。原始数据、权重与图像 ZIP 留在服务器，Windows 副本含离线分析结果、逐对检查点及完整案例页。

```powershell
ssh shenc@10.0.16.83
```

```bash
cd P2/translation
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
HPS=../../P1/baseline/.venv/bin/python
CPU=/usr/bin/python3
$HPS -B old_cases.py
$HPS -B freeze.py
$HPS -Bu score_translation.py --limit 2
$HPS -Bu score_translation.py --limit 256
$CPU -B analyze.py
$CPU -B figures.py
$CPU -B verify.py
$CPU -B report.py
```

从头重跑时，把本轮的 `.py`、`protocol.md` 和 `logs/prior_inputs.sha256` 放入新的同级目录，保留相邻旧输入。首次评分自动选择空闲单卡、验证本地权重、归档校验和及软件环境。冒烟的 52 条评分逐一与官方 `img_score.score(..., hps_version="v2.1")` 对照。在完整首次运行中，每对文本只编码一次；每个图像视图仍单独编码，模型处于 eval 状态。每对的 26 条记录原子写入 `cache/Nxxx.json`；若任务中断，重跑相同的 `score_translation.py --limit 256` 即可命中已完成图对。重复调用评分命令会追加运行日志。仅复算报告时从 `analyze.py` 开始，无需 GPU 或模型。

## 数据字典

| 文件 | 粒度与用途 |
|---|---|
| [pairs_frozen.csv](pairs_frozen.csv) | 每个独立提示词一行，保留原文、规范化文本、A/B 图像路径、名次和人工赢家 |
| [scores.csv](scores.csv) | 每图每视图一行，含有效 token 哈希、变换后 RGB 哈希、实际参数、原生 dtype 与 HPS 分数 |
| [single_view_shifts.csv](single_view_shifts.csv) | 各单一增强的原始 A−B 分差、增强后分差、位移及绝对值 |
| [combination_results.csv](combination_results.csv) | 每个图对与具体搭配的最终分差、选择、对错、改判及严格翻转 |
| [pair_strategy_results.csv](pair_strategy_results.csv) | 每对每策略的分母、选对/选错/同分数、严格翻转和改判数 |
| [pair_contributions.csv](pair_contributions.csv) | 主终点逐对贡献、分差组和各策略计数。贡献单位各有期望对数与百分点 |
| [strategy_summary.csv](strategy_summary.csv)、[transitions.csv](transitions.csv) | 原始搭配计数、百分比、期望计数，及原图对/错/同分三组的去向 |
| [endpoint.json](endpoint.json) | 整对配对 bootstrap 的主终点、95% 区间与按原图状态拆解的期望计数差 |
| [margin_groups.csv](margin_groups.csv)、[direction_counts.csv](direction_counts.csv) | 领先幅度四分组和原始同分组；逐方向的触发亮度、严格翻转、原图与单方向取均值时的选择 |
| [old_case_checks.csv](old_case_checks.csv)、[旧案例页](old_cases/index.html) | 旧 64 对里发生改判的五例，含各 BT 单元格和原图＋单平移结果 |
| [runs.json](runs.json)、[verification.json](verification.json) | 分阶段耗时、设备、检查点及官方对照次数；独立的整数精确判定、区间与完整性验证 |

所有 CSV 采用 UTF-8。新图对 ID 前缀为 N；旧案例使用三位数字。旧发现样本与独立复验使用各自的分母。「原图＋单方向平移」只用于方向诊断，主要策略比较始终固定每张图三次评分。
