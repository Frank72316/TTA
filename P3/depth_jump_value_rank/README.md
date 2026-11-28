# 低通核心的数值与边界排序

[中文报告](report.md) · [四格绝对 MAE](results/four_cells.csv) · [逐场 CSV](results/per_scene.csv) · [两条路径、交互和尺度敏感性的区间](results/summary.csv) · [分解图](decomposition.png)

## 输入、权重与单位

以 `Frank72316/TTA@79680d9963fb2957367ba82042fbd7dabe444bb7` 及其 Asus 缓存为起点。[冻结协议](protocol.md) 和 [来源哈希](sources.sha256) 固定边界、分档、分解与均值诊断。H/L 表示同一冻结 Depth Anything V2 Metric Indoor Small 在原生／低通核心上的既有四窗预测，两者共用全图 B。按原融合权重回到整图，每条边取远侧三点中位数减近侧三点中位数。主分析使用两组共用的原生尺度 `native_R`，敏感性使用两组共用的低通尺度 `low_core`。每场核验 B/H/L 缓存、边界、旧预测落差、方向、冻结分档边数及实际误差。[上轮缓存索引](../depth_jump_detail_intervention/results/cache_index.csv) · [原场景簇与清单](../depth_occlusion_context/scenes.csv)

每条边需要的修正是 GT 落差减 B 落差，供给的修正是裁窗落差减 B 落差。保留符号，用 float64 换算为 mm。按供给数值排序，接收边按指定条件排序；四格缩写中第一字母指定值，第二字母指定序。若 m 条接收边修正相同，每条边分别对这 m 个可接收值求绝对损失，再取均值。代码直接枚举并列段，用独立的秩区间和前缀和算法逐档核对全部四格。主口径 H 排序有 70 个并列段、涉及 140 条边，L 排序有 73 个、涉及 146 条边。均值诊断逐档平移原生修正量，保持边间差异。所有赋值仅使用预测。[逐档分解、均值与并列计数](results/per_bin.csv)

## 在 Asus 全量复现

主机为 Asus，Python 3.10.20、NumPy 1.26.4，沿用项目已有环境。下面命令读取全部缓存，在本目录写结果。首次全量计算用时见 [执行记录](results/checks.json)。无需启用 GPU 或下载权重。

```bash
set -e
cd /home/shenc/tta_depth_adaptive/P3/depth_jump_value_rank
export OCC_DATA=/home/data/users/shenc/tta_occlusion_data
export CTX_DATA=/home/data/users/shenc/tta_occlusion_context
export EDGE_CACHE=/home/data/users/shenc/tta_edge_conditional_assignment_v1
export DETAIL_DATA=/home/data/users/shenc/tta_jump_detail_intervention_v1
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
sha256sum -c sources.sha256
"$PY" run.py --workers 4
"$PY" verify.py
"$PY" plot.py
```

仅用仓库中的逐档 CSV 即可运行 `verify.py`，独立重建逐场四格、均值诊断、两条交换路径和所有簇区间。制图使用 Matplotlib 和中文字体。全量 `run.py` 需要上列 Asus 像素缓存，它先对账两个批次的 HH、LL 和总差（含区间），再计算交叉格。[旧结果复现](results/native_replay.csv)

## CSV 读法

| 指标列 | 单位与含义 |
|---|---|
| HH、LH、HL、LL、mean_only | 四格绝对 MAE 与只换档均值的 MAE，单位 mm，低为好 |
| D、V、S | 总优势、修正数值贡献、边界排序贡献，正值代表低通降低误差 |
| value_on_H_order、rank_with_L_values | 先换值 HH→LH，再换序 LH→LL |
| rank_with_H_values、value_on_L_order | 先换序 HH→HL，再换值 HL→LL |
| interaction | 低通排序下换值的收益，减去原生排序下换值的收益 |
| mean_gain、shape_gain_on_H_order、mean_to_LL | HH 减只换均值；只换均值减 LH；只换均值减 LL，均为 mm |

`per_scene.csv` 有 100 行（50 场 × 两套共用尺度）。`summary.csv` 先在每个场景内按边数求均值，再簇内场景等权、簇间等权；`lo/hi` 为 10,000 次配对场景簇 bootstrap 的双侧 95% 百分位区间，种子 2026。`positive/negative/zero` 单列场景计数。各路径用全精度算后再展示两位小数。
