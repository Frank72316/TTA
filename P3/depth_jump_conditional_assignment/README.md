# 遮挡深度落差的条件重配

[中文报告](report.md) · [预先写定的计算规则](protocol.md) · [分解图](figures/decomposition.png)

以提交 `78a419dad2f272da083a0693611d57ef81c614a7` 的 P3/depth_jump_assignment 为参照，从同一实验树中的 depth_occlusion_context 与 depth_occlusion_tta 只读缓存重建 B（默认全图）、R（真实周边裁窗融合）和 T（原四窗融合）。全部 50 个场景直接进入分析。Asus 主机 ASUS4090x8；Python 3.10.20，NumPy 1.26.4，复用原项目环境。隐藏 CUDA，模型训练和新增前向均为 0。原协议、预测和 GT 缓存保持只读。

## 复现命令

以下在 Asus 的 `/home/shenc/tta_depth_adaptive/P3/depth_jump_conditional_assignment` 目录执行。代码通过相邻实验目录定位旧脚本与冻结表格；本轮场内数组写入 EDGE_CACHE，公开小型表格写入本目录的 results/。

```bash
cd /home/shenc/tta_depth_adaptive/P3/depth_jump_conditional_assignment
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OCC_DATA=/home/data/users/shenc/tta_occlusion_data
export CTX_DATA=/home/data/users/shenc/tta_occlusion_context
export EDGE_CACHE=/home/data/users/shenc/tta_edge_conditional_assignment_v1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
sha256sum -c source_sha256.txt
"$PY" extract.py
"$PY" conditional.py --workers 4
"$PY" verify.py
"$PY" plot.py
```

extract.py 依次核对固定提交的源文件、50 个布景的预测及界面缓存、2014 的 23 份共享预测与推理元数据。重新得到两个方法、每个场景的旧全乱和真实 MAE，以及两批的总位置收益与区间，均通过后才处理条件分档。提取和旧结果复核耗时约 13 秒，计算可重配组及随机大小对照约 2.2 秒，独立复核约 0.9 秒。重画 PNG 只需 CSV、NumPy、Matplotlib 与可用的中文字体。

## 四个 MAE 如何复算

每条边固定同一组原生像素，每侧三个位置先取中位预测米数，再用 float64 做差、换算为毫米。B 预测的两侧差为 q；分档量 |q| 在交换两侧时不变。横向扫描 axis=0，纵向扫描 axis=1，没有按梯度正负或前后景朝向再分。对每个方向单独作 20/40/60/80 线性分位切分，重复切点合并，数值正好在切点上入较低档。分档函数只读取轴号与 |q|。2006/Cloth1 的横向无合格界面，全部纵向边仍纳入。实际所有非空方向均有四个不同切点；最小的非空档在 2014 有 133 条边，在 2005/06 有 2 条。

在含 n 条边的组内，每条接收边的 e 与所有供应修正 u 等概率配对。允许自配。把 u 排序，按 −e 的插入位置用前缀和分别累加正、负两侧，精确计算 n² 项的平均绝对值。跨组按 n 加权，随后逐场、簇内等权、再跨簇等权。采用旧实验种子 2026 和批次名、簇名排序，配对 10,000 次双侧百分位区间。比较的最小独立重采样单位是布景簇。

| 批次与方法 | A全乱（mm） | A方向（mm） | A条件（mm） | A真实（mm） |
|---|---:|---:|---:|---:|
| 2014 R | 355.05 | 354.36 | 351.38 | 337.77 |
| 2014 T | 359.98 | 359.85 | 358.07 | 339.96 |
| 2005/06 R | 254.17 | 254.17 | 253.65 | 250.14 |
| 2005/06 T | 271.41 | 271.37 | 271.24 | 268.31 |

### 同人数的随机档位

每场每轴保持真实档位标签的数量，用种子 2026、冻结场景序号与轴号构造独立随机流。每次打乱的是边到档位的标签；同一条边的 e 和 u 继续对应，R/T 共用该次标签。在随机档内仍按前缀和求供给修正的精确均值。100 次在场内取平均后才做场景簇推断。可用解析期望复核随机对照。当一条轴有 n>1 条边和 k 个非空档，平均自配概率为 k/n；对均匀随机标签，其期望误差为 A轴 + (k−1)/(n−1) × (A实轴−A轴)。此式来自同轴非自配组合的对称性。长程期望与 100 次估计见 size_control_summary.csv；其中随机重复的分位区间条件于固定场景，而 summary.csv 中的区间按场景簇重采样。

## 输出与验收

| 复核内容 | 文件 |
|---|---|
| 逐场四种 MAE、三项收益、对照、恒等式残差 | [per_scene.csv](results/per_scene.csv) · [summary.csv](results/summary.csv) |
| 各轴原始/去重切点与覆盖；逐档开闭区间、实际观测范围、占比、单元素标记 | [axes.csv](results/axes.csv) · [bins.csv](results/bins.csv) · [coverage.csv](results/coverage.csv) |
| 100 次重新分档的每场每方法结果，以及固定场景下的汇总和理论值 | [size_control_repeats.csv](results/size_control_repeats.csv) · [size_control_summary.csv](results/size_control_summary.csv) |
| 冻结清单顺序、场内数组哈希、旧总收益复现 | [input_index.csv](results/input_index.csv) · [replay.csv](results/replay.csv) · [replay_check.json](results/replay_check.json) |
| 主流程核验和独立核验 | [assignment_check.json](results/assignment_check.json) · [verification.json](results/verification.json) |

各场景的轴、B 预测落差绝对值、基线残差及 R、T 修正同步保存在 `$EDGE_CACHE/<year>/<id>.npz`。每场两个方法的三项分解均通过。独立代码用直接区间不等式重建全部 495 个档，另以经验分布函数积分核对前缀和，并对 18,452 个档内位置向同档所有对象直接枚举。最大前缀和与积分差为 6.83×10⁻¹³ mm。两批的占比和组数分母以布景与合格界面为准；100 次重排不增加独立样本数。
