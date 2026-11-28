# 裁窗深度修正的边界位置匹配

[一页中文报告](report.md) · [四种误差、边界分类与 UP 对照](details.md) · [逐场分解](results/edge_assignment_by_scene.csv)

本轮在 Asus 直接运行实验包原版 edge_cache.py。上游为 Frank72316/TTA 的 `94674659d5b4e62b9886e568ed9df1ac628bfe92`。zip 内附报告及 results/ 是公开表分析；本目录 results/ 全部来自本轮 Asus 的真实像素缓存。未运行 pilot，未训练，模型前向次数为零。程序仅用既有像素预测重建 B、T、R。GT 用于界面、分类和误差评价；UP、SHIFT、ROLL 构图时均不读取 GT。

## 实际运行

主机 ASUS4090x8，Ubuntu 22.04.5，Python 3.10.20、NumPy 1.26.4、SciPy 1.15.3。正式 CPU 程序在 2026-11-28 22:57:24–22:57:48（北京时间）执行，墙钟 23.41 s，峰值常驻内存 966,032 KiB。时间只包括包内全量程序；独立复算另行执行。原始脚本 SHA-256 为 `4c9275f5fd25764161acd83f50609122be7f21698191ad5228bd6ddcb824bd7d`，随附 zip 为 `1958acbb443d30dcb15b104bc4f5c59c715db135ba17f6f9c60d81f2c64444e5`。在正式读取前已核对 zip 校验、随包四份输入 CSV、上游提交中的代码和表格、远端同名文件。[上游文件哈希](source_sha256.txt)

以下为 Asus 上从解压目录执行的正式命令。原结果已存在，另一次重跑请将 OUT 改为新的空目录。仓库内的 edge_cache.py 与压缩包脚本字节相同，也可从本目录运行。

```bash
cd /home/shenc/tta_depth_adaptive/P3/depth_jump_assignment
sha256sum -c source_sha256.txt
cd package
sha256sum -c input_sha256.txt
export OCC_DATA=/home/data/users/shenc/tta_occlusion_data
export CTX_DATA=/home/data/users/shenc/tta_occlusion_context
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
CTX=/home/shenc/tta_depth_adaptive/P3/depth_occlusion_context
OUT=/home/data/users/shenc/tta_edge_model_results_v1
"$PY" edge_cache.py --context-dir "$CTX" --out "$OUT"
"$PY" audit.py --context-dir "$CTX" --out "$OUT"
```

当全部结果 CSV 已复制到仓库的 results/ 后，运行 `python make_details.py` 可从 CSV 重建附表。原始界面的位置、轴向和三点索引保留在上游缓存。两批均已用于形成假设。2014 的 188,235 条界面来自 23 场；2005/06 的 55,583 条来自 27 场、17 簇。每种方法在相同界面上先算场内均值、再对同簇的布景等权，最后簇等权。包内种子 2026，按批次名顺序使用同一随机数发生器，配对重复 10,000 次、双侧百分位区间。两个批次各自报告。

## 结果文件

| 要复核的内容 | 相对路径 |
|---|---|
| 每场 B、R、T、UP、DOWN、SHIFT、ROLL 的误差 | [scene_metrics.csv](results/scene_metrics.csv)；各方法在两批的绝对误差及区间见 [absolute_summary_mm.csv](results/absolute_summary_mm.csv) |
| A0、A均、A乱、A实，三项分解、净收益 | [逐场](results/edge_assignment_by_scene.csv) · [汇总及区间](results/edge_assignment_summary.csv) |
| 全图／UP／SHIFT／ROLL 的配对收益 | [paired_contrasts.csv](results/paired_contrasts.csv)，reference 减 method；有符号偏差的差值保留方向 |
| 七类界面的计数、比例和净收益贡献 | [逐场](results/scene_transitions.csv) · [簇汇总](results/transition_summary.csv) |
| 同一类由高估还是低估出发 | [逐场细分](results/sign_by_scene.csv) · [细分与区间](results/sign_summary.csv) · [真实边界与坐标](results/boundary_examples.csv) |
| 全量运行与独立验收 | [run.json](results/run.json) · [逐场数值复核](results/independent_by_scene.csv) · [独立核验汇总](results/independent_checks.json) |

全部误差与贡献以毫米计。案例对每一批、每种转移及起始误差方向，取冻结场景和界面顺序中第一条合格记录。例中轴 0/1 表示横/纵扫描，x/y 为原生图像交界左侧或上侧的零基坐标。

## 数值核验

- 主程序核对 50 份上轮预测、50 份界面缓存、23 份 2014 共享预测的哈希；额外核对 50 场的 RGB、真值、标定和上轮推理元数据。每条 GT 落差都用缓存的三近点、三远点独立排序取中位重建。
- 所有 50 场 × 3 个历史策略 × 8 个指标通过，最大差 4.55×10⁻¹³ mm。另用公开像素移动表还原 150 个 UP 的场级近/远/等权 MAE，最大差 2.28×10⁻¹³ mm。
- 主程序通过排序和前缀和精确计算场内重配。另由两组经验分布函数直接积分求独立抽样的期望绝对差，全部 100 个场景×方法的四项 MAE 最大差 3.42×10⁻¹³ mm；对分布于各场景的 3,100 条边，还逐一枚举了同场所有可匹配修正。
- 主程序的 100 个不等幅代价均非负。独立算法最小值为 −2.84×10⁻¹⁴ mm，在双精度误差范围内。每场分解恒等式和七类贡献加总通过，配对汇总及区间重算最大差 1.71×10⁻¹³ mm。
