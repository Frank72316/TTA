# P3 · 三次深度推理的独立场景复验

[一页中文报告](report.md) · [冻结协议](protocol.md) · [逐帧结果](frame_results.csv) · [逐区域对结果](region_pairs.csv) · [汇总与区间](summary.csv)

### 样本与来源

- [全 654 帧官方场景映射](official_test_scenes.csv)从 NYU labeled MAT 的 scenes 引用读取；归档帧名索引 0 对应 MATLAB 索引 1。全部验证 JPEG 的编号加一后与 splits.mat 的 testNdxs 集合相同。[官方场景元数据](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) · [官方测试索引](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat)
- [旧帧逐项来源](history.csv)扫描 8a8d0cf 下的结构化实验产物，24 个文件共引用 192 个不同的 NYU 帧。额外核对仓库源码及说明中出现的 NYU 实验入口。来自同一拍摄序列的帧一同排除。[排除场景及其全部帧](excluded_scenes.csv) · [冻结的新帧及场景](frames.csv) · [冻结哈希](freeze.json)
- [预测实现](../depth_translation_pairing/run.py)固定为 3e0ea7c；[取窗与配方计算](../depth_order_pairing/order_data.py)、[帧内估计与场景重采样](../depth_order_pairing/summarize.py)固定为 8a8d0cf。本轮直接调用这些函数。[精确提交与源文件 SHA-256](sources.json) · [开发集从逐区域数据复算的结果](development_replay.csv)

### 预算与产物

Asus 物理 3 号 RTX 4090，按 UUID 绑定。Depth Anything V2 Metric Indoor Small 使用 revision `8078d68a9c75a972131914f6afd0c1723be0da7f`；模型前向 FP32，关闭 TF32。O 和四个平移视图都从新帧生成。主分析采用全高、原图横坐标 [26,614) 的共同视野，真值 (0,10] m，每窗至少 922 个有效像素。区域评价用 float64。每套配方独立评分，LL、RR 与 LR、RL 各自在策略内等概率平均。G 在同向误差较大时为正，A 在反向正确率较高时为正；与 O 的差同样规定正值为反向较好。

每帧离线生成 5 个唯一预测，218 帧共 **1,090 次实际前向**。前两帧冒烟做 10 次，续跑新增 1,080 次，命中冒烟的 10 个视图。两阶段墙钟合计 40.66 秒，包含数据和模型核验、装载与缓存写入，不含后续 CPU 区域分析。三次策略单独部署整批为 654 次前向；只用原图为 218 次。[环境与权重身份](environment.json) · [运行记录](runs.json) · [冒烟核验](smoke.json)

约 1.2 GiB 像素预测在 Asus：`/home/shenc/tta_depth_adaptive/P3/depth_pairing_scene_holdout/predictions/<frame_id>.npz`。每个文件含 5 张已逆对齐的 FP32 图（480×588）。[逐文件 SHA-256](prediction_files.csv) · [逐视图输入张量、对齐预测哈希及前向时间](view_audit.csv)。[逐窗均深与坐标](windows.csv)与[逐区域对](region_pairs.csv)可用 frame_id 和窗编号相连。每个区域对的 d、x、q、平方误差、判定和乘积核验均有保留。独立复核覆盖 130,800 个配方判定，且直接按抽中场景展开全部帧，复算四项差值的区间。[分割复核](split_replay.json) · [指标复核](replay.json) · [完整性检查](validation.json)

### 只读发布的表格复算

从仓库根目录执行。需要 Python 3.10+、NumPy、Pillow。输出到独立目录。重复核验场景补集、区域对清单、帧内统计、纠正/改错与配对区间。开发结果可单独执行 development 子命令；该命令在实验目录写 development_replay.csv。

```sh
S=P3/depth_pairing_scene_holdout
OUT=$(mktemp -d)
cp "$S/windows.csv" "$S/region_pairs.csv" "$OUT/"
python -B "$S/replay_cached.py" --out "$OUT"
python -B "$S/evaluate_holdout.py" development
```

### 在 Asus 从原始数据重建

使用已冻结清单可直接从下面的 development 步骤开始。重新提取官方场景映射需 h5py、网络或本地完整 MAT；划分核验需 SciPy，推理需项目原 PyTorch 与 Transformers 环境。历史帧盘点需有 Git 历史的仓库。官方场景元数据本次在本机提取，划分、推理与区域评价在 Asus 执行。

```sh
S=P3/depth_pairing_scene_holdout
DATA=/home/shenc/tta_depth_adaptive/data
ZIP="$DATA/nyu-depth-png.zip"
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
OUT=$(mktemp -d)
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
# 在有 Git 历史的完整仓库盘点；场景元数据提取可用带 h5py 的 Python
python -B "$S/inventory.py"
python -B "$S/fetch_scenes.py" --archive "$ZIP"
# 已有官方 MAT 也可给 fetch_scenes.py 追加 --mat /path/to/nyu_depth_v2_labeled.mat
"$PY" "$S/freeze_holdout.py" --archive "$ZIP" --split "$DATA/splits.mat"
"$PY" "$S/evaluate_holdout.py" development
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/predict_holdout.py" --archive "$ZIP" --model "$DATA/model" --out "$OUT" --smoke
cat "$OUT/smoke.json"
"$PY" "$S/predict_holdout.py" --archive "$ZIP" --model "$DATA/model" --out "$OUT"
CUDA_VISIBLE_DEVICES="" "$PY" "$S/evaluate_holdout.py" holdout --archive "$ZIP" --out "$OUT"
CUDA_VISIBLE_DEVICES="" "$PY" "$S/replay_cached.py" --out "$OUT"
```

本轮实际 OUT 为 Asus 上本实验目录。可从原始像素缓存重跑末两条命令，也可只从仓库 CSV 重跑排序和统计。
