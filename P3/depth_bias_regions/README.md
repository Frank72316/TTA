# P3 · 四视图收益的像素级分解

独立目录的 CPU 实验。读取 depth_adaptive_budget 的 512 个原始 FP32 对齐预测、冻结帧与场景 ID，以及 SHA-256 已核对的 NYU Depth V2 数据包。为旧模型 Depth Anything V2 Metric Indoor Small 的同一快照。每帧比较 O/F 两视图与 O/F/L/R 四视图；无新前向。缓存已逆对齐，直接裁真值到几何共同区，避免再次翻转或位移。

从仓库根目录复算已发布的帧级统计：
```sh
python -B P3/depth_bias_regions/analyze.py
```

从 Asus 像素缓存重建（Python 3、NumPy、Pillow、SciPy；绘图另需 Matplotlib）：
```sh
D=/home/shenc/tta_depth_adaptive/data
S=P3/depth_bias_regions
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
"$PY" "$S/measure.py" --zip "$D/nyu-depth-png.zip" --cache "$D/pred_cache"
"$PY" "$S/analyze.py"
"$PY" "$S/plot.py" --zip "$D/nyu-depth-png.zip" --cache "$D/pred_cache"
```

真值范围 (0,10] m。先用共同有效区上的实际预测均值差构造 p_shift；四邻有效且任一邻居的绝对 log 深度差大于 log(1.1) 时计跳变种子，在原分辨率按半径 3 像素圆盘膨胀。四邻不全的中心像素单列，其中包括共同视野边缘。所有分区的误差都使用同一个全帧位移，带权相加时覆盖每一个有效像素。场景 bootstrap 按 95 簇有放回重采样 10,000 次，每次带上所有帧，用副本真实帧数做分母。原实验按 FP32 单视图误差转 FP64 聚合，此处保持该口径并用先聚合 FP64 预测再减真值交叉验证。逐帧空间收益两种精度路径相差最多 2.64×10⁻¹⁰ m²。两个冒烟帧通过分解和区域对账，人造常量位移的空间项为零。机器、数据、耗时与代码哈希见 [环境记录](environment.json)。[冻结协议](protocol.md) · [汇总](summary.json) · [报告](report.md)
