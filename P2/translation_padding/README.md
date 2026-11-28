# P2 · 整数平移的补边干预

在仓库根目录执行。GPU 实跑使用 Asus 的 RTX 4090、冻结的 HPS v2.1 和历史单图单文本 CUDA autocast FP16 环境。以下 Linux 变量与实跑一致。
```sh
ROOT=/home/shenc/tta_geometry_specificity_asus
PY="$ROOT/runtime/bin/python"
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=GPU-d329c8d0-f276-5f23-03a9-df221a2021ed
S=P2/translation_padding
"$PY" "$S/infer.py" --root "$ROOT" --out "$S"
python -B "$S/analyze.py"
python -B "$S/plot.py"
```

仅复算已提交数据时运行最后两条，依赖 NumPy、Matplotlib。重做 GPU 测量可在新的仓库副本运行，脚本按环境指纹逐图对保存断点。不含断点的已发布结果来自一次完整测量。实际 157.998 秒，2,048 个新 edge 输入，20 次冒烟重评和 36 次精度复评，共 2,104 次视觉前向。2,560 条原图及反射历史评分核对了权重、预处理、归一化输入、token、原图和视图 RGB 及人工映射。每方向零位移一致；所有正位移只在移出画面后出现的边带允许不同。[环境](environment.json) · [运行记录](run.json)

[预设](protocol.md) · [中文报告](report.md) · [逐图分数](scores.csv) · [像素统计](pixels.csv) · [分差及选择](pair_directions.csv)。input_masks.npz 包含 2,048 张实际改变掩膜，键与 pixels.csv 一致；`np.unpackbits(archive[key]).reshape(224,224)` 恢复布尔格点。重复单另列于 repeated.csv，未计入部署评分。主分析 256 对 ImageRewardDB validation 原图，每图部署两视图。
