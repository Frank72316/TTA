# P2 · 压缩网格的相位

在仓库根目录执行。GPU 实跑使用 Asus 的 RTX 4090、冻结的 HPS v2.1 和历史单图单文本 CUDA autocast FP16 环境。以下 Linux 变量与实跑一致。
```sh
ROOT=/home/shenc/tta_geometry_specificity_asus
PY="$ROOT/runtime/bin/python"
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
S=P2/compression_phase
CUDA_VISIBLE_DEVICES="" "$PY" "$S/calibrate.py" --root "$ROOT"
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/infer.py" --root "$ROOT" --device 3
python -B "$S/analyze.py"
python -B "$S/plot.py"
```

从已发布结果复算时仅需最后两条。旧 64 对的 calibration.csv 记录 128 张图 ×（四个 JPEG＋二十档高斯）的能量和 RGB MSE。校准不读取名次字段或模型分数。strength.json 先于主批推理写入。独立重做全流程宜在新副本中运行，保留现有 strength.json 及环境签名供比对；耗时字段随机器运行而变化。主批不按结果调半径或相位。

JPEG 使用 Pillow 12.2.0，libjpeg-turbo 3.1.4.1，4:4:4、非渐进、optimize=False。反射保护边界至少 16 源像素；相位顺序为 (0,0)、(4,0)、(0,4)、(4,4)。只在官方实际可见的内容矩形计算图像统计，剔除模型输入的黑色补边。亮度按 Rec.601 系数取 RGB 线性组合，水平及垂直一阶差分的均方各占一半；像素 MSE 在反归一化的 RGB 上以 0–1 为量纲。主表在每个相位先以 O 和该版本等权判定，再等概率平均四个判定。每图实际部署 K=2。

主批 4,608 条记录、512 条经输入哈希核对的历史 O 分数、1,769 次同图同提示下的重复张量复用；2,331 次实际视觉前向含 4 次冒烟原图复评。CPU 校准 47.12 秒，主批 132.90 秒。两对冒烟先运行，之后按图对保存断点。[协议](protocol.md) · [报告](report.md) · [校准](calibration.csv) · [评分](scores.csv) · [逐对结果](pair_results.csv) · [机器和模型](environment.json)
