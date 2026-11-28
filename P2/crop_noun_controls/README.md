# P2 · 裁剪视图中的名词身份

在仓库根目录执行。GPU 实跑使用 Asus 的 RTX 4090、冻结的 HPS v2.1 和历史单图单文本 CUDA autocast FP16 环境。以下 Linux 变量与实跑一致。
```sh
ROOT=/home/shenc/tta_geometry_specificity_asus
PY="$ROOT/runtime/bin/python"
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
S=P2/crop_noun_controls
# COCO 2017 官方 instances_val2017.json，散列见 freeze.json
python -B "$S/freeze.py" /path/to/instances_val2017.json
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/infer.py" --root "$ROOT" --data /home/shenc/tta_prompt_flip/data/val2017
python -B "$S/analyze.py"
```

从仓库已有分数复算时只需最后一条（Python、NumPy）。每种方案、每张图片都问四次相似度。左、右半图实际输入张量与 spatial_crop_tta 一致，主结果使用每图对的同一对替代词。每张图的 E 为左 X 减右 X 加右 Y 减左 Y；真实左图 E 减真实右图 E 的符号决定正确、颠倒、同分。可逐侧核对 pair_results.csv 的 E。两真名词复用旧 512 条；原实验未保存可用半图特征，此次按原精度重编码 256 个视觉输入，供三个映射共享。原 2 对冒烟的 16 个分数与旧值完全一致，主映射新增 512 个相似度，另两个映射合计额外计算 820 个独立相似度。推理 27.37 秒。

预先随机化邻接表后做最大二分匹配，一供体一受体。mappings.csv 保存供体、可选供体数和两个候选的所有已标注类别并集。匹配不读 HPS 输出。空间真值、既有框分层及镜像实验的图对顺序不变，比较在每个映射的相同图对上进行。[协议](protocol.md) · [冻结哈希](freeze.json) · [映射](mappings.csv) · [环境](environment.json) · [运行量](run.json) · [报告](report.md)
