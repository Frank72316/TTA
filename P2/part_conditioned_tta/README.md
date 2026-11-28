# P2 · 按题目选择鸟体部位视图

[中文报告](report.md) · [四格表](four_cells.csv) · [真实案例](cases.jpg) · [全部评分](scores.csv) · [冻结协议](protocol.md)

**材料。** 随附包中的 `pairs.csv`、`analyze.py`、`freeze_pairs.py` 与《运行提示词》保留原字节。ZIP 的 SHA-256 为 `eeb901ad71d3a240969e657d3c44393edf59144f35aa2d7e49d1811df7a94487`。本次直接使用清单，没有运行重新抽样程序。官方数据取 CUB-200-2011 的 2011 版原始 JPEG，归档 MD5 为 `97eceeb196236b17998738112f37df78`。数据出处为 Wah、Branson、Welinder、Perona 和 Belongie 的 [CaltechDATA 记录](https://data.caltech.edu/records/65de6-vp158)。示例图与冒烟图为数据集原照片的裁剪、缩放和标示；原照片保存在 Asus 的数据目录中。

**如何查一个结果。** 在 [图对清单](pairs.csv) 按 `pair_id` 找 A/B、颜色与鸟冠或翅膀的阳性标签。按 `image_id,view,query` 在评分表中找原生点积。双视图分别对 A、B 做算术平均，再比较两个均值。每题独立决定 A、B 或同分。可在 [逐题判定](questions.csv) 查候选聚合分数、选择、赢家和有向分差，在 [逐对判定](pair_choices.csv) 查两题组合，在 [策略汇总](strategy_summary.csv)、[状态转移](transitions.csv) 和 [配对区间](contrasts.csv) 查分母与差异。状态转移按全体或高确定度子集、两题合计或每题单列，保留零数格。`results/` 为实验包分析器自己的输出。

**如何查模型实际看到了什么。** [评分输入](inputs.csv) 仅含输入相关列以及 JPEG 哈希。[坐标记录](geometry.csv) 每张照片三行，右下角为开区间。填边比例相对于整张方窗，冠翅 IoU 相对于未截短的两个源图坐标方窗。两题共享的图像张量 SHA-256 和实际文本 token 哈希均在评分表中。示例使用反归一化后的 224×224 张量显示。冒烟图每行从左到右依次为带标点原图、实际原图输入、鸟冠窗、指定翅窗、画心窗。[C001 A](smoke/C001_A.png) · [C001 B](smoke/C001_B.png) · [C002 A](smoke/C002_A.png) · [C002 B](smoke/C002_B.png)

### 仅从缓存复算

在仓库根目录执行。需要 Python 3.10+ 和 NumPy；运行评分脚本另需项目 HPS 环境，生成图片使用 Pillow。独立核验先按候选累加精确有理数，再核对实验包全部有向分差与三个物种簇区间。下面的输出位置可自行替换。

```sh
S=P2/part_conditioned_tta
OUT=$(mktemp -d)
(cd "$S" && sha256sum -c SHA256SUMS)
cp "$S/scores.csv" "$OUT/"
python "$S/analyze.py" "$OUT/scores.csv" --out "$OUT/results"
python "$S/verify_results.py" --out "$OUT"
```

### 从原照片复现

以下为本次 Asus 环境。使用同一源码版本并在仓库根目录执行；`DATA` 放数据，`OUT` 放可续跑的逐照片断点及实际输入图。只复算报告数字可直接使用上面的缓存命令。解包仅取冻结的照片和核验所需标注，图级属性文件仅保留所选图像的原始行。`prepare.py` 在实验目录写入可复核的评分输入和数据核验结果。重新绘制案例需要推理阶段保存的 `display_cache/`。

```sh
S=P2/part_conditioned_tta
BASE=/home/shenc/tta_geometry_specificity_asus
DATA=/home/shenc/tta_prompt_flip/data/cub_part_96
PY="$BASE/runtime/bin/python"
OUT=$(mktemp -d)
mkdir -p "$DATA"
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
# 归档只需获取一次
test -f "$DATA/CUB_200_2011.tgz" || curl -fL 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1' -o "$DATA/CUB_200_2011.tgz"
"$PY" "$S/prepare.py" --archive "$DATA/CUB_200_2011.tgz" --data "$DATA"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/infer.py" --pilot "$BASE/inputs/tta_pilot" --data "$DATA" --out "$OUT" --smoke
# 核对 OUT/smoke/ 中的两对及其 32 行评分，再续跑
"$PY" "$S/infer.py" --pilot "$BASE/inputs/tta_pilot" --data "$DATA" --out "$OUT"
"$PY" "$S/analyze.py" "$OUT/scores.csv" --out "$OUT/results"
"$PY" "$S/verify_results.py" --out "$OUT"
"$PY" "$S/examples.py" --out "$OUT"
```

本次在 Asus 实验目录执行，`OUT` 取该实验目录本身。视觉和文本的批大小均为 1，模型与权重、官方预处理、软件版本及设备 UUID 见 [environment.json](environment.json)。[冒烟运行](smoke_run.json) 与 [正式运行](run.json) 分开记录模型加载在内的墙钟、评分循环耗时、视觉前向、文本前向和断点命中数。每个视觉输入服务两道题，每次启动仅对不同提示词各编码一次。
