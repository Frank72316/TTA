# P2 · 镜像时是否应该同步交换提示词方向？

[中文报告](report.md) · [案例图](figures/cases.png) · [冻结图对](pairs.csv) · [512 条原始评分](scores.csv) · [逐对结果](pair_results.csv)

实验使用 COCO 2017 validation 实例框定义左右真值。主指标要求同一对自然照片对两道方向相反的题均严格选对。自身镜像对的代数检查单独保存。仅重新出统计表时无需下载照片或加载模型。

## 从缓存复算

从本目录执行。统计使用 Python 3、NumPy，准备图片与绘图还需 Pillow。

```bash
python -B analyze.py
```

[summary.csv](summary.csv) 记录三种策略的正确数、同分数与换词仍选同图的频率。[comparisons.csv](comparisons.csv) 记录以完整图对为单位的差值与配对 bootstrap 区间。[question_transitions.csv](question_transitions.csv) 区分逐题纠正和改错。[mirror_check.csv](mirror_check.csv) 记录八张图各两个提示的检查结果。

## 从 COCO 标注和原图复现

下面的命令从本目录执行。使用一个新的输出目录可独立重跑。DATA 为数据缓存，OUT 为实验输出目录，PILOT 为以往 HPS 实验的输入目录，包含 environment.json 和已经校验的 data/model-cache。GPU 环境与相邻的 geometry_specificity 实验相同，源代码通过该目录的 score.init 加载固定的 HPS v2.1 权重和官方验证预处理。

```bash
export DATA="$HOME/tta_flip_data"
export OUT="$HOME/tta_flip_replay"
python -B prepare.py --data "$DATA" --out "$OUT"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -Bu inference.py --data "$DATA" --out "$OUT" --pilot "$PILOT"
python -B analyze.py --out "$OUT"
python -B present.py --out "$OUT" --views "$DATA/model_views"
python -B verify.py --out "$OUT" --annotations "$DATA/instances_val2017.json"
```

首次准备时下载 COCO 实例标注和入选的 128 张照片。已有 instances_val2017.json 时直接校验并使用。正式运行使用官方 trainval 标注 ZIP 中的 validation JSON 成员，SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。输入图片按 COCO 发布的 JPEG 原始像素方向转成 RGB，与标注尺寸逐张核对。[COCO 官方下载](https://cocodataset.org/#download)

准备步骤写出 [带框预览 HTML](box_previews.html)，下载后可用浏览器打开，图片来自 COCO。完整离线版会生成在 DATA/box_previews/。冻结哈希覆盖 pairs.csv、images.csv 和预览，评分器开跑前核验。各图对断点写入 DATA/flip_scores/，再次运行时按清单与评分环境签名校验。正式批次的模型输入另存为 DATA/model_views/，供案例图使用。[图片与标注详情](images.csv) · [案例图署名](photo_credits.md)

## 正式 GPU 运行

Asus 的物理 2 号 RTX 4090。已安装的 HPS 包、权重和 Python 环境均复用自前一轮。评分程序在同一进程内先通过 2 对冒烟，落盘 smoke.json 后续跑 62 对。每张照片只做两次图像编码，分别把每个图像特征与两个文本特征打分。实际评分命令如下。

```bash
cd /home/shenc/tta_prompt_flip
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export CUDA_VISIBLE_DEVICES=GPU-d329c8d0-f276-5f23-03a9-df221a2021ed
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/home/shenc/tta_geometry_specificity_asus/runtime/bin/python -Bu P2/prompt_aware_flip/inference.py \
  --data /home/shenc/tta_prompt_flip/data \
  --pilot /home/shenc/tta_geometry_specificity_asus/inputs/tta_pilot
```

[模型权重、版本、预处理与清单指纹](environment.json) · [两对冒烟](smoke.json) · [实际耗时](runs.json) · [独立核验](verification.json)
