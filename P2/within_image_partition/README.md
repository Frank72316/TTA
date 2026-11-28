# P2 · 同画布分区与名词方向证据

[一页中文报告](report.md) · [逐视图原始分数](scores.csv) · [冻结分组](groups.csv) · [全部单照片条件](image_conditions.csv) · [主分析照片](primary_photos.csv) · [中线图对诊断](midline_pairs.csv) · [案例图](cases.png)

**样本与冻结。** [协议](protocol.md)和[清单及哈希](freeze.json)在加载评分前生成。[照片表](photos.csv)保留原图与双替换名词、种子 42 的供体、COCO 方向和对应图对。模型实际可见原图在黑边内，原有输入框与源框仿射映射的坐标逐边核对。几何分类在裁到原图边界的连续框上计算，渲染在原始 RGB 上进行。384 个照片与线组合中有 184 个分开、115 个同侧、85 个部分跨线。将连续线改为冻结的整数像素线后，组别差异为 0。主样本中 16 对有两张合格，32 对有一张，合计 64 张；方向分组的图对数可以重叠。

**计分。** 每个保留侧均使用相同灰色 (128,128,128) 替换另一侧。固定模板为 `A photo of the {noun}.`，tv 按旧实验写作 TV。每个视图只编码一次，四个名词分别打分。设 E 为「左 X − 右 X ＋ 右 Y − 左 Y」；当照片中 X 在右时整体取负，替代词沿用相同符号。主指标先在同一照片上计算「真词分开减同侧」减「替代词分开减同侧」，再按照片等权平均。区间以原始自然图对为簇，10,000 次有放回抽样，种子 42，同一张照片的条件和同一图对中的合格照片一起进入副本。中线检查另比较真实左图 E 减真实右图 E，旧裁图的 1,024 条真/替代词评分从缓存重算。[汇总、区间及九格转移](summary.json)

**运行。** Asus 物理 3 号 RTX 4090，启动前占用 15 MiB、利用率 0%，按设备 UUID 绑定。eval，单图单文本，CUDA autocast FP16；原生 FP16 分数转 float64，再作差。2 张冒烟用 12 次视觉前向、48 条评分；正式阶段复用这 48 条，新增 756 次视觉前向和 3,024 条评分。两阶段含各自模型加载的总墙钟 54.38 秒。逐图验证原 JPEG 哈希、全图输入张量哈希；两个冒烟原图及各分割的实际模型输入保存在 smoke/。12 个完整保留目标框检查覆盖两个目标；在避开插值缝的框内，归一化张量与原图同坐标的最大差均为零。用相同仿射映射同时核对框的位置与尺度。[冒烟检查](smoke.json) · [模型、环境、权重](environment.json) · [计数与耗时](runs.json)

从仓库根目录复算已保存的结果，仅需 Python、NumPy。重画图另需 Pillow 和 Matplotlib。
```sh
S=P2/within_image_partition
python -B "$S/design.py"
python -B "$S/analyze.py"
python -B "$S/verify.py"
python -B "$S/figure.py"
```

在 Asus 重跑推理的命令如下。独立输出目录保存每张照片的断点。先完成两张冒烟，随后继续其余 126 张。用于评分的冻结文件位于脚本目录。
```sh
S=P2/within_image_partition
ROOT=/home/shenc/tta_geometry_specificity_asus
DATA=/home/shenc/tta_prompt_flip/data/val2017
PY="$ROOT/runtime/bin/python"
OUT=$(mktemp -d)
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" "$S/score_views.py" --data "$DATA" --pilot "$ROOT/inputs/tta_pilot" --out "$OUT" --smoke
"$PY" "$S/score_views.py" --data "$DATA" --pilot "$ROOT/inputs/tta_pilot" --out "$OUT"
"$PY" "$S/analyze.py" --out "$OUT"
"$PY" "$S/verify.py" --out "$OUT"
"$PY" "$S/figure.py" --out "$OUT"
```

**照片署名。** 两个公开展示的案例恰为冒烟照片。F001 A，COCO 199771，为 DC Central Kitchen 的 [DSC01812](https://www.flickr.com/photos/dccentralkitchen/7115744429/)；F002 A，COCO 294350，为同一摄影者的 [DSC01124](https://www.flickr.com/photos/dccentralkitchen/6762303723/)。两张作品均采用 [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) 许可。展示版本经过模型输入缩放、补边、灰色遮挡，原图示意面板另加 COCO 框和分线。
