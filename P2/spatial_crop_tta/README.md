# P2 · 固定半图与名词证据

[一页中文报告](report.md) · [案例图](figures/cases.png) · [冒烟实图](smoke/overview.jpg) · [512 条新评分](scores.csv) · [逐图定位](image_results.csv) · [逐对分配](pair_results.csv)

自然照片、COCO 标注、A/B、X/Y 和全图方向句均沿用已冻结的 [prompt_aware_flip](../prompt_aware_flip/README.md)。全图只读取其 512 个原生分数。新评分使用同版本 HPS v2.1、官方验证预处理、CUDA autocast FP16、单视觉输入及单文本输入批量。两张候选共享同一对名词、同一方向约定。统计单位是一个原始自然图对。

## 半图怎样进入模型

1. 先核对原 JPEG 哈希。原 HPS 验证预处理保留全部 RGB 画面，长边缩放并补黑边到 224×224。重新生成的 128 个全图输入张量均与上轮哈希相同。裁剪取自完整可见的原始 RGB 像素，去掉上轮显示用的边框。
2. 原图宽高为 W、H，连续像素边坐标的固定中线为 W/2。将每个像素原值复制为 2×2，再在格点 W 处分开，得到两个大小完全相同的 W×2H 像素数组。奇数宽时中心像素的左右半格各分给一个视图。每一半只依赖对应半幅的像素，左右对称。两个半图各走一次相同的官方整幅缩放及居中补边。
3. 令 `s = 224 / max(W, 2H)`。有效宽高为 `u = round(W × s)` 和 `v = round(2H × s)`。左、上补边分别为 `padLeft = (224 − u) // 2`、`padTop = (224 − v) // 2`。源半幅起点 x₀ 为 0 或 W/2。原图边坐标映射为 `x′ = padLeft + (x − x₀) × u / (W/2)`、`y′ = padTop + y × v / H`。每张照片都保存原画幅、上轮有效矩形、精确中线、两半起点、比例和实际补边。还用常量画布逐张核验有效区。[映射与名词](views.csv)
4. 分类分层在评分前完成。对每个可见框计算落在两个半边的面积比例，以精确分数与 3/4 比较。由于只作竖切，面积比例也等于框的水平宽度比例。分层脚本从原始框计算，并与上轮模型输入框坐标核对。评分器只使用固定几何、原图和名词；框位置仅供分层、判真值与展示。[预设与文件哈希](freeze.json) · [逐图面积比例](strata.csv)

名词模板为 `A photo of the {noun}.`，使用全部 49 个冻结类别名，tv 统一写作 TV。两对冒烟以图对 ID 顺序取首两对可公开展示裁图的照片，分别是 F002、F005。报告案例分别从“四评分全图分反、局部纠正”和“局部选反”中取满足展示许可的最小 ID。抽样与主统计始终保留全部 64 对。

## 复算与复跑

以下命令从本目录执行。分析只需 Python 3.10+ 和 NumPy。各方法对真实左图的证据减真实右图的证据作严格符号判定；缓存中的原生值转为 binary64 再作差。独立核验从原始分数直接比较“正确与颠倒分配的总分”，用精确有理数核对全部 192 个有向分差。主区间以种子 42 对 64 对配对正确指标差重采样 10,000 次，取 2.5% 和 97.5% 分位数，平局留在分母中。

```bash
python -B analysis.py
python -B verify.py
```

重新推理需本机已有的原 COCO JPEG、与上一轮相同的 HPS 环境和权重。DATA 指向 val2017 JPEG 目录、PILOT 指向含 environment.json 和权重缓存的先导目录。可以用新的 OUT 输出目录复现，两个阶段共用该目录和逐对断点。运行期间按环境、预处理、数据清单、评分脚本及权重签名核验断点。

```bash
export OUT="$HOME/tta_spatial_crop_replay"
export DATA="$HOME/tta_prompt_flip/data/val2017"
export PILOT="$HOME/tta_geometry_specificity_asus/inputs/tta_pilot"
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export CUDA_VISIBLE_DEVICES=GPU-d329c8d0-f276-5f23-03a9-df221a2021ed
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY="$HOME/tta_geometry_specificity_asus/runtime/bin/python"
"$PY" -B design.py --out "$OUT"
"$PY" -Bu inference.py --data "$DATA" --pilot "$PILOT" --out "$OUT" --smoke-only
"$PY" -Bu inference.py --data "$DATA" --pilot "$PILOT" --out "$OUT"
"$PY" -B analysis.py --out "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -B verify.py --out "$OUT" --photos "$DATA"
python -B present.py --out "$OUT" --photos "$DATA"
```

实际评分在 Asus 物理 2 号 RTX 4090 上完成，工作目录为 `/home/shenc/tta_prompt_flip/P2/spatial_crop_tta`，使用默认 OUT（该工作目录）。本地画图使用 Pillow 和微软雅黑，也支持 Linux Noto Sans CJK。完整评分、冒烟及定位表的分数保留原精度。案例图中的分数显示至小数点后六位。

## 核验与数据索引

- [逐图原始分数](scores.csv) 中每张照片都有左右两半 × X/Y 的四项评分。256 个视觉张量互异。GPU 每个半图只编码一次，然后与两个名词分别计算相似度；名词文本特征在单次进程内按提示复用。
- [逐图定位](image_results.csv) 为全部 128 张照片保存四分数、E、两个名词各自对 E 的有向贡献及两个全图对照。仅分居清楚的 78 张有框导出的半边定位真值，其余 50 张相应列留空。[机制计数](localization_summary.csv)
- [逐对分配](pair_results.csv)、[分层汇总](summary.csv)、[转移与全部图对 ID](transitions.csv)、[配对区间](analysis.json) 可复算主结果。仅一个物体定位正确时，以该照片的 E 符号判定它能否单独读出方向，和两张照片之间按 E 排序的主指标分别计数。
- [核验结果](verification.json) 包含 CPU 上用独立像素索引重建的 256 个半图张量哈希，以及 8 张反归一化冒烟视图的逐像素相等检查。[实际运行耗时](runs.json) · [设备、权重与软件版本](environment.json) · [图片来源与许可](credits.md)
