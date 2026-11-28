# P3 · 区域深度差与远近排序

[一页中文报告](report.md) · [冻结协议](protocol.md) · [案例图](cases.jpg) · [汇总](summary.csv)

**来源。** 固定在 `3e0ea7c340a5bf27faf784b6e16a45a27571a7f0` 的 [depth_translation_pairing](../depth_translation_pairing/README.md)。[来源文件 SHA-256](source.json) 与 [本轮实际运行记录](run.json)关联 NYU 深度归档、上游环境、冻结清单、视图审计、分析代码和协议。上游缓存为 FP32，文件签名、帧 ID、视图、输入及对齐预测哈希逐项核对。每个窗的预测在 CPU 转 float64 后求均值。以毫米真值的整数和及像素数精确判断阈值边界，再用米制 float64 计算误差。依赖 Python 3.10+、NumPy、Pillow；案例图使用 Noto Sans CJK。全程不需要模型、GPU 或联网数据。

**如何读表。**

- [逐窗数据](windows.csv)用共同视野坐标记录 32×32 方窗、有效像素数、真值毫米和、均深与各视图均深。加 26 即得原始照片横坐标。1–16 号按行优先编号。
- [逐区域对](region_pairs.csv)中 A<B。d 为真值 A−B，x 为各视图 A−B，q 为独立部署一套配方时预测的 A−B。正数为 A 更远。每个配方分别给出平方误差与判定；G 为同向误差减反向误差，G_product 独立由左右方向差乘积计算。每个区域对只占一行。
- [逐帧结果](frame_results.csv)每帧每层一行，n 为有效区域对数，n=0 时指标留空。[汇总](summary.csv)按层与指标给出点估计和场景簇区间。acc、fix_rate、harm_rate 等比率单位为百分比，A 和 acc_vs_O 为百分点，G 和 gain_vs_O 为 m²。两种 gain 指标取正均有利于反向。main 包含 small、medium、large，near 与 equal 单列。各层分别对有合格区域对的帧等权。[各层覆盖](coverage.csv)
- [判定转移](transitions.csv)分别给出两个方向的原始计数及含同分的九格。逐帧表中的纠正率和改错率给两个方向各 1/2 权重。案例以 [validation.json](validation.json) 保存的顺序和数量为准，主层中符合“误差变小且判定维持”的有 10,888 次方向比较，符合“误差变小且纠正”的有 48 次。

### CPU 复算汇总和独立核验

从仓库根目录运行。复算只需已发布的逐区域对、逐窗文件和运行记录。独立核验改用先在 A、B 各自聚合配方再作差，以精确有理数复核真值分层，并直接抽取整簇帧复算四项主比较和原图对比的区间。

```sh
S=P3/depth_order_pairing
OUT=$(mktemp -d)
cp "$S/windows.csv" "$S/region_pairs.csv" "$S/run.json" "$OUT/"
python -B "$S/summarize.py" --out "$OUT"
python -B "$S/verify.py" --out "$OUT"
```

### 在 Asus 从像素重建

实际运行目录为 `/home/shenc/tta_depth_adaptive/P3/depth_order_pairing`。从仓库根目录运行以下命令。正式阶段读取已通过哈希核验的冒烟产物，只处理剩下 126 帧。

```sh
S=P3/depth_order_pairing
BASE=/home/shenc/tta_depth_adaptive
ZIP="$BASE/data/nyu-depth-png.zip"
PRED="$BASE/P3/depth_translation_pairing/predictions"
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
OUT=$(mktemp -d)
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
"$PY" "$S/order_data.py" --archive "$ZIP" --predictions "$PRED" --out "$OUT" --smoke
cat "$OUT/smoke.json"
"$PY" "$S/order_data.py" --archive "$ZIP" --predictions "$PRED" --out "$OUT"
"$PY" "$S/summarize.py" --out "$OUT"
"$PY" "$S/verify.py" --out "$OUT"
"$PY" "$S/cases.py" --archive "$ZIP" --out "$OUT"
```

[冒烟记录](smoke.json)包含 2 帧、10 个数组、240 个区域对及 1,200 次先聚合像素再作差的配方检查；[独立复核](replay.json)覆盖 15,360 个区域对和 76,800 个配方判定。两次提取在 Asus CPU 上共约 3.94 秒，含归档哈希检查，不含后续统计和绘图。
