# P3 · 固定三次推理的平移方向搭配

[中文报告](report.md) · [冻结协议](protocol.md) · [逐帧数据](frame_results.csv) · [总体比较](summary.csv) · [极值帧](cases.png)

输入来自 [depth_adaptive_budget 的冻结清单](../depth_adaptive_budget/frames.csv)。按帧 ID 连接原实验逐帧结果的 scene 字段；在仓库中可独立核对[场景映射](../depth_adaptive_budget/scenes.csv)。本批为已用的 128 个开发帧。[上游原图推理与缓存实现](../depth_adaptive_budget/measure.py)及[上游的像素级米制评价](../depth_bias_regions/measure.py)给出原始口径。

### 从逐帧数据复算统计

从仓库根目录运行，需要 Python 3 和 NumPy。生成 summary.csv 和 validation.json。环境、逐帧视图哈希、模型版本和 GPU 计时可分别从 [environment.json](environment.json)、[view_audit.csv](view_audit.csv) 和 [runs.json](runs.json) 查看。

```sh
S=P3/depth_translation_pairing
OUT=$(mktemp -d)
cp "$S/frame_results.csv" "$S/view_audit.csv" "$OUT/"
python -B "$S/analyze.py" --out "$OUT"
```

### 在 Asus 从图像和旧 O 缓存复现

本次工作目录为 `/home/shenc/tta_depth_adaptive`，下列命令从仓库根目录执行。使用项目已有 PyTorch、Transformers、NumPy、SciPy、Pillow、Matplotlib 环境。完整像素缓存保存在 Asus 本实验的 `predictions/`。示例将新的运行写入单独目录。如只复查已有像素缓存或重画案例，可设置 OUT 为本次结果目录，直接执行最后三条 CPU 命令。

```sh
S=P3/depth_translation_pairing
DATA=/home/shenc/tta_depth_adaptive/data
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
OUT=$(mktemp -d)
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/run.py" --archive "$DATA/nyu-depth-png.zip" --model "$DATA/model" --old-cache "$DATA/pred_cache" --out "$OUT" --smoke
cat "$OUT/smoke_proof.json"
"$PY" "$S/run.py" --archive "$DATA/nyu-depth-png.zip" --model "$DATA/model" --old-cache "$DATA/pred_cache" --out "$OUT"
"$PY" "$S/analyze.py" --out "$OUT"
"$PY" "$S/verify_pixels.py" --archive "$DATA/nyu-depth-png.zip" --old-cache "$DATA/pred_cache" --out "$OUT"
"$PY" "$S/cases.py" --archive "$DATA/nyu-depth-png.zip" --out "$OUT"
```

**缓存与精度。** 模型版本为 `8078d68a9c75a972131914f6afd0c1723be0da7f`。核对数据 ZIP、帧清单、原推理和视图源码、模型与预处理器权重散列、框架版本。历史缓存目录以旧环境文件 SHA-256 为名。原图缓存已经按 3% 实验的共同区域裁过，宽 602；本轮只需其中居中的 588 列。每帧核对原图缓存 FP32、尺寸与有限性，记录原 JPEG 解码后的 RGB 哈希、处理器输出张量和对齐预测哈希。旧缓存中没有历史逐帧输入张量哈希；输入身份通过同一归档、帧路径、预处理与两帧 O 的逐像素前向重放核对。两帧复验在整个原缓存区域逐值相同。旧 3% 平移预测没有进入本轮。运行时仅加载 O 历史缓存，所有新增平移均从原 RGB 独立生成。[冒烟记录](smoke_proof.json)

**坐标与单位。** 全部照片为 640×480。左、右移动 2% 对应 13 像素，4% 对应 26 像素，预处理前在源图左右作 reflect 填边。FP32 前向结果经双三次插值回 480×640，再按实际位移逆对齐。像素交集为全高、原图横坐标 [26,614)，再取 (0,10] m 的真值。真值从原始毫米在 float64 中换算，聚合、平方、乘积和统计也用 float64。128 帧共 36,126,720 个几何共同像素，均满足深度范围。图中 D2、D4 使用跨两帧两列合并的对称色标；贡献另用两帧共用的对称色标。色标上限取相应绝对值的 99% 分位。

**核验与计时。** 冒烟新增 8 个平移前向并复验 2 次 O；正式阶段续跑 126 帧，命中冒烟的 10 个已对齐视图，新增 504 个平移前向。全实验评估 640 个不同的帧与视图组合，其中 128 个 O 取自旧缓存。两阶段墙钟包括数据、权重核验、加载、推理及逐帧评价；同步模型前向合计 7.00 秒。每视图的 forward_ms 对 O 只记录两帧复验时间，对余下的历史命中记 0。[独立 CPU 复核](pixel_replay.json)重新对账全部 128 个历史 O 与 640 个对齐预测，对残差取均值后平方并复核贡献。[分差恒等式及统计验收](validation.json)
