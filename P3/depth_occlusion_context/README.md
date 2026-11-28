# Middlebury 裁窗周边实验

[中文结果](report.md) · [冻结方案](protocol.md) · [逐年诊断](diagnostics.md) · [实际输入与案例](figures/examples.jpg)

运行目录为 Asus 的 `/home/shenc/tta_depth_adaptive/P3/depth_occlusion_context`。只读基线是本仓库 aba9d08 的相邻目录 `../depth_occlusion_tta`。模型固定 Depth Anything V2 Metric Indoor Small，revision `8078d68a9c75a972131914f6afd0c1723be0da7f`。PyTorch 2.5.1+cu124、Transformers 4.46.0、Pillow 12.2.0、NumPy 1.26.4。采用 FP32，关闭 TF32。实际两张 4090 的 UUID、驱动和逐文件权重哈希保存在 worker_0.json 与 worker_1.json。使用两卡按冻结顺序交错分片，每图 R/P 同一设备，组内交替先后。图像、GT 与像素缓存留在 Asus。

## 数据及冻结

2005/06 取官方 FullSize 单曝光双视图包。评分用 view1，对应 disp1 和各自 dmin。view5/disp5 仅用于核验相机方向。2014 使用上一轮的 perfect 左图与索引。官方说明分别给出 2005 的 Illum1/Exp1 和 2006 的 Illum1/Exp2；焦距 3740 px、基线 160 mm、原始整数视差的零值无效。[2005 官方发布](https://vision.middlebury.edu/stereo/data/scenes2005/) · [2006 官方发布](https://vision.middlebury.edu/stereo/data/scenes2006/) · [2014 官方发布](https://vision.middlebury.edu/stereo/data/scenes2014/)

2026-11-28 22:29:36（北京时间）冻结 50 个输入及内容哈希。三年内容哈希无重复；747 对跨年图像组合经感知哈希和灰度缩略图相关核查（以 cross_year_audit.csv 的实际行数为准），并逐图检查了 [场景总览](scene_audit.jpg)。未发现可辨认的相同布景或跨年近重复；例如 2005 Dolls 为多玩偶陈列，2006 Baby 为单玩偶前景，系列在同年合簇。部分图使用相似道具，同一采集团队和同一公开数据来源仍限制独立性。2014 的 Sword1/Sword2 按预设作为两个发现样本场景，独立复验的编号系列合簇。freeze.json 记录清单、旧实现及协议摘要。核验新输出时使用库内已冻结清单。

## 在 Asus 复现

以下在含既有只读基线、模型、原生 2014 缓存的实验树执行。数据路径可通过环境变量迁移。复算历史结果时保留各文件名和签名。

```bash
cd /home/shenc/tta_depth_adaptive/P3/depth_occlusion_context
export OCC_DATA=/home/data/users/shenc/tta_occlusion_data
export CTX_DATA=/home/data/users/shenc/tta_occlusion_context
export OCC_MODEL=/home/shenc/tta_depth_adaptive/data/model
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python

# 取新年份原始包（已有文件可复用）。全新预登记时，先看 scene_audit.jpg，
# 再在无 freeze.json 的独立副本执行 "$PY" prepare.py freeze。复现沿用已发布的清单。
"$PY" prepare.py fetch
"$PY" coverage.py
```

已有完整预测时可直接运行下方 CPU 命令。确需从头计时时，使用独立输出副本及新的 CTX_DATA，在该副本让 synthetic_*.json 和 blend_time_*.json 作为新产物生成。正式复现最多 535 次新前向。已完成的单场按输入哈希、代码和预测签名断点复用。两张卡都应空闲，每个实验进程只看一个 UUID。

```bash
mapfile -t cards < <(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<100 && $3==0 {print $1}' | head -2)
test "${#cards[@]}" -eq 2 || exit 1
pids=()
for i in 0 1; do
  ( export CUDA_VISIBLE_DEVICES="${cards[i]}"; "$PY" score.py --shard "$i" && "$PY" time_blends.py "$i" ) > "gpu_$i.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
```

仅汇总表、区间及延迟汇总可从本仓库 CSV 和 JSON 重建。像素复算与案例需要 Asus 已记录的原始图和缓存。机器只做 CPU 复算时可隐藏 CUDA。

```bash
export CUDA_VISIBLE_DEVICES=""
"$PY" analysis.py
"$PY" summarize.py
"$PY" independent.py
"$PY" cases.py
"$PY" diagnostics.py
```

## 查表

| 要核查的内容 | 文件与单位 |
|---|---|
| 样本、簇、曝光、原图/GT/标定哈希 | scenes.csv、freeze.json、cross_year_audit.csv |
| GT 资格、相机、视差与米制、像素去重 | coverage.csv、coverage_summary.csv；分位深度为米，视差量化步长为毫米 |
| B/T/25%/C/P/R 及固定尺度 R/P | per_scene.csv 中八种图级预测，距离用米；paired_differences.csv 的逐图差与 summary.csv 的配对区间用毫米 |
| 深度被推近/推远及其误差变化 | pixel_movements.csv 的三档互斥，movement_summary.csv 同时给簇等权比例及直接像素汇总比例 |
| 大画布核心和外圈 | windows.csv 给出原生矩形、编码器核心矩形、连续视野、实际倍率、输入哈希、尺度；两种大画布的核心均逐元素等于同一 T 张量 |
| 成本和结果复核 | latencies.csv、latency_summary.csv、synthetic_*.json、blend_time_*.json、worker_*.json、cache_index.csv、verification.json |
| 正反例的取样位置 | cases.csv、figures/examples.jpg |

新缓存位于 `$CTX_DATA/pred/<年份>/<场景>/`。NPZ 中 R/P 为四窗 FP32 原核心大小的预测，新年份同时保存 B 全图与 T 四窗。inference.json 记录哈希、坐标与尺度系数，含 GT 的文件不进入预测。2005/06 评价位置在 `$CTX_DATA/edges/`，2014 读取旧 edges.npz。场景级指标基于相同像素。汇总先在每个编号系列内平均，再按簇等权；直接合并所有像素或对 27 场简单平均会得到不同数字。
