# 遮挡边界的局部放大 TTA

[一页中文报告](report.md) · [完整统计](summary.csv) · [数据与尺度诊断](diagnostics.md) · [正式设计](protocol.md) · [PFM 单位勘误](pfm_correction.md)

**数据状态。** 预期清单 100 个 iBims-1 主库 ID 和 23 个 Middlebury 外部场景均已冻结。Middlebury 完成 23/23。可验证的 iBims-1 高清配对为 0/100，原因逐行见 inventory.csv。真实数据、FP32 像素预测及评价位置留在 Asus，共享存储目录为 `/home/data/users/shenc/tta_occlusion_data`。禁止把 VGA 插值或手持辅助图当成高清激光配对。

## 重跑环境

从仓库根目录进入下列目录。实跑代码在 Asus 的 `/home/shenc/tta_depth_adaptive/P3/depth_occlusion_tta`。上级 depth_adaptive_budget/environment.json 提供权重 SHA-256。使用模型 revision `8078d68a9c75a972131914f6afd0c1723be0da7f`，torch 2.5.1+cu124、transformers 4.46.0、Pillow 12.2.0、NumPy 1.26.4、SciPy 1.15.3。GPU 物理编号 1–7 为 RTX 4090，均按 UUID 绑定。除图像像素外，模型无需相机参数；标定和激光/视差只进入评价。所有方法均按光轴深度、米制和 (0,20] m 真值范围评价。

```bash
cd P3/depth_occlusion_tta
export OCC_DATA=/home/data/users/shenc/tta_occlusion_data
export OCC_MODEL=/home/shenc/tta_depth_adaptive/data/model
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
bash fetch.sh
"$PY" io_data.py
"$PY" inventory.py
```

如取得独立可验证的 HD 发布，在 `$OCC_DATA/ibims_hd/<冻结ID>/` 放入该记录的原生 `rgb.png`、同坐标的 FP32 米制光轴深度 `depth_m.npy`、`metadata.json`。后者应记录真实来源、单位 `m`、坐标 `optical_axis`、已核验配准的 `verified_registration`，可确认时写具体的 `room_id`。可用的透明和无效掩膜命名为 mask_transp.png、mask_invalid.png，0 表示须排除；可选同坐标 edges.png 及 mask_floor.png、mask_table.png、mask_wall.png 保持原生分辨率。转换从可追溯的原始单位完成。若只有现有官方 VGA 包，脚本仅调度 Middlebury。在独立运行副本内重做盘点和评分可接续缺失的主库。

## 校准、评分与审计

正式登记的时间按 UTC+8。设计开始于 21:32:15，Asus 上协议写入 21:33:40，预期清单及两份摘要写入 21:33:42，合成校准冻结在 21:35:22。首个正式缓存 21:36:05 写出。GT-only 单位勘误在 21:39:35 写入，21:41:52 生成正式方法比较。严格保留原协议与勘误。选取测试数据时未使用模型效果。实际代码从 RGB 到原坐标主策略输出计时，公开的逐元素评分均由缓存重算。

```bash
# 参考卡校准只需做一次。复用正式缓存时保留已有 calibration.json/csv。
CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4 "$PY" predict.py calibrate
# 用空闲卡按冻结顺序轮转分片，各场景的 N/L 始终在同一卡。
mapfile -t cards < <(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<100 && $3==0 {print $1}')
test "${#cards[@]}" -gt 0
pids=()
for i in "${!cards[@]}"; do
  CUDA_VISIBLE_DEVICES="${cards[i]}" "$PY" predict.py score --shard "$i" --shards "${#cards[@]}" > "gpu_$i.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
"$PY" edges.py --jobs 8
"$PY" figures.py audit
"$PY" evaluate.py --jobs 8
"$PY" summarize.py
"$PY" verify.py
"$PY" account.py
"$PY" figures.py cases
```

为重建同一份正式评分，请使用原有的 calibration.json，跳过计时校准。重新校准可能改变时间、选档和签名，建议另建运行副本和缓存目录。所有正式推理均按 RGB 哈希、校准和实现摘要复查断点。本次在独立真值审计中修正了一处 PFM 单位读取。repair_cache.py 只服务于本轮旧签名到更正后签名的无损校验迁移；从头运行无需执行。无需再次 GPU 推理即可从原始像素缓存重现全部图表和诊断。仅查看已发布的数字，可以在任意有 Python 和 NumPy 的系统执行 `python account.py`，或直接从 paired_results.csv 按场景复算区间。

## 产物

| 文件 | 内容 |
|---|---|
| expected.csv、freeze.json、inventory.csv、data_audit.json、disparity_units.csv | 预期清单与摘要，高清可用性与缺失原因、官方七个文件校验、逐场景单位和输入哈希 |
| calibration.csv、latencies.csv、latency_summary.csv、worker_timing.csv、gpurun_*.json、accounting.json | 每个校准档、参数/真实网络尺寸、正式图延迟、峰值已分配显存、冷启动分层、UUID、前向及命中数 |
| cache_index.csv、coverage.csv、coverage_summary.csv、edge_diagnostics.csv | 已完成输入、预测和界面缓存的路径/哈希，各步资格与排除计数、宽度和亮度对比 |
| per_image.csv、per_window.csv、paired_results.csv、summary.csv、scale_summary.csv | 每图全部条件的边界/全图指标，裁窗坐标与尺度、各场景配对收益及区间；各字段带 signed 的误差方向是预测减 GT |
| cases.csv、figures/、verification.json | 冻结顺序下的配准图、收益最大/中位/最小案例，独立复核与数值误差 |

每场的 `predictions.npz` 保存原生 NB、降质 LB、四个原生/降质窗 NW/LW，以及原生 H，均已按各自完整源图或窗的尺寸回映。inference.json 保存 xyxy 半开窗、图像/预测哈希和尺度。edges.npz 的 anchor 是交界左侧/上侧像素行优先索引，axis 0/1 为横/纵扫描，near_direction 指向近侧，near/far 各给三个原生索引，span 的单位为默认输入像素，censored 表示至少 24，width 0/1/2 对应其他/窄/宽，jump 使用米。contrast_N/L 是两侧中位 sRGB 加权亮度的绝对差，范围 [0,1]。逐像素缓存不上传公开仓库。
