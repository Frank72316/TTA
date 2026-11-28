# P3 · 按深度分歧分配整批预算

[一页报告](report.md) · [逐帧结果](frame_results.csv) · [冻结清单](frames.csv) · [仅用预测的调度名单](schedule.csv) · [汇总及区间](summary.json)

### 只读结果复算

在仓库根目录运行以下命令。需要 Python 3、NumPy、Matplotlib。输入为 measurements.csv、scenes.csv、schedule.csv、timings.csv 和冻结记录。

```sh
python -B P3/depth_adaptive_budget/analyze.py
```

### 从图像重跑

本次运行复用 P3/depth.py 的读图、模型版本、米制深度和掩膜口径，复用 score_gpu.make_translation 的 3% 整数平移及反射补边。模型为 Depth Anything V2 Metric Indoor Small，revision `8078d68a9c75a972131914f6afd0c1723be0da7f`。启动前，Asus 物理 3 号 RTX 4090 显示显存占用 15 MiB、利用率 0%。使用单图、FP32、eval、禁用 TF32。数据、权重、预处理器与运行环境散列见 [environment.json](environment.json)。以下命令创建独立复跑目录，保留原始计时记录。

```sh
# 从仓库根目录执行
S=P3/depth_adaptive_budget
REPLAY=$(mktemp -d)
git archive HEAD P1/baseline/pilot_io.py P1/baseline/score.py P2/views.py P2/study.py \
    P3/analyze.py P3/analyze_study.py P3/budget.py P3/depth.py P3/score_gpu.py \
    P3/results/depth/depth_moments.json P3/results/depth/depth_environment.json \
    "$S"/*.py "$S/frames.csv" "$S/freeze.json" "$S/scenes.csv" | tar -x -C "$REPLAY"
cd "$REPLAY"
DATA=/home/shenc/tta_depth_adaptive/data
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" "$S/freeze.py" "$DATA/nyu-depth-png.zip"
"$PY" "$S/measure.py" --archive "$DATA/nyu-depth-png.zip" --model "$DATA/model" \
    --split "$DATA/splits.mat" --cache "$REPLAY/pred_cache"
"$PY" "$S/verify_cache.py" --archive "$DATA/nyu-depth-png.zip" --cache "$REPLAY/pred_cache"
python -B "$S/analyze.py"
```

已冻结的帧、场景和模型测量结果足够复算统计。另行复现数据准备时，使用 [原实验 ZIP](https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth-png.zip)、[官方训练/测试索引](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) 和 [固定模型快照](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf/tree/8078d68a9c75a972131914f6afd0c1723be0da7f)。模型目录仅需 config.json、preprocessor_config.json、model.safetensors。机器备有 h5py 时，可用 `python P3/depth_adaptive_budget/scene_metadata.py` 重新读取官方 HDF5 的 scenes 元数据；已有完整 MAT 文件可用 `--mat FILE`。场景为具体拍摄序列，例如 bedroom_0122。压缩包文件名从 0 起算，逐一加 1 后与官方 MATLAB 的全部 654 个 testNdxs 完全一致。

### 调度、评价与核验

1. 以种子 20261128 从排好序的未用帧不放回抽 128 帧。SHA-256 和与旧 64 帧的交集写入 freeze.json。两个坐标测试图（奇偶宽各一）和前两张真实 RGB 的 O/F/L/R 逆对齐像素在共同区域逐一相同。
2. 设 s 为宽度 3% 最近整数，半数向上。所有预测先双三次插值回原尺寸，align_corners=False。L/R 按带符号位移采样回原坐标，F 反向水平翻转，统一保留横坐标 [s, W−s)。完成所有 O/F 后只用预测与几何区域计算 D，以降序选 64 帧，平局按冻结帧 ID 升序。调度文件先于 L/R 推理和真值读取得到。部署时须允许整批图像先完成前两视图。
3. 按 P3 原实现以毫米转米，真值有效范围为 (0,10]。FP32 单视图残差转 float64 再等权平均、取平方，每帧一票。本批 36,986,880 个几何共同像素全部满足深度范围。独立 CPU 核验改用先平均 float64 深度预测再减真值，逐帧 MSE 最大差小于 1×10⁻⁹ m²，D 完全一致；重建的前 64 帧全部相同。见 [cache_audit.json](cache_audit.json)。
4. 场景簇 bootstrap 固定种子 20261128，每次从 95 个场景等概率有放回抽 95 簇，带上全部所属帧，同步比较所有策略，重新排名 D、G。簇大小不同使副本帧数可变；奇数帧时将边界帧以 1/2 概率追加，精确使用该随机边界的期望误差，使平均预算保持每帧三次。真实批次是偶数，严格追加 64 帧。区间使用 2.5%/97.5% 分位。
5. 推理分开记录视图变换、预处理与传输、同步 GPU 前向、输出插值与传回、逆对齐。逐视图计时见 timings.csv，前 4 次用于预热。浮点预测留在 Asus 缓存路径下，由环境指纹分目录，可在 CPU 上运行 verify_cache.py。仓库存放逐帧精确指标、时间及调度记录。
