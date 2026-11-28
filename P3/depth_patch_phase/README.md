# 模型输入网格相位与三视图深度配对

[中文报告](report.md) · [预注册预测表及判据](protocol.md) · [32 场景清单](frames.csv) · [结果图](pairing_results.png)

## 输入与样本

本轮在 `9b789d5` 的项目基础上新增。冻结清单只使用既往 128 帧的 [路径](../depth_adaptive_budget/frames.csv) 和 [场景](../depth_adaptive_budget/scenes.csv)，选择过程不加载预测或真值。源模型、权重、处理器和 NYU ZIP 与原深度实验的 [环境摘要](../depth_adaptive_budget/environment.json) 对账。阶段使用既往开发场景，每个具体场景一帧。32 行便是 32 个独立重采样单元。复核了实际 patch 投影模块、输出 token 网格及预测尺寸。

每张 RGB 经官方处理器一次得到 518×686 的归一化张量；中央画布左右各留 42 列作为可见外部上下文。平移在张量上按整数重排，无再次图像插值。两臂各在相同的 518×602 上推理，O 共享。真实边条臂的“右移 d”通过从父图横坐标 42−d 开始取 602 列实现。反射臂取中央画布索引 reflect(x−d)。二者在保留的内容区按位相同。逐视图记录新露出面积、不同像素数和 SHA-256。图片以外上下文在真实单图部署时是否可用，取决于采集或裁窗方式。

共同可见的模型输入坐标为中心画布 [31,571)。逆对齐后用父图坐标一次 bicubic 映回 480×640；源图评价范围 x∈[70,570)。实测 bicubic 横向取样支撑在父图第 74–611 列内（精确值见 smoke_proof.json）。真值保持原始整数毫米，先对相同有效像素求窗均值；门槛按整数交叉乘积判定。旧的区域窗口要求用于本轮的新视野；旧的原图和 2%/4% 平移预测均未复用。

## 文件与复现

| 文件 | 包含的信息 |
|---|---|
| freeze.json、frames.csv、protocol.md | 冻结的文件哈希、预测矩阵、抽样顺序、裁剪及统计规则 |
| environment.json、runs.json、view_audit.csv | 权重、模型精度、GPU UUID、逐视图输入与逆对齐结果哈希、前向耗时和断点命中 |
| windows.csv、region_pairs.csv | 原图坐标的方窗、有效像素、整数毫米和、17 个窗口均深；无序区域对真值差及主分析资格 |
| frame_results.csv、summary.csv | 场景×填边×幅度组合的各配方 MSE、有向差及恒等式；场景等权均值和场景配对区间 |
| smoke_proof.json、smoke_regions.json、validation.json、replay.json | 坐标测试、跨像素平均检查、原始区域对上的乘积核验、独立重新聚合核验 |
| prediction_files.csv | Asus 上 32 个逐帧 NPZ 的 SHA-256，内含 544 个有效区域 FP32 数组 |

仅复算结果时，在仓库根目录执行。统计只需 Python 3.10+、NumPy；绘图另用 Matplotlib，中文图使用微软雅黑或 Noto Sans CJK。所有汇总均可从窗口表重建。

```bash
python -B P3/depth_patch_phase/freeze.py
python -B P3/depth_patch_phase/analyze.py
python -B P3/depth_patch_phase/verify.py
python -B P3/depth_patch_phase/plot.py
```

在 Asus 的完整仓库中重新推理，先完成上述冻结步骤中的 freeze.py。已发布的真实运行目录是 `/home/shenc/tta_depth_adaptive/P3/depth_patch_phase`，大数组位于其 `predictions/<frame_id>.npz`，约 500 MiB。以下示例在新的输出目录运行，前两帧的 34 个预测只计算一次。脚本从代码目录读取已冻结文件。先推理、检验像素与取窗，再进行全量前向和离线评价。

```bash
S=P3/depth_patch_phase
DATA=/home/shenc/tta_depth_adaptive/data
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
OUT=$(mktemp -d)
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/run.py" --archive "$DATA/nyu-depth-png.zip" --model "$DATA/model" --out "$OUT" --smoke
"$PY" "$S/windows.py" --archive "$DATA/nyu-depth-png.zip" --out "$OUT" --smoke
"$PY" "$S/run.py" --archive "$DATA/nyu-depth-png.zip" --model "$DATA/model" --out "$OUT"
"$PY" "$S/windows.py" --archive "$DATA/nyu-depth-png.zip" --out "$OUT"
"$PY" "$S/analyze.py" --out "$OUT"
```

[模型仓库配置](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf/blob/8078d68a9c75a972131914f6afd0c1723be0da7f/config.json) 给出分块大小，运行时以实际卷积核、步长及特征图维度复核。2026 年的工作已在稠密预测中研究多相位平均。本实验检验提前指定的四种位移搭配能否预测哪一方向 MSE 更低。[Phase Marginalization for Patch-Grid Instability in Vision Transformers](https://arxiv.org/abs/2606.08132)
