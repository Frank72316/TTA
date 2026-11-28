# P2 · 分开增强和叠加增强

[中文结果](report.md)｜[逐图评分](scores.csv)｜[逐搭配分差和判定](combinations.csv)｜[逐对概率](pair_results.csv)｜[统计](summary.json)｜[案例及原提示词](cases.csv)

从仓库根目录运行。已提交的评分足以在 CPU 复算结果，依赖 Python 3、NumPy 和仓库中的只读脚本。

```sh
python -B P2/composed_vs_separate/analyze.py
```

Asus 推理依赖沿用 geometry_specificity 的运行环境、模型缓存和 ImageRewardDB 原始验证图像包。以下命令中的两个根目录指向本次实跑的环境和输入，仓库相对路径指向对应脚本。

```sh
export CUDA_VISIBLE_DEVICES=GPU-d329c8d0-f276-5f23-03a9-df221a2021ed
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ROOT=/home/shenc/tta_geometry_specificity_asus
PY="$ROOT/runtime/bin/python"
S=P2/composed_vs_separate
# Fresh writable output for an independent run. The full pass resumes the smoke checkpoint.
OUT="$PWD/composed-replay"
for n in 2 256; do
  "$PY" "$S/inference.py" --pilot "$ROOT/inputs/tta_pilot" --prior "$ROOT/production/dev256" \
    --pairs "$ROOT/inputs/pairs_frozen.csv" --out "$OUT" --limit "$n" || exit 1
done
cmp "$OUT/scores.csv" "$S/scores.csv"
# Recreate the displayed cases from the checked-in run, on CPU.
CUDA_VISIBLE_DEVICES="" "$PY" "$S/cases.py" --pilot "$ROOT/inputs/tta_pilot" --pairs "$ROOT/inputs/pairs_frozen.csv"
```

亮度在 RGB 原图上由 Pillow 实现，随后按原宽高 3% 四舍五入到最近整像素（半数向上），反射补边，不插值。两对冒烟重新计算全部 60 条，32 个复合像素阵列与独立坐标索引逐像素一致，28 条原图/单增强评分与历史值逐条相等。正式阶段复用单增强前校验 P0、图像 ID、A/B、人工赢家、参数、权重及打分实现、环境版本、原始与变换 RGB、有效文本 token 及最终张量哈希。模型处于 eval，单图单批 CUDA autocast FP16。分数按原生值保存；CPU 枚举使用 float64 和严格符号。设备、依赖、权重及运行计数见 [environment.json](environment.json) 和 [runs.json](runs.json)。两阶段时长含各自初始化，视觉评分计时含设备回传。
