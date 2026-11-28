# P2 · 对应部位窗是否更能利用颜色词？

[中文报告](report.md) · [预设协议](protocol.md) · [逐题连续分差和实际选择](questions.csv) · [双题图对均值](pairs_results.csv) · [最大与最小案例](cases.jpg)

上游代码、清单和真颜色评分来自提交 `3cafdb28e17aa7d51197c520885423f010783746`。[源文件哈希](source.json)用于检查仓库及 Asus 的文件内容。[新评分](neutral_scores.csv)逐行保存原生 FP16 可精确表示的十进制分数、输入张量和文本 token 的 SHA-256。每行可用 `(image_id,view,query)` 连接[旧评分](../part_conditioned_tta/scores.csv)。列 `view` 为 original、crown、wing。评分使用原 CUB JPEG 和项目验证预处理，对新生成的张量先核验旧哈希，再共享视觉特征评分。高精度运算和统计都在 CPU 完成。

[连续量](means.csv)和[严格判定计数](decision_counts.csv)分别按全体、鸟冠题、翅膀题列出。`matched` 为问题对应窗，`swapped` 为另一个窗，`crop` 只看局部，`tta` 为原图和局部窗等权平均。`d` 为原颜色标签赢家的有向分差，`adv` 为同一提示下的对应窗分差减交换窗分差。[转移表](transitions.csv)中 before 是删除颜色词，after 是加回原颜色词，agree、opposite、tie 分别表示与原颜色标签一致、相反、精确同分。两问各自独立选图。[主指标与验收](summary.json) · [两对冒烟检查](smoke_check.json) · [有效文本 token](tokens.json)

### 从仓库缓存复算

从仓库根目录执行。Python 3.10+ 和 NumPy 即可；输出写入单独目录。图像显示脚本另需 Pillow 和 DejaVu Sans 字体。

```sh
S=P2/part_color_word_control
OUT=$(mktemp -d)
cp "$S/neutral_scores.csv" "$OUT/"
python -B "$S/analyze.py" --out "$OUT"
```

### 在 Asus 重跑新增评分

使用上游 CUB 实验的原照片、权重与运行环境。以下 UUID 是本次使用的物理 3 号卡，启动前可用 nvidia-smi 查询。程序逐张保存断点。同一份 OUT 可重复执行正式阶段；如需重新计时，使用新的 OUT。实际运行的 OUT 为 Asus 上的本实验目录。

```sh
S=P2/part_color_word_control
BASE=/home/shenc/tta_geometry_specificity_asus
DATA=/home/shenc/tta_prompt_flip/data/cub_part_96
PY="$BASE/runtime/bin/python"
OUT=$(mktemp -d)
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" "$S/score_neutral.py" --data "$DATA" --pilot "$BASE/inputs/tta_pilot" --out "$OUT" --smoke
cat "$OUT/smoke_check.json"
"$PY" "$S/score_neutral.py" --data "$DATA" --pilot "$BASE/inputs/tta_pilot" --out "$OUT"
"$PY" "$S/analyze.py" --out "$OUT"
"$PY" "$S/examples.py" --out "$OUT"
```

[实际运行环境与权重](environment.json) · [冒烟计时](smoke_run.json) · [续跑计时](run.json)。两轮分别对同两句无颜色提示做文本编码。重画案例使用本轮 `display_cache/` 中反归一化的模型输入，原始 CUB 数据沿用[上游数据说明](../part_conditioned_tta/README.md)。
