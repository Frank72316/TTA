# P2 开发实验 · 平移与中心缩放

[正式中文报告](report.md) · [历史七例追踪表](dev256/seven_cases.csv) · [开发集总体比较](dev256/full_comparison.csv) · [稳定纠正与改错清单](dev256/full_strategy_summary.csv) · [逐图评分](dev256/scores.csv) · [案例图](figures/cases.jpg)

目标是用 HPS v2.1 的原始提示词评分选回 ImageRewardDB 人工偏好的原图。64 对用于实现与强度校准，256 对用于开发分析。两批独立计算，全程只读取 validation 的原图和已有人工排名。

## 七对案例与参数

[入选时冻结的七对](historical_seven.csv) 来自 P2/translation 的三次预算，条件为原图选错、BB／CC／BC 均为零次纠正、BT 至少一次纠正。TT 也能纠正七对。原始分数独立枚举，并与历史逐对策略表核对。[P3 逐对预算结果的对应行](historical_p3_budget.csv) 来自仓库提交 df16f46 的 P3/results/development-3/pair_results.csv，K=1、2、4、8，与上述三次预算属于不同设计；两者使用同一批 256 个原始提示词和 A/B。

光度使用 Pillow Brightness／Contrast 的 1 ± {0.0125, 0.025, 0.0375, 0.05}。平移为左右上下各 1%／3%，最近整像素、半数向上、无插值反射填充。中心缩放倍率为 1 ± {0.0075, 0.015, 0.0225, 0.03}。保持原画布，围绕画布中心做逆仿射映射，Pillow 双三次插值，先在 RGB 原图上以 NumPy reflect 扩边。所有变换均独立生成，随后通过同一官方验证预处理。冒烟用奇偶两种尺寸的坐标图检验映射，还显式核验首两对实际模型输入。[真实输入抽查图](qa/smoke_model_inputs.jpg)

## 运行

评分使用先导实验已安装的 HPS 环境、两个固定权重文件和 validation 压缩包。生产参数及环境保存在各批次内。下面的命令从本目录执行，将 PILOT 设置为含 data/、environment.json 与 .venv/ 的先导目录。DEV_PAIRS 是 P2/translation/pairs_frozen.csv。

```bash
python -B history.py
# 如需同时核验 P3 八方向记录，在包含对应提交的仓库内执行
python -B history.py --p3 ../../P3/results/development-3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=GPU-d329c8d0-f276-5f23-03a9-df221a2021ed
PY="$PILOT/.venv/bin/python"
"$PY" -Bu run.py --pilot "$PILOT" --old "$PILOT/pairs.csv" --dev "$DEV_PAIRS" --out "$OUT" --device cuda --batch 1 --workers 1
"$PY" -B cases.py --pilot "$PILOT" --pairs "$DEV_PAIRS" --root "$OUT"
"$PY" -B verify.py --root "$OUT"
"$PY" -B report.py --root "$OUT"
```

Asus 正式运行使用物理 2 号 RTX 4090，数据版本 c493721e19e296eb615420036f2a2eed08412bb4。共享先导目录提供带哈希的模型权重、验证集 ZIP 和已安装的 HPS 包。两台服务器的同名账号数值 UID 不同，因此在 Asus 私有运行目录中使用经 SHA-256 对账的两个清单副本，以及原先导 Python 3.10 环境的独立副本。实际入口如下。

```bash
cd /home/data/users/shenc/zhangyc/P2/geometry_specificity
work=/home/shenc/tta_geometry_specificity_asus
export GPU_UUID=GPU-d329c8d0-f276-5f23-03a9-df221a2021ed
export GEOMETRY_PYTHON="$work/runtime/bin/python" GEOMETRY_OUTPUT="$work/production"
export GEOMETRY_PILOT="$work/inputs/tta_pilot" GEOMETRY_DEV_PAIRS="$work/inputs/pairs_frozen.csv"
bash run_asus.sh
```

本地仓库用 history.py 从旧缓存冻结七例，再把清单与脚本传到评分机。run.py 各批先做两对冒烟，再恢复逐对断点。每张图固定 33 条评分，实际输入张量相同时始终只进行一次图像前向，同提示词的 A/B 共享文本特征。FP16 GPU 分数和早期 FP32 CPU 探查分开留档。

离线分析可直接读逐图 CSV，机器上只需要 Python、NumPy 和 Pillow。示例中 OUT 为本文目录下的生产结果根目录或复现时的新目录。

```bash
python -B analyze.py freeze --scores "$OUT/old64/scores.csv" --out "$OUT"
python -B analyze.py analyze --scores "$OUT/old64/scores.csv" --n 64 --out "$OUT/old64"
python -B analyze.py analyze --scores "$OUT/dev256/scores.csv" --n 256 --freeze "$OUT/strength_freeze.json" --out "$OUT/dev256"
python -B verify.py --root "$OUT"
python -B report.py --root "$OUT"
```

## 统计口径

- 人工赢家分数减另一侧分数为有向分差。float64 对同一套版本的分差求和，严格大于、等于、小于零分别计选对、同分、选错。对搭配的判定均匀平均。可达与稳定纠正的分母是本轮原图选错组，稳定门槛用整数比较 `4 × 选对套数 ≥ 搭配套数`。稳定改错在原图正确组用严格选错数做相同计算。
- 三档预算的纯策略分别为 28、70、1 套，TS 混合分别为 64、784、4,900 套。比较 T−S、TS−T、TS−S、B−C。每个预算以 10,000 次、种子 20261128 的完整提示词配对 bootstrap 给出严格选对率差的 95% 百分位区间。
- 旧 64 对每个版本先取绝对分差位移中位数，并在双方图像上取模型输入平均像素绝对变化的中位数。取 T、S 位移中位数范围的交集，在区间内保留尽可能多的不同版本，以总绝对位移差最小匹配，相同代价按版本名稳定处理。冻结校准表与哈希后再评分 256 对。若池大小不足以支持原预算，仅报告可以等预算比较的档位；只剩一个版本时用 O＋一档比较。
- 案例图按 K=9 全池纯策略确定类别，取最小图对 ID。展示原图和每种方法中最支持案例判定的单视图，分别给出单视图和全池策略的分差；展示图逐张校验输入哈希。颜色与补边按 HPS 官方归一化参数反变换，输出缩略图。
