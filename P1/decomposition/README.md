# P1 单图响应分解

[中文报告](report.md) · [逐对分解](pair_decomposition.csv) · [汇总数据](summary.csv) · [单图交互图](figure1_interactions.png) · [改选边界图](figure2_choice_boundaries.png)

从仓库根目录进入 `P1/decomposition`。输入位于同级的 `prompts/`。代码仅依赖 Python 标准库、NumPy 和 Matplotlib。

## 复跑

实际使用 Keqing 的系统 Python 3.12.3、NumPy 1.26.4、Matplotlib 3.6.3，在 CPU 上完成。中文图表使用系统 Noto Sans CJK 字体。保持现有目录结构，执行以下命令即可重建两个 CSV、两张 PNG 和中文报告。

```bash
cd P1/decomposition
CUDA_VISIBLE_DEVICES="" /usr/bin/python3 -B analyze.py
```

[分析脚本](analyze.py)负责读取、校验和算术；[图表与报告脚本](present.py)由同一入口调用。运行只写本目录中的本轮产物。没有模型、图片、联网服务或人工标注依赖。

## 输入与溯源

| 短提示词实验文件 | 用途 | SHA-256 |
|---|---|---|
| scores.csv | 全部统计的原始输入 | 39159d9a93bff42222bdb4c7024307db3f0eba827955ebc42bb0c75cdc9a4345 |
| prompts.csv | 冻结的提示词、A/B 路径与无冲突样本标记 | 0bad2a1b662c1ee23358f57c1812c6bb38b7fd80f594e2483ac3b96c468b8429 |
| pair_results.csv | 交叉核查原分差、调亮分差、TTA 分差、模型选择、I 和 G | eda98e8eeb13cacccc01ca556cdd43370887d075ef85aa9b508cd67e79b4c616 |
| summary.csv | 核对全体和无语义冲突组的 G 统计 | d3076d69a55be4726129efb7d01ce4ce778a54106b169a87fe12ab0a3a8a0399 |
| score_new.py | 实际推理时的源码摘要；目录迁移的对应摘要见 baseline/layout_provenance.py | f35498e71332d1eabc571d0c574411c04e9c48e27c315300c2a89169ca97174f |

运行时也读取短提示词实验 environment.json，复核模型版本、调亮系数以及评分脚本和冻结提示词的哈希。精度核查另阅读了上一轮 README、smoke.log、full.log 和服务器已有的官方 img_score.py 源码，源码哈希与 environment.json 记录相符。

## 字段与统计口径

- pair_decomposition.csv 中 A_Q0_L0 等八列为原始分数，b_A0、b_A1 是 A 在基础、夜景提示词下各自的调亮变化；B 同理。h_A、h_B 是同一候选两种亮度响应之差。delta0、delta1 分别是同一提示词下 A、B 的亮度响应之差。
- I = delta1 − delta0，G = abs(delta1) − abs(delta0)，C = (h_A+h_B)/2，D = (h_A−h_B)/2。C_sq、D_sq、total_sq 分别是共同分量平方、差异分量平方及两者之和。total_amplitude 为二者平方和的平方根，也是两张图 h 的均方根。max_ulp_A/B 取该候选四个原始 float16 分数的最大局部可表示步长，仅用于量级参照。
- Q0/Q1 的 d0、d1、d_tta 分别为原图、调亮、等权均值的 A 减 B。lead 是原领先幅度；toward 是朝向平局的亮度变化，原图同分时留空。两个终点分六类，原图同分、零响应、同向扩大、反向未过零、恰好打平、严格翻转。若原图同分且变化为零，优先归入原图同分；本批没有这样的重叠。每个分区之和等于组内图对数。
- R_without_pair_all 是在全体中移除本行后的共同平方和占比；R_without_pair_no_conflict 针对 58 对复核组，组外记 NA。分母为零时 R 记 NA。留一保留完整 A/B 配对。
- summary.csv 为 scope、metric、value 长表。all、no_conflict 分别表示 64 对主结果、58 对预标记子集。raw_scores 下 adjacent_gap_k_steps 是排序后的相邻唯一分数间隔恰为 k 倍最小观测间隔的数量；binary16_ulp 各行给出对应局部可表示步长的原始评分条数。loo_min_ids 与 loo_max_ids 保留所有并列 ID。
- 计算采用 binary64，仅在输出精度校验时将原始分数往返转换为 float16。CSV 用足以回读原值的十进制，分位数按线性插值，计数以精确零为准。表中普通数字为展示做了舍入。
- 案例分别取最大绝对 C、最大绝对 D，以及总变化幅度最接近中位数者；并列按图对 ID 升序。偶数样本中，两项中央顺序统计量到其中点等距，代码直接按名次找齐并列候选。本批中位数案例的并列 ID 为 017、019、060，选 017。

