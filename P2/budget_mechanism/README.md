# P2 · 预算如何改变亮度与对比度的纠错覆盖

[中文报告与两张图](report.md) 根据同一批 256 个 P0 提示词的冻结评分，复算全部四格机制对照。每个提示词一对图，A/B 名次来自人对原图的排序。计算全程离线，在 CPU 上枚举有限参数池，新增模型评分为 0。

## 复现

在仓库根目录依次运行，实际使用 Python 3.13.9、NumPy 2.3.5、Matplotlib 3.10.7 和 Microsoft YaHei。Linux 绘图可使用 Noto Sans CJK SC。第三条独立复核只用 Python 标准库。

```bash
python -B P2/budget_mechanism/analyze.py
python -B P2/budget_mechanism/present.py
python -B P2/budget_mechanism/verify.py
```

实际 Windows 命令将上面的 `python` 换为 `py`。分析和绘图只在当前实验目录写出 CSV、PNG 和报告。历史输入在 [translation](../translation/README.md) 下。逐图缓存 `scores.csv` 有 6,656 行，本次选取 O、B、C 共 4,608 行，按 pair_id、side、view 查找；从 param 读取因子，并核对文本、token、图像 ID、人工赢家和原生 FP16 精度。`pairs_frozen.csv` 保留原始 A/B 对应关系。

[reference.csv](reference.csv) 是上一份关系表中 n=256、keep=B、replace=C 的两个预算行的相关字段快照，原表 SHA-256 为 `7852597136a49807c548ec1b17eb63ce07d8a8fb358405cbe6295394354724a9`。如本地有 `P2/method_relations/cached_relations.csv`，脚本还会逐字段校对快照。原始分数和清单的哈希由独立复核脚本验证。

## 对照表怎么读

`views` 只数增强图，`weight` 为原图权重。只有“两档、1/3”和“四档、1/5”两格是原始等权策略。四格均先作每个组合的决策，再对组合等权平均。同分单列。原图选错组的选对概率之和是期望纠正数；原图选对组的严格选错概率之和是期望改错数。严格选对率分母始终为 256。

| 产物 | 内容 |
|---|---|
| [parameter_margins.csv](parameter_margins.csv) | 256 × 9 个原图／逐参数有向分差，附双方原始分数、名次、提示词、图像标识和相对原图位移 |
| [case_migration.csv](case_migration.csv) | 曾被任一等权策略纠正的全部七对，含 K=3、5 的判定计数、概率和可达策略 |
| [parameter_chains.csv](parameter_chains.csv) | N078、N137 各族六种两档搭配，加上剩余每一档的单独影响与五次最终均值 |
| [pair_probabilities.csv](pair_probabilities.csv) | 每对 × 四格 × 三策略，共 3,072 行，均给出选对、同分和选错的计数与概率 |
| [factorial_summary.csv](factorial_summary.csv)、[coverage.csv](coverage.csv) | 12 个策略格的严格选对率、纠正／改错／同分转移，以及两种覆盖阈值下的全部样本 ID |
| [complement_checks.csv](complement_checks.csv)、[identity_summary.csv](identity_summary.csv) | 4,608 对互补组合的两侧分差、两种纯策略、理论目标、数值残差与条件适用性，以及三项验收汇总 |
| [mixed_cases.csv](mixed_cases.csv) | 三个混合独有样本的分差范围、三种判定数、选对概率和一套成功方案及其互补方案 |
| [参数分差图](figures/parameter_margins.png)、[混合分差图](figures/mixed_margins.png) | 七个案例中的两个预算迁移实例与三个混合独有实例，图中 u = 1/8192 HPS 分 |

运行验收通过。独立实现将原始十进制评分无损换算为整数，分别加权两侧后比较，核对了全部 33,792 次组合判定，并与旧 P2 的 768 个三次预算逐对策略结果对账。图对是解释单位；遍历搭配得到的是此参数池上的精确概率。
