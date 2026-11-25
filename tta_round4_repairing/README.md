# 第四轮 · 单图响应随机重配对照

[中文报告](report.md) · [对照图](random_pairing.png) · [汇总](summary.csv)

使用同级 tta_round3_analysis 中已保存的 h_A、h_B，并利用 tta_round2 的冻结清单核对原提示词、图像路径、实际文本编码与冲突标记。范围为全部 64 个真实图对，以及预先标记无语义冲突的 58 个图对。每张图的 h 始终在它自己的 Q0、Q1 与亮度条件下计算。

## 复跑

在 Keqing 的 `/home/data/users/shenc/zhangyc/tta_round4_repairing` 中执行。只需 Python、NumPy、Matplotlib 和中文字体。实际数值运行使用系统 Python 3.12、NumPy 1.26.4；制图使用 Matplotlib 3.6.3。脚本仅读取前轮 CSV。随机种子 42，每组每种抽样方案各重复 1,000,000 次。

```bash
cd /home/data/users/shenc/zhangyc/tta_round4_repairing
/usr/bin/python3 -B analyze.py
/usr/bin/python3 -B present.py
```

## 比较对象

- **derangement** 为主对照。固定 A，对 B 做均匀一对一置换，拒绝任何保留原位的整次置换，直至收集满指定次数。每次严格跨提示词，每张 B 恰好使用一次，同时保留两侧的均值、方差、符号与零值。去掉原配对后，期望乘积和等于所有非对角乘积之和除以 n−1，因为每个允许的 B 在该位置出现的概率为 1/(n−1)。
- **unrestricted** 用于补充的可交换性检验。保留相同边际响应，在全部 B 置换中抽样，允许部分 B 回到原位置。其期望乘积和为 Σa × Σb / n。正式单侧 Monte Carlo p 使用 (达到或超过实测的次数 + 1) / (模拟次数 + 1)。主对照报告命中计数；0 次命中不能确定精确尾概率。
- **R** 是共同平方和占比。**same_nonzero** 是乘积大于零的图对数，同正、同负均计入，零值保留在分母。符号计数的理论均值可将上面的乘积更换为“同向非零”指示变量，分别在相应的候选组合上求和。
- 程序将保存的二进制有理数精确转换到共同的整数网格，以点积比较尾部命中，保留并列，无随机扰动。程序同时复核每个 h 的原始八分数来源、模拟均值与理论均值是否吻合，以及各直方图频数是否加总至 1,000,000。

## 数据文件

| 文件 | 内容 |
|---|---|
| [inputs.json](inputs.json) | 输入 SHA-256、随机数发生器、随机种子、批大小和重复次数 |
| [summary.csv](summary.csv) | 实测值、解析期望、模拟均值、超出期望的量、经验分位数、单侧命中与并列次数 |
| [null_distribution.csv](null_distribution.csv) | 按样本组、重配方案及指标保存所有离散取值及其频数，可直接复核区间、尾数、绘图 |
| [mean_decomposition.csv](mean_decomposition.csv) | 两侧平均 h、共同占比的均值项与去均值项、同向计数和 Pearson 相关 |

随机分布的中央 95% 是重配参照区间。报告用它说明真实组合与同批边际响应重配的差距，结论仅涉及本批图片的额外配对关联。

