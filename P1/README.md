# P1 · 亮度增强与提示词交互

同一对候选都调亮 10% 后，HPS 的选择是否改变？追加夜景要求后，两张图的亮度响应是否更容易一起变化？以下四项分析共用 64 对图。

| 子目录 | 内容 | 实验记录 |
|---|---|---|
| [baseline](baseline/README.md) | 原提示词 P0、追加夜景的提示词 P1，比较原图、调亮图和等权均值 | [报告](baseline/report.md) · [冻结图对](baseline/pairs.csv) |
| [prompts](prompts/README.md) | 核查光照词和截断，使用短提示词 Q0、Q1 再评分 | [报告](prompts/report.md) · [原始评分](prompts/scores.csv) |
| [decomposition](decomposition/README.md) | 离线分解候选的共同变化与差异变化 | [报告](decomposition/report.md) · [逐对结果](decomposition/pair_decomposition.csv) |
| [pairing](pairing/README.md) | 固定单图输出，跨提示词重配，检验同一对候选的额外同步性 | [报告](pairing/report.md) · [汇总](pairing/summary.csv) |

目录中的 P1 是实验编号。评分表中的提示词条件 P1 是追加夜景要求后的原提示词。人工偏好标签仅用于原图及其原始提示词 P0。
