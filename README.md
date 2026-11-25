# 测试时增强如何改变图像偏好排序

这个项目用 HPS v2.1 给同一提示词下的两张候选图评分，考察亮度、对比度、JPEG 压缩和平移如何改变两图分差，以及这些改判是否与 ImageRewardDB 对原图的人工排名一致。

新加入的队友可以从 [8 页中文交接报告](tta_project_handoff/TTA阶段交接报告.pdf) 开始。第 5–7 页并排展示了真实候选的原图、扰动图和分数。另附[可编辑源稿和重建说明](tta_project_handoff/README.md)。

| 研究阶段 | 样本与问题 | 入口 |
|---|---|---|
| 调亮先导 | 同一批 64 对，比较原始提示词和追加夜景要求时的调亮响应 | [结果](tta_pilot/report.md) · [复现](tta_pilot/README.md) |
| 短提示词复核 | 核查截断和已有光照描述，再用完整的短提示词配对重测 | [结果](tta_round2/report.md) · [复现](tta_round2/README.md) |
| 单图响应分解 | 分开估计两张图共同变化的部分和传到分差的部分 | [结果](tta_round3_analysis/report.md) · [复现](tta_round3_analysis/README.md) |
| 随机重配 | 固定单图响应，检验同一提示词下的候选是否更容易同向变化 | [结果](tta_round4_repairing/report.md) · [复现](tta_round4_repairing/README.md) |
| 增强组合 | 回到原始提示词，按每图三次评分比较同类与跨类增强 | [结果](tta_p2_complementarity/report.md) · [复现](tta_p2_complementarity/README.md) |
| 平移独立复验 | 换用 256 个新提示词，核对额外纠正、改错代价与方向效应 | [结果](tta_p3_translation/report.md) · [复现](tta_p3_translation/README.md) |

前五个阶段使用同一批 64 对图。256 对新图单独报告。数据集人工排名适用于原提示词 P0 和原图，改写的 Q0、Q1 用于分析模型自身的响应。

各轮目录保留冻结清单、原始评分、逐对结果、分析代码和图表。从缓存复算请使用相应 README 中的分析命令。从原图重新推理还需要按先导实验的说明准备公开数据、模型权重和 GPU 环境。
