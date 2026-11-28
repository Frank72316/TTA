# 复算光度配方

从仓库根目录运行 `python -B P2/photometric_pairing/analyze.py`，仅需 Python、NumPy。读取 translation 的冻结清单及 scores.csv（原生 FP16 转 float64）；从 param 字段识别幅度，逐图核验 P0、A/B、图像 ID 与名次方向。四个单视图在两种方案中的边际频率均为 1/2，每图部署三次评分。输出 [逐对量](pair_results.csv)、[九格转移](transitions.csv) 和 [区间](summary.json)。[环境、输入哈希及冒烟 ID](environment.json) · [预设规则](protocol.md) · [中文报告](report.md)
