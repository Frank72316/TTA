# P2 · 用相反提示的分差读取左右关系

[一页中文报告（PDF）](report.pdf) · [图文报告](report.md) · [64 对的分数、分差与选择](pair_results.csv) · [128 张照片的镜像对照](mirror_results.csv) · [原先选同图的逐组 ID 清单](same_choice_groups.csv)

本实验复用相邻目录中已冻结的 64 对自然照片和 512 条 HPS v2.1 原生评分。全程在 CPU 上读缓存，新增模型评分 0 条。[上轮清单](../prompt_aware_flip/pairs.csv) · [COCO 实例框导出](../prompt_aware_flip/images.csv) · [评分输入](../prompt_aware_flip/scores.csv) · [本次汇总及输入哈希](summary.json)

## 从仓库复算

从本目录执行，仅使用 Python 3.10+ 标准库。第一次命令依据仓库内保存的原始 COCO 框坐标及 HPS 输入框恢复真值，再核验上一轮 384 个逐题选择。第二次命令生成带精确分数的图文报告。四张案例原图已随目录保存。

```bash
python -B analyze.py
python -B report.py
```

正式运行额外核对了本地 COCO 原始 JSON 的 SHA-256 及 256 个实例的图像 ID、类别、非 crowd 标志和框坐标。下面是带原始标注复核和重建 PDF 的命令。Chrome 或 Edge 只用于排版，命令行已关闭 GPU。

```bash
python -B analyze.py --annotations "/path/to/instances_val2017.json"
python -B report.py --browser "/path/to/chrome"
```

## 表格读法

- **自然图对。** X 和 Y 的水平方向由标注框判定，A/B 保留上轮随机顺序。先按两张图、两条提示、两个视图核对缓存的唯一键，再分别复现原图、只翻图、图文同步三种旧策略。新规则在每张原图上作两提示之差，按 e 的大小同时分配两道题。若两个 e 相同，两种分配同分。M 取真实左图的 e 减真实右图的 e。原生 FP16 值转为 Python binary64 计算，再逐对用精确有理数检验 M，CSV 存储完整数值。
- **参考水平。** 已知恰有一左一右，随机猜测有两种分配，命中率为 50%。符号检验在 64 对自然图对上计正负号，剔除精确零，双侧 p 值使用二项分布两侧尾概率。四分位数使用线性插值（含两端点）。旧策略独立回答两道题，可以两次选同图，以前的双题全对与新规则的正确分配使用同一真值，决策约束不同。
- **同图镜像。** 每张照片各做两次原图与自身镜像的直接比较，提示不变。若标注中的 X 在 Y 左边，左题答案是原图、右题答案是镜像，反之亦然。任一题同分，整张照片只计入“含同分”。其余按严格答对两题、一题、零题划分。119 张仅答对一题的照片在两题都严格选择了同一版本。未对 128 张照片作独立抽样推断。
- **“原来始终选同图”。** `original_same` 表示原图策略在两题选同图。`each_strategy_same` 表示三种旧策略各自在两题选同图，允许策略之间选的照片不同。`all_six_same` 要求三种旧策略六次都选中同一候选。各口径分对、选反、同分的全部 ID 和计数都在分组 CSV 中。

## 极端案例与照片来源

案例只按 M 的全体最大、最小值选取。若同一极值有多对，按图对 ID 取第一对并在 summary.json 保留全部并列 ID。本次两个极值各有一对。以下四张 JPEG 与冻结清单的 SHA-256 逐字节一致。报告只缩放显示完整原图。许可和来源取自 COCO 图像元数据，原图保留各自许可。冻结标注与本地 JPEG 未提供可核实的作者姓名。[上轮框坐标](../prompt_aware_flip/images.csv)

| 图对／候选 | COCO ID | 照片来源 | COCO 记录的许可 |
|---|---:|---|---|
| F061 A | 492758 | [COCO 原图](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000492758.jpg) · [Flickr 3966060902](https://www.flickr.com/photo.gne?id=3966060902) | [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) |
| F061 B | 559543 | [COCO 原图](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000559543.jpg) · [Flickr 3167056823](https://www.flickr.com/photo.gne?id=3167056823) | [CC BY-NC-ND 2.0](https://creativecommons.org/licenses/by-nc-nd/2.0/) |
| F020 A | 15335 | [COCO 原图](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000015335.jpg) · [Flickr 10257288534](https://www.flickr.com/photo.gne?id=10257288534) | [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) |
| F020 B | 278463 | [COCO 原图](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000278463.jpg) · [Flickr 1280232774](https://www.flickr.com/photo.gne?id=1280232774) | [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) |
