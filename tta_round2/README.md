# 第二轮 · 亮度增强与提示词交互

[中文报告](report.md)　[分组统计](summary.csv)　[人工盲评](human_eval/index.html)

本轮工作目录为 Keqing 的 `/home/data/users/shenc/zhangyc/tta_round2`，本地副本位于 `C:\Users\abcde\Desktop\TTA\tta_round2`。与上一轮 tta_pilot 并列。服务器保存原始验证集和权重。本地副本包含所有分析结果、图片、盲评材料及每对评分的断点文件。

## 实验输入与环境

- 只读输入包括上一轮的 pairs.csv、scores.csv、pair_results.csv、lighting.csv、README.md、report.md、environment.json，以及两个 validation ZIP。原图 A/B 路径和数据提交由原抽样清单确定。A/B 实验均直接从旧原始分数重算；旧逐对结果仅用于交叉核查。
- 复用上一轮的 Python 3.10.20 venv，PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124、Pillow 12.2.0 和 HPSv2 1.2.0。调用 `hpsv2.score(..., hps_version="v2.1")`，校验已安装评分源码、HPS 和 CLIP 权重哈希、数据 ZIP 哈希。实际设备为物理 GPU 0，RTX 4080 SUPER，驱动 570.133.20。源代码与各版本见 [environment.json](environment.json)。统计使用 NumPy，制图使用 Matplotlib 和系统 Noto CJK 字体。
- HPS 验证预处理与 eval 模式、固定权重、CUDA AMP 原生 float16 输出；CSV 保存可无损读回的原值，推导量用 float64。GPU 前向时间由 CUDA event 计量，API 同步墙钟与评分阶段总墙钟分列在 [runs.json](runs.json)。
- 从 ZIP 解码 RGB 原图，Pillow 将亮度乘 1.1 后直接送入验证变换。两个候选和两种文本用相同亮度系数。冒烟检查逐侧复核像素、验证预处理张量及文本编码，记录在 [smoke_checks.json](smoke_checks.json)。
- 全部 512 个新分数实际推理。每对的 8 个分数作为一个事务保存到 cache/，缓存绑定冻结文本、权重、图像、程序与软件版本。本次续跑读取冒烟的 16 条记录，另产生 496 条。前一轮的分数缓存命中为 0。

## 文件与计算口径

| 文件 | 可复核的内容 |
|---|---|
| [rewrites.tsv](rewrites.tsv)、[prompts_round2.csv](prompts_round2.csv)、[freeze.json](freeze.json) | 64 组逐条英文改写、被删的光照或色调、语义冲突原因、实际 token 长度、token 与清单哈希，以及预先固定的统计和选例规则 |
| [old_audit.csv](old_audit.csv) | 每对一行，两种旧提示词各有四个原始分数、两张图各自的亮度漂移、共同与差异漂移、分差、选择、翻转、分区和 r；原图同分时 r 留空 |
| [old_prompt_audit.csv](old_prompt_audit.csv) | 原 P0/P1、完整长度、实际长度、非 padding token ID、相同输入标记、旧光照标记与依据 |
| [scores_round2.csv](scores_round2.csv) | 每行一个原生模型分数，四列唯一键为 pair_id、side、prompt_condition、light；记录原图路径、亮度系数、token 哈希及来源 |
| [pair_results_round2.csv](pair_results_round2.csv) | Q0、Q1 的同套逐对模型指标，I、交互绝对值、G，以及冻结时确定的语义冲突标记 |
| [summary.csv](summary.csv) | 旧全体、三个互斥文本子组、截断中完全相同的嵌套子组，以及新全体与无冲突子集 |
| [figures](figures/)、[cases.csv](cases.csv) | 两张分析图和六个案例的冻结选取规则所对应的选择、图像和排名数值 |
| [human_eval](human_eval/) | 量表页面、无损 PNG、空白 CSV 及分析者使用的独立私有映射表 |
| [verify.py](verify.py)、[验收日志](logs/verification.txt) | 从原始 CSV 独立复算分差、配对 bootstrap、条件分区，逐题核对盲评图片像素与候选侧别，检查旧项目文件哈希 |

分差统一以 A 减 B。I 度量两种文本下有符号亮度响应之差；G 在每对内取两种响应的绝对值再相减。G 的正、负、零对应增强、减弱和不变。summary.csv 中 c0/c1 分别代表各轮的基础/夜景文本，bin0 到 bin5 顺序为 r＜1、r＝1、1＜r＜2、r＝2、r＞2、原图同分。initial_tie 计原图同分，new_tie 计非同分变成同分，end_tie 计比较终点同分。所有严格翻转均按实际符号判断，同分阈值为零。四分位数和百分位区间均线性插值。

人工冲突标记在评分前完成，覆盖时段、受光景观是原场景显著内容，或夜景与原描写明确相冲突的情形。轻微场景修饰和物体本身的颜色、灯具、作品名另行判断，逐条依据见冻结清单。旧 lighting.csv 的口径直接保留。两轮的文本编码有不同目的。旧标签只用于 A 中 P0 原图人工赢家核查。

## 运行命令

在服务器同级目录复现。若在新机器运行，先按照上一轮 README 恢复相同 venv、公开权重和验证集，再并排放置两个目录。评分脚本需要 Linux、NVIDIA GPU 和 nvidia-smi；执行时自动选择无计算进程且利用率为零的一张卡。

```bash
cd /home/data/users/shenc/zhangyc/tta_round2
PY=../tta_pilot/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MPLCONFIGDIR="$PWD/runtime/mpl"
mkdir -p logs
"$PY" -B freeze.py
"$PY" -Bu score_new.py --limit 2
"$PY" -B -c 'from common import *; assert len(jsonfile(ROOT/"smoke_checks.json")) == 4; print("smoke checks OK")'
"$PY" -Bu score_new.py --limit 64
"$PY" -B analyze_round2.py
"$PY" -B figures.py
"$PY" -B blind.py
"$PY" -B report.py
"$PY" -B verify.py
```

从现有结果重建分析时，从 analyze_round2.py 开始即可。只核对旧轮时，冻结清单后执行 `"$PY" -B -c 'from analyze_round2 import audit_old; audit_old()'`。评分中断时重跑相同的 --limit 命令即可，已完成的整对会校验并复用。初次执行的评分时间保存在 runs.json，后续调用会追加本次计时。

完全从零重跑评分时，在上一轮目录旁另建空目录，仅复制本轮 *.py、rewrites.tsv 和 blind_template.html。要启用旧文件保护验收，先在新目录保存基线，再依次运行上面的命令。

```bash
mkdir -p logs
sha256sum ../tta_pilot/*.csv ../tta_pilot/*.md ../tta_pilot/*.py ../tta_pilot/*.json > logs/pilot-before.sha256
```

## 人工盲评与导入

[评审者离线包](human_eval/participant.zip) 仅包含盲评页面、图片和空白表。解压后打开 index.html 即可。

向评审者分发 human_eval/index.html、同级 assets/ 与 annotations_blank.csv。mapping_private.csv 单独保管。直接用浏览器打开 index.html，判断提示词与左右图的匹配程度，选择“左更好”“右更好”“难以区分”。点击“导出标注 CSV”保存。页面可在支持 localStorage 的浏览器中保留本机进度；多位评审者使用独立浏览器配置或设备，各自导出一份 CSV。手填时保留全部 question_id，choice 可写 left、right、tie，中文选项也可导入；未回答留空。

```bash
# 检查空白模板，无真实标注时应返回 0/64
"$PY" -B import_annotations.py human_eval/annotations_blank.csv
# 收到真实标注后，为每位评审者指定单独的新输出文件
"$PY" -B import_annotations.py /path/to/annotations.csv --out human_eval/annotator01_comparisons.csv
```

导入器把左右选择映射回 A/B，只比较同一图对；每位评审者分别得到固定文本时 L0→L1 和固定亮度时 Q0→Q1 的结果。结果区分严格改选、选择相同、同为难分、进入或离开难分，以及缺答。当前真实标注为 0。

