# HPS v2.1 亮度与夜景提示词先导实验

结果见 [实验报告](report.md)。服务器工作目录为 `/home/data/users/shenc/zhangyc/tta_pilot/`。本地副本包括代码、报告、三个案例图和完整结果。验证集压缩包和项目虚拟环境保存在服务器；代码直接调用公开模型。

## 输入与输出

- [pairs.csv](pairs.csv) 保存 64 对的 ID、P0/P1 完整原文、A/B 数据集相对路径、两张图各自的 rank 和 P0 人工赢家。首尾空格按发布原文保留，P1 用一个额外空格连接固定后缀。读取 pair_id 时按字符串处理，以保留前导零。
- [scores.csv](scores.csv) 每行一个原始图文分数。唯一键为 pair_id、side、prompt_condition、light。score 为官方原生分数的可无损读回十进制表示。tokens_before_truncation 含首尾标记，same_P0_P1_tokens 根据编码器的实际输入做比较。
- [pair_results.csv](pair_results.csv) 每对一行，P0/P1 各有 d0、d1、delta、change_A/B、tta_A/B、d_tta、三种 pick、两种 change，并含 I 和 abs_I。pick 为 A、B 或 tie；change 为 flip（严格异号）、tie（任一端同分）或 same。人工对错只在 P0 计算。
- [lighting.csv](lighting.csv) 是打分前根据已冻结的 64 条原文做的光照编码。文件列出所有命中项和英文证据。完整原文在 pairs.csv。更换样本须重新审阅。
- [manifest.json](manifest.json)、[environment.json](environment.json)、[runs.json](runs.json) 分别记录抽样、执行环境与分阶段耗时。[cases](cases/) 中的 PNG 为报告预览，打分直接读取公开验证集 WebP 并在内存中转换。

## 数据与抽样

读取 [ImageRewardDB](https://huggingface.co/datasets/zai-org/ImageRewardDB) 的 `metadata-validation.parquet`，并从同一提交的两个 validation ZIP 按文件名读取原图。检查了实际 9 个字段和 2,584 行。官方构造器对 `rank[idx] < rank[idy]` 的记录指定 `img_better=idx`，排名数值小者优。见[固定版本的构造器](https://huggingface.co/datasets/zai-org/ImageRewardDB/blob/c493721e19e296eb615420036f2a2eed08412bb4/ImageRewardDB.py)。

抽样使用 Python 3.10 的 `random.Random(42)`。先对不同提示词原文排序，再从 409 条合格原文等概率无放回选 64 条。同一原文有多个 prompt_id 时，在有合格图对的 ID 中随机选一个。按路径稳定排序，等概率抽一个名次不同的无序图对，独立随机分配 A/B。先写 pairs.csv 及 SHA-256，再访问 HPS。P0 和 P1 的实验条件均保留抽中的所有样本。

## 评分协议

每张原图解码并 `convert("RGB")`；L1 使用 `ImageEnhance.Brightness(rgb).enhance(1.1)`，两种 PIL 图像都直接传入 [HPS 官方实现](https://github.com/tgxs002/HPSv2/blob/866735ecaae999fa714bd9edfa05aa2672669ee3/hpsv2/img_score.py)。每次显式调用 `hpsv2.score(images, prompt, hps_version="v2.1")`。官方接口使用 eval、验证预处理和 CUDA 自动混合精度；实验关闭 cuDNN benchmarking，启用确定性算法。每轮仅对一张启动时空闲的 GPU 设置 CUDA_VISIBLE_DEVICES。实际两阶段都使用 Keqing 的 GPU 0，RTX 4080 SUPER。

独立项目 venv 只读复用服务器现有 Python 3.10.20、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124。其余实际版本在 environment.json。上游运行时所用 HPS 代码固定在 866735ec；HPS 与初始化所用 CLIP 权重均验证 revision 和 SHA-256。评分时使用离线缓存。

## 本次实际执行

以下为本次在 Keqing 运行的命令。验证集首次传输经过 HF 镜像；下载文件已针对固定版本校验。模型使用服务器现有公开 HF 缓存。

```bash
cd /home/data/users/shenc/zhangyc/tta_pilot
mkdir -p vendor data logs
~/miniconda3/envs/llada_new/bin/python -m venv --system-site-packages .venv
REV=866735ecaae999fa714bd9edfa05aa2672669ee3
curl -fsSL --retry 2 --max-time 90 "https://codeload.github.com/tgxs002/HPSv2/tar.gz/$REV" -o vendor/hps-source.tar.gz
tar -xzf vendor/hps-source.tar.gz -C vendor
.venv/bin/python -m pip install --no-deps --no-build-isolation "./vendor/HPSv2-$REV"
.venv/bin/python -m pip install -i https://pypi.org/simple ftfy einops braceexpand sentencepiece 'protobuf<4' 'timm==0.9.16' 'webdataset==0.2.111' clint

hf=https://hf-mirror.com/datasets/zai-org/ImageRewardDB/resolve/c493721e19e296eb615420036f2a2eed08412bb4
for f in metadata-validation.parquet ImageRewardDB.py images/validation/validation_{1,2}.zip; do
  (dest=data/${f##*/}; curl -fsSL --retry 2 --max-time 600 "$hf/$f?download=true&run=$(date +%s)" -o "$dest.part" && mv "$dest.part" "$dest") &
done
wait
.venv/bin/python prepare.py
OMP_NUM_THREADS=4 .venv/bin/python -u score.py --limit 2 > logs/smoke.log 2>&1
OMP_NUM_THREADS=4 .venv/bin/python -u score.py --limit 64 > logs/full.log 2>&1
.venv/bin/python analyze.py
```

冒烟阶段产生的 16 条记录经完整性检查后直接纳入正式结果。脚本按整对原子写入 scores.csv，并用文件锁阻止并行写入。中断后重跑相同的 score 命令，将从已完成的图对继续。analyze.py 可单独重算全部统计；交付副本已附案例 PNG，已有图片时可在无数据压缩包的机器上复核结果。

## 在新的 Linux CUDA 环境从头复现

新建空目录，复制本目录的 Python 脚本、requirements.txt 和 lighting.csv。使用独立目录可保留本次实测的原始记录。以下以 Python 3.10 和 CUDA 12.4 的 PyTorch 构建为例。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation 'https://codeload.github.com/tgxs002/HPSv2/tar.gz/866735ecaae999fa714bd9edfa05aa2672669ee3'
# 需要镜像时先执行 export HF_ENDPOINT=https://hf-mirror.com
python fetch.py
export HF_HUB_CACHE="$PWD/data/model-cache"
python prepare.py
OMP_NUM_THREADS=4 python score.py --limit 2
OMP_NUM_THREADS=4 python score.py --limit 64
python analyze.py
```

fetch.py 下载固定版本、验证文件哈希，并在项目内建立官方接口可读取的权重引用。代码不读旧 3D 项目的源码、环境或任何分数。独立复算统计的入口为 `python analyze.py`（需 Pillow）；完整的 GPU 推理入口是 `score.py`。

