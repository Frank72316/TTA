# 第三次评分的价值

[中文结果](report.md) · [预先冻结的判据](protocol.md) · [测试清单](pairs_frozen.csv) · [样本排除与哈希](freeze.json)

## 离线复算

本次运行基于仓库提交 `22f9cb2`。Asus 文件时间（UTC+8）记录了冻结与评分顺序：11 月 28 日 19:52:49 写入协议，19:53:07 写入清单与哈希，19:54:28 完成两对冒烟，19:57:56 写出正式评分，19:58:40 写出延迟记录。

从仓库根目录运行，Python 3.10+、NumPy 即可。测试和开发结果分别输出；核验脚本还逐条比对延迟基准输出的 A/B 选择与主评分。数据文件使用 UTF-8，清单和协议按 LF 文本字节计算 SHA-256。

```bash
python P2/third_view_value/analyze.py --dev
python P2/third_view_value/analyze.py
python P2/third_view_value/verify.py
```

| 文件 | 阅读方式 |
|---|---|
| scores.csv、smoke_scores.csv | 每行一个原生 FP16 分数，完整表按清单、A/B、O/B/C 排序。按 pair_id 连接清单可取原提示词、排名和人工赢家 |
| strategy_summary.csv、intervals.csv | 严格对/错/平期望对数、原图到策略的九格转移；主问题和五项替代界限（百分点） |
| prompt_results.csv、original_errors.csv | 各提示词的策略成功概率；原始错例中的可达覆盖、从明确选错救回的组合数及分母 |
| recipe_margins.csv、fixed_c1_actions.csv | 每套配方的原始/第三视图证据、是否跨零；同一 C1 下追加 B 或 C 的联合选择 |
| prompt_transitions.csv、transition_summary.csv | 固定第一次对比度的九格转移；总表还按原图对/错/平分层。小数对数是均匀取搭配的期望值 |
| replacement_cases.json、cases.json、cases.png | 严格纠正丢失、改错消失等涉及的 ID、各自触发频率；按冻结规则选出的首个案例 |
| timings.csv、latency_summary.json、timing_run.json、runs.json | 逐对实测、分位数、计时额外前向、评分缓存命中和墙钟秒数 |
| dev_*.csv、dev_precision.json | validation 的 256 对历史复算与精度规划，独立于测试主要推断 |

## 从官方数据重建（Asus）

实跑工作目录为 `/home/shenc/tta_prompt_flip/P2/third_view_value`，同级 geometry_specificity 提供 HPS 初始化；固定模型权重路径和摘要见 environment.json。使用现成 Linux Python 3.10 环境、PyArrow、Pillow、PyTorch、hpsv2、NumPy 和 Matplotlib。模型加载器离线校验 OpenCLIP 和 HPS 两份固定权重。原图和逐对断点 JSON 留在 Asus。下面从含历史三份清单的完整仓库执行；若运行目录只同步了本实验，可用 freeze.py 的 `--history` 依次指定 64、256、88 的 CSV。

```bash
cd P2/third_view_value
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/data/users/shenc/zhangyc/tta_pilot/.venv/lib/python3.10/site-packages
PY=/home/shenc/tta_geometry_specificity_asus/runtime/bin/python
PILOT=/home/shenc/tta_geometry_specificity_asus/inputs/tta_pilot
DATA=/home/shenc/tta_prompt_flip/data/third_view_value
REV=c493721e19e296eb615420036f2a2eed08412bb4
mkdir -p "$DATA"
for file in metadata-test.parquet images/test/test_1.zip images/test/test_2.zip; do
  dest="$DATA/${file##*/}"
  if [ ! -f "$dest" ]; then
    curl -fL --retry 3 "https://hf-mirror.com/datasets/zai-org/ImageRewardDB/resolve/$REV/$file?download=true" -o "$dest.part" && mv "$dest.part" "$dest" || exit 1
  fi
done
"$PY" freeze.py "$DATA/metadata-test.parquet"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4
"$PY" run_gpu.py score --limit 2 --pilot "$PILOT" --data "$DATA"
"$PY" run_gpu.py score --pilot "$PILOT" --data "$DATA"
"$PY" run_gpu.py time --pilot "$PILOT" --data "$DATA"
"$PY" analyze.py
"$PY" verify.py
"$PY" make_cases.py --data "$DATA"
```

正式评分为两候选各九个唯一视图。冒烟样本计入 400 对一次，完整运行读取其断点；实验签名绑定输入数据版本、模型、设备、数值精度、脚本和清单。重新测量延迟会更新 timings.csv，请在副本上执行。GPU 基准以压缩图片字节已在内存为起点，每个请求重新解码两图、生成视图、预处理、编码文字一次、逐图编码并同步取回选择。真实部署的延迟还会随存储、并发、批大小和文本缓存策略变化。

官方记录的 rank 越小越好，标注对应原始提示词和原图。每个参数组合独立做出 A/B 决策，然后对决策取平均。计分时用 float64 加和原生二进制分数；组合数不增加独立提示词数。冻结协议沿用项目既有的有向替代标准，五项界限及十个方向的同时校正已完整写入 protocol.md。
