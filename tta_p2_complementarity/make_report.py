"""Build tables from the enumerated results. Observed conclusions live in findings.md."""
from collections import defaultdict
from study import *

def fmt(x):
    return ("%.4f"%float(x)).rstrip("0").rstrip(".") or "0"

def pct(x):
    return f"{100*float(x):.2f}%"

def table(headers,rows):
    def line(items):return "| "+" | ".join(str(x).replace("|","\\|") for x in items)+" |"
    return "\n".join([line(headers),line(["---"]*len(headers)),*(line(r) for r in rows)])

def interval(r):
    return f"{float(r['diff_pp']):+.2f} [{float(r['lo_pp']):+.2f}, {float(r['hi_pp']):+.2f}]"

def idlist(ids):return "、".join(sorted(ids)) if ids else "无"

def report():
    stats={r["strategy"]:r for r in read_csv(ROOT/"strategy_summary.csv")}
    comparisons={(r["strategy"],r["comparison"]):r for r in read_csv(ROOT/"mix_comparisons.csv")}
    per=read_csv(ROOT/"pair_strategy_results.csv");combos=read_csv(ROOT/"combination_results.csv")
    shifts={r["variant"]:r for r in read_csv(ROOT/"score_shifts.csv")}
    env=file_json(ROOT/"environment.json");runs=file_json(ROOT/"runs.json")
    events={(r["strategy"],r["event"]):r for r in read_csv(ROOT/"mechanism.csv")}
    support=lambda st,base,ending:{r["pair_id"] for r in per if r["strategy"]==st and r["base_outcome"]==base and int(r[ending+"_n"])}
    fixes={s:support(s,"wrong","correct") for s in PURE+MIX}
    harms={s:support(s,"correct","wrong") for s in PURE+MIX}
    photo_f=set.union(*(fixes[s] for s in ("BB","CC","BC")))
    photo_h=set.union(*(harms[s] for s in ("BB","CC","BC")))
    bypair=defaultdict(list)
    for r in combos:
        if r["pick"]!=r["base_pick"]:bypair[r["pair_id"]].append(r)
    lines=["# P2 · 固定三次评分预算的增强互补性实验",
           "① **平移增加了一个光度策略未纠正的图对，JPEG 没有显示独立的新增纠正。** 003 的原图选择 A，人工赢家为 B。BB、CC、BC 和 JJ、BJ、CJ 在所有版本下都未纠正；TT 有 4/6 种组合纠正，BT、CT、JT 各有 4/16 种。就这批参数看，单用平移已经能找到这对图。",
           "② **这些新增纠正没有抵过新增改错。** 限定在光度策略从未出现相应事件的图对，BT 新增期望纠正 0.25 对、新增期望改错 1.50 对。计入所有图对，BT 相对原图期望纠正 0.25 对、改错 2.00 对，净少选对 1.75 对；其中 008 的 16/16 种 BT 都把原赢家改错。BJ 没有纠正，期望改错 0.75 对，另有 0.0625 对由选对变同分。含 J 或 T 的每种策略，严格选对数都低于原图参考。",
           "③ **若只安排一个独立复验，可优先复验 BC，并保留原图、BB、CC 作为对照。** BC 在 053 上提供了唯一一次跨类混合独有的纠正，期望选对数为 43.0625/64，较 CC 仅高 0.10 个百分点，配对 95% 区间为 [0.00, 0.29]。增量全部来自同一图对的 1/16 种组合；它是待检验的线索，当前没有可推广的优胜策略。",
           "## 固定预算下的人工偏好",
           "冻结 ImageRewardDB 验证集中的同一批 64 个不同原提示词和 128 张候选图。标签取原图对中 rank 较小的一侧。原图 P0 基线为 43 对严格选对、21 对严格选错、0 对同分。候选 A/B 使用同一个版本组合，每个组合先计算三个分数的等权均值并独立排序，再在组合上等权平均结果。同分指两侧三项和精确相等。",
           table(["策略","组合数","严格选对","严格选错","同分","期望选对数 / 64","相对原图净选对数"],
                 [[s,r["n_combinations"],pct(r["correct_rate"]),pct(r["wrong_rate"]),pct(r["tie_rate"]),fmt(r["correct_expected"]),fmt(r["net_correct_pairs"])] for s,r in stats.items()]),
           "O 为一次原图评分参考。BB、CC、JJ、TT 为单类；BC、BJ、BT 是预设的三个主要比较，CJ、CT、JT 为探索组。每项三次预算策略只用原图及另外两张独立增强视图。各行概率在舍入前合计 100%。",
           "![策略相对原图的配对准确率差](figures/strategies.png)",
           "### 原图判定的去向",
           table(["策略","错→对","对→错","对→同分","错→同分","同分→对 / 错 / 同分"],
                 [[s,*[fmt(r[k]) for k in ("wrong_to_correct","correct_to_wrong","correct_to_tie","wrong_to_tie")],
                   " / ".join(fmt(r["tie_to_"+y]) for y in ("correct","wrong","tie"))] for s,r in stats.items()]),
           "表中小数是均匀抽取版本组合时的期望图对数。例如，单个图对在 16 种组合中有 1 种被纠正，对该策略的错→对贡献为 1/16。原图没有同分，仍保留其全部转移列；对→同分也会降低严格选对率。",
           "### 每项混合策略与两个组成单类的比较",
           table(["混合","性质","减第一单类 · 百分点 [95% 区间]","减第二单类 · 百分点 [95% 区间]","样本内较强单类","减较强者 · 百分点 [95% 区间]"],
                 [[s,"预设主要" if s in PRIMARY else "探索",s[0]*2+"  "+interval(comparisons[s,s[0]*2]),s[1]*2+"  "+interval(comparisons[s,s[1]*2]),
                   comparisons[s,"best_of_components"]["observed_stronger"],interval(comparisons[s,"best_of_components"])] for s in MIX]),
           "以完整图对为单位配对 bootstrap 10,000 次，种子 42，每次保留该图对在全部版本和策略下的结果。差值单位为严格选对率的百分点。对较强单类的一列在每次重采样中重新取两个组成单类准确率的最大值，区间可与固定比较对象的区间不同。区间为逐项百分位区间，未作多重比较校正；独立样本单位始终只有 64 个提示词。",
           "## 纠正发生在谁身上，代价发生在谁身上",
           "只要有一个组合出现某项事件，该图对便进入相应集合。下表同时列出期望频率，避免把只有一个版本有效误读为稳定收益。光度对照集合取 BB、CC、BC 的并集。",
           table(["策略","出现过纠正的图对","出现过改错的图对","光度未纠正图对上的期望纠正数","光度未改错图对上的期望改错数"],
                 [[s,idlist(fixes[s]),idlist(harms[s]),
                   fmt(sum(float(r["correct_rate"]) for r in per if r["strategy"]==s and r["base_outcome"]=="wrong" and r["pair_id"] not in photo_f)),
                   fmt(sum(float(r["wrong_rate"]) for r in per if r["strategy"]==s and r["base_outcome"]=="correct" and r["pair_id"] not in photo_h))]
                  for s in PURE+MIX]),
           "比较混合策略与其各自组成单类时，“额外纠正”要求两个单类的所有版本组合都没有纠正过该图对。Jaccard 只用两个单类的纠正集合计算，其中任一集合为空即记不可判定。",
           table(["混合","组成单类均无的纠正","组成单类均无的改错","两个单类纠正集合的交集","纠正集 Jaccard"],
                 [[s,events[s,"fix"]["unique_vs_components"],events[s,"harm"]["unique_vs_components"],
                   events[s,"fix"]["component_intersection"],"不可判定" if events[s,"fix"]["jaccard_components"]=="NA" else fmt(events[s,"fix"]["jaccard_components"])] for s in MIX]),
           "### 所有发生改选或进入同分的图对",
           "003 给出了互补性的边界：例如 B090 和向下平移分别评分后与原图等权平均，A−B 分差从 +0.00537109 变为 −0.00195313，选中人工赢家 B；检查单类发现，TT 同样能纠正它。JJ 和 TT 则都在 005、008 上出现改错。平移的绝对分差位移中位数为 0.005005，高于亮度的 0.001282；更多跨界同时带来纠正和改错，必须结合净选对数判断收益。",
           f"共有 {len(bypair)}/64 对至少一次改变原图判定。以下分差为 A 减 B，给出仅在改变判定的组合中实际出现的增强后分差范围，包括精确的零。每种版本组合的分差与选择可在 [逐组合结果](combination_results.csv) 查到。",
           table(["图对","原图领先方","原始领先幅度","人工赢家","改变判定的策略","增强后分差范围","变化次数 / 120"],
                 [[i,rs[0]["base_pick"],f"{abs(float(rs[0]['d0'])):.8f}",rs[0]["human_winner"],"、".join(s for s in PURE+MIX if any(r["strategy"]==s for r in rs)),
                   f"{min(float(r['d_aug']) for r in rs):.8f} 至 {max(float(r['d_aug']) for r in rs):.8f}",len(rs)] for i,rs in sorted(bypair.items())]),
           "## 实例核查"]
    for e,label in [("fix","纠正"),("harm","改错")]:
        only=set(); own=set(); strict=set()
        for s in MIX:
            for k,target in [("unique_vs_all_pure",only),("unique_vs_components",own),("unique_vs_all_other_strategies",strict)]:
                term=events[s,e][k];target.update(term.split() if term!="—" else [])
        lines.append(f"跨类混合独有{label}：相对全部四种单类为 **{len(only)} 个不同图对**（{idlist(only)}）；若只要求超出自身两个组成单类则有 {len(own)} 对（{idlist(own)}）。进一步要求其余九种策略均无此事件，有 {len(strict)} 对（{idlist(strict)}）。")
    for c in read_csv(ROOT/"cases.csv") if (ROOT/"cases.csv").exists() else []:
        s=c["strategy"]
        lines.extend([f"### {c['event']}实例 · {c['pair_id']} · {s}",c["P0"].strip(),f"![图对 {c['pair_id']}，上 A 下 B，列为原图与两个独立版本]({c['image']})",
                      table(["候选","原图得分",c["view1"]+" 得分",c["view2"]+" 得分"],
                            [[side,*[c[side+str(j)] for j in range(3)]] for side in ("A","B")]),
                      f"人工赢家 {c['human']}，原图选择 {c['base_pick']}（分差 {float(c['d0']):.8f}），此组合选择 {c['pick']}（均值分差 {float(c['d_aug']):.8f}）。事件归类为“{c['case_type']}”。同类事件的命中数 / 组合数：{s} 为 {c[s+'_count']}，组成单类 {s[0]*2} 为 {c[s[0]*2+'_count']}，{s[1]*2} 为 {c[s[1]*2+'_count']}。图示仅用于回看，完整输入以评分缓存的像素哈希为准。"])
    lines.extend(["## 扰动强度与最终输入",
                  "B 和 C 使用 Pillow 的亮度、对比度操作，系数 0.90、0.95、1.05、1.10。J 从同一张原图分别做质量 85、89、93、97 的 JPEG 内存编解码，固定 4:2:0，关闭 progressive 和 optimize。T 水平或竖直移动边长的 3%，按最近整数像素取整，半数向上，使用不重复边缘的反射填充。每个版本独立从同一 RGB 原图生成；+ 表示两次独立评分。",
                  table(["族","单图平均有符号位移","单图绝对位移中位数 [四分位区间]","绝对分差位移中位数","模型输入平均像素绝对差的中位数（0–255 单位）"],
                        [[f,f"{float(r['mean_signed']):.8f}",f"{float(r['median_abs']):.8f} [{float(r['q25_abs']):.8f}, {float(r['q75_abs']):.8f}]",
                          f"{float(r['median_abs_gap_shift']):.8f}",f"{float(r['median_input_RGB255']):.3f}"] for f in FAMILIES for r in [shifts[f]]]),
                  "每族 512 个单图位移、256 个图对分差位移。官方验证变换实际使用 ResizeMaxSize 等比缩放并补边，随后归一化。2,048 个增强输入在模型空间都保留了像素变化。示例固定为清单第一对，以下为最终 224×224 输入反归一化后的显示；末行把 JPEG 与原图的绝对差放大八倍，便于看清压缩较轻时的残差。不同族的输入和评分位移不同，不能仅凭改变的图对多就推断错误互补。",
                  "![001 A 模型输入](figures/input_001_A.png)","[查看同一图对 B 侧全部版本](figures/input_001_B.png) · [每个强度的具体位移](score_shifts.csv)",
                  "## 运行与适用范围",
                  f"Keqing 的 {runs[0]['gpu']['name']}，物理 GPU {runs[0]['gpu']['index']}。官方 HPS v2.1 固定权重，验证预处理、eval、CUDA AMP，返回 float16；CSV 完整保存返回值。分差、频率、重采样统计使用双精度。原图 + B110 在前两对中重新推理的 8 条均与旧值精确一致。",
                  table(["阶段目标","新增图像推理","读取旧 P0 缓存","已完成的本轮评分命中","累计唯一记录","墙钟秒","GPU forward 秒"],
                        [[t["limit"],t["new_inferences"],t["old_hits"],t["p2_hits"],t["records"],f"{t['wall_seconds']:.2f}",f"{t['gpu_forward_seconds']:.2f}"] for t in runs]),
                  f"评分阶段共 {sum(t['wall_seconds'] for t in runs):.2f} 秒，新增 {sum(t['new_inferences'] for t in runs):,} 次真实推理。128 张图各有 17 条唯一评分，共 2,176 条；4×6 + 6×16 = 120 种三次预算参数组合，共 7,680 个图对决策。后续离线分析只读这些分数。原图同分单列，以完整图对为重采样单位。",
                  "这 64 对已用于此前的探索。参数池没有按感知强度或单图评分位移匹配。任何样本内领先均须用新的、未用于探索的提示词和图对独立复验。"])
    (ROOT/"report.md").write_text("\n\n".join(lines)+"\n",encoding="utf8")

if __name__=="__main__":report()

