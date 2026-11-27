"""Build the Chinese report entirely from audited CSVs and the recorded device timings."""
from collections import Counter
from common import *

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|",
                      *["| "+" | ".join(str(v).replace("|","\\|") for v in row)+" |" for row in rows]])

def fmt(x):
    return f"{float(x):.8f}".rstrip("0").rstrip(".") or "0"

def interval(s,lo,hi):
    return f"[{fmt(s[lo])}, {fmt(s[hi])}]"

def main():
    sums={r["group"]:r for r in read_csv(ROOT/"summary.csv")}
    old={r["pair_id"]:r for r in read_csv(ROOT/"old_audit.csv")}
    new={r["pair_id"]:r for r in read_csv(ROOT/"pair_results.csv")}
    prompts={r["pair_id"]:r for r in frozen()}
    tags=read_csv(ROOT/"old_prompt_audit.csv")
    ident=[r["pair_id"] for r in tags if int(r["identical"])]
    trunc=[r for r in tags if not int(r["P1_complete"])]
    lit=sum(int(r["light"]) for r in tags)
    overlap=sum(int(r["light"]) for r in trunc)
    ns,clean,os=sums["new_all"],sums["no_conflict"],sums["old_all"]
    lines=["# 亮度增强与夜景提示词的交互：短提示词配对实验",
           f"旧 P0 的两次调亮翻转都没有跨过均值的严格翻转边界：005 恰好打平，053 仍选择原来的 A。  ",
           f"旧提示词中 {lit} 条原有光照描述、{len(trunc)} 条追加后截断，两者重叠 {overlap} 条；其中 {len(ident)} 条追加前后的模型输入完全相同。  ",
           f"新短提示词全部完整进入模型，夜景条件下的亮度响应没有一致变强：配对幅度变化 G 的中位数为 {fmt(ns['G_median'])}，95% bootstrap 区间为 {interval(ns,'bootstrap_low','bootstrap_high')}。",
           "## A · 旧分差如何跨过两条边界",
           "对同一图对，原分差 d₀ = A₀ − B₀，调亮分差 d₁ = A₁ − B₁，Δ = d₁ − d₀ = δA − δB。共同漂移 (δA + δB)/2 描述两张图整体涨跌，差异漂移 Δ 决定它们之间的分差。等权 TTA 的分差是 d₀ + Δ/2，因而只有调亮后反向的分差绝对值超过原分差，均值才会严格翻转。",
           "按 d₀、d₁ 和均值分差的实际符号分区；r = −Δ/d₀ 仅在 d₀ 非零时计算。每行各分区之和为总数。",
           table(["提示词","r＜1 · 均保持","r＝1 · 调亮平","1＜r＜2 · 仅调亮翻","r＝2 · 均值平","r＞2 · 均翻","原图同分"],
                 [[c,*[s[f"c{j}_bin{k}"] for k in range(6)]] for s,cs in [(os,("P0","P1")),(ns,("Q0","Q1"))] for j,c in enumerate(cs)]),
           "### 两个旧翻转的原始分数",
           table(["图对","P0 人工赢家","A 原图","B 原图","A 调亮","B 调亮","原图 → 调亮 → 均值"],
                 [[i,old[i]["human_P0"],*[old[i]["P0_"+k] for k in ("A0","B0","A1","B1")],
                   " → ".join(old[i]["P0_"+k].replace("tie","同分") for k in ("pick0","pick1","pick_tta"))] for i in ("005","053")]),
           table(["图对","原图分差","调亮分差","差异漂移 Δ","共同漂移","均值分差","r"],
                 [[i,*[old[i]["P0_"+k] for k in ("d0","d1","delta","common_drift","d_tta","r")]] for i in ("005","053")])]
    counts=Counter((r["P0_pick0_human"],r["P0_pick_tta_human"]) for r in old.values())
    baseline=sum(v for (a,b),v in counts.items() if a=="correct")
    tta=sum(v for (a,b),v in counts.items() if b=="correct")
    assert (baseline,tta,int(os["c0_L1_flip"]),int(os["c0_TTA_flip"]),int(os["c0_TTA_end_tie"]))==(43,42,2,0,1)
    lines.extend([f"005 两张图都涨分，B 涨得更多。053 两张图都降分，A 降得更多。两次差异漂移都使调亮图选 B，但取均值后，005 由选对变成同分，053 维持原来的错误选择。",
                  f"按 P0 原图人工标签，原图选对 {baseline}/64，TTA 选对 {tta}/64。错→对 {counts['wrong','correct']}，对→错 {counts['correct','wrong']}，对→同分 {counts['correct','tie']}。旧 scores.csv、pair_results.csv 和报告数值一致。旧评分脚本只输分，旧分析脚本与报告均以严格异号判翻转，任一端为零单列。本表进一步拆开原图同分与新产生的同分。"])
    lines.append(table(["范围","方式","严格翻转","原图同分","非同分→同分","终点同分"],
                       [[c,mode,s[f"c{j}_{key}_flip"],s[f"c{j}_{key}_initial_tie"],s[f"c{j}_{key}_new_tie"],s[f"c{j}_{key}_end_tie"]]
                        for s,cs in [(os,("P0","P1")),(ns,("Q0","Q1"))] for j,c in enumerate(cs)
                        for key,mode in (("L1","只看调亮"),("TTA","等权均值"))]))
    lines.extend(["![分差与两条翻转边界](figures/flip_boundaries.png)",
                  "横轴为原图分差，纵轴为差异漂移。实线 Δ = −d₀ 时调亮图正好打平，虚线 Δ = −2d₀ 时均值正好打平。三面板使用相同坐标范围。",
                  "## B · 旧夜景追加语真正送入模型的范围",
                  "复用了旧 lighting.csv 的 23 条逐条文本判断。HPS ViT-H-14 实际 tokenizer 输出 77 个位置，长度含起止标记，按结束标记定位非 padding 区域。旧组仅按原文标记和实际 token 分组。I = Δ₁ − Δ₀ 衡量响应改变；G = abs(Δ₁) − abs(Δ₀) 衡量幅度增强或减弱。例如由 +0.002 变为 −0.002，交互绝对值为 0.004，而 G 为零。",
                  table(["旧样本组","图对","原提示词响应幅度中位数","夜景响应幅度中位数","交互绝对值中位数","G 中位数","G 四分位区间","G 正 / 负 / 零"],
                        [[label,s["n"],fmt(s["median_amp0"]),fmt(s["median_amp1"]),fmt(s["abs_I_median"]),fmt(s["G_median"]),interval(s,"G_q1","G_q3")," / ".join(s[k] for k in ("G_positive","G_negative","G_zero"))]
                         for key,label in [("old_all","全部"),("unlit_complete","无光照要求且追加完整"),("lit_complete","有光照要求且追加完整"),("truncated","追加被截断"),("identical","↳ 其中输入完全相同")] for s in [sums[key]]]),
                  f"前两个完整组和截断组合计 64 对。截断组中，原无光照要求 {len(trunc)-overlap} 对、已有要求 {overlap} 对。"+"、".join(ident)+" 的四个对应图像分数在 P0、P1 下逐项相同，每对 I = 0。其余截断样本仍有部分追加语进入模型。逐条 token ID、原文、证据见 [旧提示词核查](old_prompt_audit.csv)。",
                  "在追加完整且原无光照要求的 37 对中，19 对增强、16 对减弱、2 对不变。可见截断确实让一部分干预失效，但去掉截断和已有光照要求后，响应方向仍然混合。各子组由原有文本特征决定，组间差异不能单独归因于截断。两个幅度列分别取中位数，G 列先在同一图对内相减再取中位数。",
                  "## C · 用短提示词重新评分",
                  "保留原图、原 A/B、相同亮度系数和同一评分接口。Q0 以简短英文保留主体、动作与空间关系，删去光照、时段、整体色调和部分风格词；Q1 原样保留 Q0，再追加 “The scene is at night, with dim lighting.”。命名作品、物体固有颜色和作为物体的灯具可保留。最长 Q1 为 "+str(max(int(p["Q1_tokens"]) for p in prompts.values()))+" 个 token。每一对的 Q0、Q1 编码不同，后缀 token 全部保留。",
                  table(["新样本组","图对","基础响应幅度中位数","夜景响应幅度中位数","G 中位数","G 的 95% 区间","G 正 / 负 / 零","交互绝对值中位数","交互绝对值四分位区间"],
                        [[label,s["n"],fmt(s["median_amp0"]),fmt(s["median_amp1"]),fmt(s["G_median"]),interval(s,"bootstrap_low","bootstrap_high")," / ".join(s[k] for k in ("G_positive","G_negative","G_zero")),fmt(s["abs_I_median"]),interval(s,"abs_I_q1","abs_I_q3")] for label,s in [("全部 · 主结果",ns),("无语义冲突",clean)]]),
                  "主结果在打分前指定。对完整图对有放回重采样 10,000 次，每次同时带入该对全部条件，随机种子 42；报告中位数的百分位区间。去掉预标记冲突项后区间跨零，全体的区间上界为零。两组均未显示一致增强。",
                  "语义冲突标记采用保守口径，包含时段或发光景观是原文重要内容、删改会削弱原意的样本。冻结时标出的图对为 "+ "、".join(i for i,p in prompts.items() if int(p["semantic_conflict"]))+"；逐条原因和删除内容见 [提示词清单](prompts.csv)。短提示词也删减了部分画风细节，旧实验与新实验的差异同时包含这些改写的影响。",
                  "![配对亮度响应散点](figures/response.png)",
                  "每一点代表一对图，横纵轴同尺度。对角线上方表示 G 为正，下方表示 G 为负；相同坐标会重叠。",
                  "## 六个预先约定规则选出的图对",
                  "按交互绝对值降序取前三（同值按图对 ID 升序），从其余图对中用种子 42 随机抽三对。每个缩略图的左列是 A、右列是 B，上排原图、下排调亮。以下为模型原始分数。"])
    for case in read_csv(ROOT/"cases.csv"):
        i=case["pair_id"];p=prompts[i];r=new[i]
        lines.extend([f"### {case['selection']} · 图对 {i}",
                      "原文　"+p["original_prompt"].strip(),"Q0　"+p["Q0"],"Q1　"+p["Q1"],
                      f"![图对 {i} 的四张候选图](figures/case_{i}.png)",
                      table(["提示词","A 原图","B 原图","A 调亮","B 调亮","亮度响应 Δ"],
                            [[c,*[r[c+"_"+k] for k in ("A0","B0","A1","B1","delta")]] for c in ("Q0","Q1")]),
                      f"交互绝对值 {fmt(r['abs_I'])}，幅度变化 G = {fmt(r['G'])}。"])
    runs=jsonfile(ROOT/"runs.json");env=jsonfile(ROOT/"environment.json")
    lines.extend(["## 运行与验收",
                  f"服务器 {env['host']}，{env['gpu']}，物理 GPU {runs[0]['gpu']['index']}。HPS v2.1，官方 CUDA AMP，原始输出 float16；CSV 无损保存原始输出，再以 float64 计算全部分差、比值和统计量。模型自身的量化会影响近零分差和同分。软件版本、权重与输入哈希见 [环境记录](environment.json) 和 [冻结记录](freeze.json)。",
                  table(["阶段目标","新评分","本轮缓存命中","旧分数命中","评分阶段墙钟秒","同步 API 秒","GPU 前向秒"],
                       [[t["limit"],t["new_scores"],t["round2_cache_hits"],t["old_score_hits"],f"{t['wall_seconds']:.2f}",f"{t['api_seconds']:.2f}",f"{t['forward_seconds']:.2f}"] for t in runs]),
                  f"两个评分阶段合计 {sum(t['wall_seconds'] for t in runs):.2f} 秒。GPU 前向累计 {sum(t['forward_seconds'] for t in runs):.2f} 秒，由 CUDA event 在每次模型 forward 前后测量；同步 API 时间含权重加载与官方打分路径。GPU 前向时间不包括点积、主机预处理或脚本初始化。",
                  "旧、新评分各有 64 个相同 pair_id、每对 8 条有限分数、各 512 条，候选图像路径和 A/B 映射一致。逐对验证 Δ = δA − δB，均值分差也与两张图分别取均值后再相减严格相等。旧结果仅用于复核，Q0、Q1 的统计只使用模型输出。",
                  "## 人工验证待完成",
                  "从 64 个图对等概率抽 16 对，种子 42；每对四个条件，共 64 题，题序和左右位置随机化。[盲评页面](human_eval/index.html) 可离线打开，已附空白 CSV。提交题目时只分发页面、assets 与空白表，将私有映射留给分析者。当前没有真实作答。收到标注后，用导入脚本按左右映射解码，分别统计固定提示词下调亮前后的改选、固定图像条件下更换提示词的改选，同分和无法比较分列。"])
    (ROOT/"report.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()

