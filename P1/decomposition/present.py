"""Produce the two offline figures and a report from the checked decomposition."""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
GROUPS = {"all":"全体","no_conflict":"无语义冲突"}
INK="#22354a"
TEAL="#187e7c"
ORANGE="#b35a23"
PURPLE="#79548c"


def style():
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Noto Sans CJK SC","Microsoft YaHei","Noto Sans CJK JP","SimHei","PingFang SC") if f in fonts),"DejaVu Sans")
    plt.rcParams.update({"font.family":[font,"DejaVu Sans"],"font.size":10,"text.color":INK,
                         "axes.labelcolor":INK,"axes.titlecolor":INK,"axes.spines.right":False,
                         "axes.spines.top":False,"axes.edgecolor":"#7b8890","axes.unicode_minus":False,
                         "savefig.facecolor":"white"})


def plot_interactions(rows,s):
    coords=Counter((r["h_A"]*1000,r["h_B"]*1000) for r in rows)
    lim=max(abs(z) for point in coords for z in point)*1.12
    fig,ax=plt.subplots(figsize=(7.3,6.2),layout="constrained")
    x=np.array([-lim,lim])
    ax.axhline(0,color="#a6b0b6",lw=.8)
    ax.axvline(0,color="#a6b0b6",lw=.8)
    ax.plot(x,x,c=TEAL,ls="--",lw=1,label="共同方向  y = x")
    ax.plot(x,-x,c=PURPLE,ls=":",lw=1.4,label="相反方向  y = −x")
    for (a,b),n in coords.items():
        ax.scatter(a,b,s=32*n,c=INK,edgecolors="white",linewidth=.55,alpha=.85,zorder=3)
    ax.set(aspect="equal",xlim=(-lim,lim),ylim=(-lim,lim),
           xlabel="候选 A 的单图交互 h（×10⁻³ HPS 分）",ylabel="候选 B 的单图交互 h（×10⁻³ HPS 分）")
    ax.legend(loc="upper left",frameon=False)
    fig.suptitle("夜景提示词怎样改变同一图对中两张图的调亮响应",fontsize=15,fontweight="bold")
    ax.set_title(f"{s['pairs']} 对 · 原点 {s['origin']} 对 · {s['pairs_at_repeated_locations']} 对位于 {s['repeated_locations']} 个重合位置 · 点面积与图对数成正比",fontsize=9,pad=10)
    fig.savefig(ROOT/"figure1_interactions.png",dpi=185)
    plt.close(fig)


def plot_leads(rows):
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.9),layout="constrained",height_ratios=[1.2,1])
    all_x=[r[q+"_lead"]*1000 for r in rows for q in ("Q0","Q1")]
    all_y=[r[q+"_toward"]*1000 for r in rows for q in ("Q0","Q1") if r[q+"_toward"] is not None]
    far=max(all_x)*1.045
    low=min(all_y)*1.1
    high=max(all_y)*1.18
    for j,(q,title) in enumerate((("Q0","新基础提示词"),("Q1","追加夜景要求"))):
        ties=[r for r in rows if r[q+"_d0"]==0]
        present=[r for r in rows if r[q+"_d0"]!=0]
        note="、".join(r["pair_id"] for r in ties)
        for k in range(2):
            ax=axes[k,j]
            right,lo,hi=(far,low,high) if k==0 else (8,-3,6)
            ax.axhline(0,color="#98a2a8",lw=.8)
            line=np.array([0,right])
            ax.plot(line,line,c=ORANGE,lw=1.4)
            ax.plot(line,2*line,c=PURPLE,ls="--",lw=1.4)
            ax.scatter([r[q+"_lead"]*1000 for r in present],[r[q+"_toward"]*1000 for r in present],
                       s=26,c=TEAL,edgecolors="white",linewidth=.5,alpha=.8,zorder=3)
            for r in present:
                cat=r[q+"_class_L1"]
                if cat not in ("exact_tie","flip"):
                    continue
                pt=r[q+"_lead"]*1000,r[q+"_toward"]*1000
                ax.scatter(*pt,s=44,facecolors="white",edgecolors=ORANGE,linewidths=1.5,zorder=4)
                if k==1:
                    ax.annotate(r["pair_id"]+(" 调亮翻转" if cat=="flip" else " 调亮平局"),pt,
                                xytext=(14,9 if cat=="flip" else -18),textcoords="offset points",fontsize=8,
                                arrowprops=dict(arrowstyle="-",lw=.7,color=ORANGE))
            ax.set_xlim(-right*.015,right)
            ax.set_ylim(lo,hi)
            ax.grid(alpha=.13)
            if j==0:
                ax.set_ylabel("朝平局方向的变化（×10⁻³ 分）")
            if k==0:
                ax.set_title(f"{q} · {title}　{len(present)} 对有领先方",fontsize=12)
                ax.add_patch(Rectangle((0,-3),8,9,fill=False,lw=.85,ls=":",edgecolor="#76878d"))
                ax.text(.99,.025,f"原图同分另列：{len(ties)} 对"+(f"（{note}）" if note else ""),transform=ax.transAxes,ha="right",fontsize=8.5)
            else:
                ax.set_title("小分差区域放大",fontsize=10)
                ax.set_xlabel("原图领先幅度（×10⁻³ HPS 分）")
    axes[0,1].legend(handles=[Line2D([],[],color=ORANGE,label="调亮平局  y = x"),
                               Line2D([],[],color=PURPLE,ls="--",label="均值平局  y = 2x")],
                     loc="upper right",frameon=False,fontsize=9)
    fig.suptitle("调亮朝平局移动了多远，原领先方还剩多少优势？",fontsize=15,fontweight="bold")
    fig.savefig(ROOT/"figure2_choice_boundaries.png",dpi=185)
    plt.close(fig)


def table(headers, rows):
    def line(vals):
        return "| "+" | ".join(str(x) for x in vals)+" |"
    return "\n".join([line(headers),line(["---"]*len(headers)),*(line(row) for row in rows)])


def f(v):
    return f"{v:.8f}".rstrip("0").rstrip(".") if v else "0"


def iq(s,key):
    return f"{f(s[key+'_median'])} [{f(s[key+'_q1'])}, {f(s[key+'_q3'])}]"


def pct(v):
    return f"{100*v:.2f}%" if v is not None else "NA"


def trio(s,keys):
    return " / ".join(str(s[k]) for k in keys)


def render(rows,sums,prec,prompts,cases):
    s=sums["all"];clean=sums["no_conflict"]
    style()
    plot_interactions(rows,s)
    plot_leads(rows)
    report=[
        "# P1 单图响应分解 · 调亮响应从单图到候选排序",
        f"① 在这 64 对图、HPS v2.1 和调亮 10% 的条件下，追加夜景要求使单图调亮响应的有符号中位变化为 **{f(s['h_median'])}**，绝对变化中位数为 **{f(s['abs_h_median'])} 分**，约合原始输出的 2–4 个可表示步长。",
        f"② 变化以两图共有的分量为主，共同分量占交互平方和的 **{pct(s['R'])}**，逐次删一对后为 {pct(s['loo_min'])}–{pct(s['loo_max'])}；少数大响应会影响占比，输出精度也限制了微小差异的分辨。",
        f"③ 多数亮度变化维持或扩大原领先优势，即使朝平局移动，Q0、Q1 的位移中位也只覆盖各自原领先幅度的 **{pct(s['Q0_toward_fraction_median'])}、{pct(s['Q1_toward_fraction_median'])}**，单一增强幅度只能辨认这一相对门槛，无法独立归因于响应偏小或原分差偏大。",
        "## 分数、条件与独立复算",
        "读取短提示词实验冻结的 64 个图对及图文评分，以图对 ID、A/B、Q0/Q1、L0/L1 作为唯一键。Q0 是重写后的基础提示词，Q1 在其后统一追加夜景与低照度要求。L0 是原图，L1 为在官方预处理前将 RGB 亮度乘以 1.1。八种组合齐全，512 条分数均有限且无重复。图像路径、A/B 顺序、文本 token 哈希、亮度系数与冻结清单相符。",
        f"用原始分数逐对重算 G = abs(Δ₁) − abs(Δ₀)，得到增强 / 减弱 / 不变 **{trio(s,['G_pos','G_neg','G_zero'])}**，中位数 {s['G_median']}。64 个 G、I 和两种提示词的原图、调亮与均值分差均与短提示词实验逐对表完全相同。以下计数中的同分和零均指精确相等。",
        "## 1　单图的调亮响应",
        "对候选 X，先在每个提示词下算 b = 调亮分数 − 原图分数，再算 h = b(Q1) − b(Q0)。h 为负表示调亮带来的涨幅缩小或降幅加深。A/B 是随机位置，单图分布将两侧合并；同向性与敏感性检查保留完整图对。方括号表示第 25 至 75 百分位。",
        table(["样本","提示词","图片数","调亮涨分","降分","同分"],
              [[GROUPS[k],q,t["images"],*[t[q+"_brightness_"+a] for a in ("pos","neg","zero")]]
               for k,t in sums.items() for q in ("Q0","Q1")]),
        table(["样本","h 中位数 [四分位区间]","h 绝对值中位数 [四分位区间]","单图 h 正 / 负 / 零","图对同正 / 同负 / 异号 / 至少一零"],
              [[GROUPS[k],iq(t,"h"),iq(t,"abs_h"),trio(t,["h_pos","h_neg","h_zero"]),trio(t,["h_pair_both_pos","h_pair_both_neg","h_pair_opposite","h_pair_zero"])] for k,t in sums.items()]),
        f"全部 {s['images']} 张图中，{s['h_neg']} 张的调亮响应在 Q1 下变得更负，{s['h_pos']} 张变得更正，{s['h_zero']} 张完全相同。绝对变化的中位数很小，{s['h_within_local_max_ulp']} 张的绝对变化至多相当于该图四个输入分数中最大的一个 float16 局部步长。这个量级比较没有给单张图划分显著性。",
        "![单图交互配对分布](figure1_interactions.png)",
        "图 1　两轴同尺度，每个坐标是一种 (h_A, h_B) 组合，重复位置合并显示。靠近原点表示两张图都变化很小；远离原点而靠近 y=x 表示两图同向、幅度相近；靠近 y=−x 表示两图近似等量反向。",
        "## 2　两图共有的变化与分差中的变化",
        "C = (h_A + h_B)/2 是两图都分到的共同分量，D = (h_A − h_B)/2 是方向相反的半分差。重建为 h_A = C+D、h_B = C−D。改变提示词后，传到候选分差的交互为 I = h_A − h_B = 2D，共同分量在做差时抵消。逐对同时验证了 C²+D² = (h_A²+h_B²)/2。",
        table(["样本","C 绝对值中位数 [四分位区间]","D 绝对值中位数 [四分位区间]","共同更大 / 相等 / 差异更大","I 绝对值中位数"],
              [[GROUPS[k],iq(t,"abs_C"),iq(t,"abs_D"),trio(t,["absC_minus_absD_pos","absC_minus_absD_zero","absC_minus_absD_neg"]),f(t["abs_I_median"])] for k,t in sums.items()]),
        table(["样本","共同平方和","差异平方和","平方和合计","共同平方和占比 R","留一最小–最大"],
              [[GROUPS[k],f"{t['sum_C_sq']:.9g}",f"{t['sum_D_sq']:.9g}",f"{t['sum_total_sq']:.9g}",pct(t["R"]),pct(t["loo_min"])+"–"+pct(t["loo_max"])] for k,t in sums.items()]),
        f"R 的分子为 ΣC²，分母为 Σ(C²+D²)，反映按变化幅度平方加权后的样本分配。按图对逐个计数则为 {s['absC_minus_absD_pos']}/{s['pairs']} 对共同分量更大。删去 {s['loo_min_ids']} 得到最小 R，删去 {s['loo_max_ids']} 得到最大 R。总平方和最大的一对 {s['largest_energy_id']} 贡献 {pct(s['largest_energy_share'])}，前三对合计贡献 {pct(s['top3_energy_share'])}；同时删去前三对后 R 仍为 {pct(s['R_without_top3'])}。留一范围是影响诊断，不能当作抽样置信区间。",
        "## 3　原领先幅度与改选门槛",
        "对每个提示词分别算原图分差 d₀ 和同一提示词的 Δ = b_A − b_B。有领先方时记 x = abs(d₀)、y = −sign(d₀) × Δ；y 为负扩大领先，y 为正缩小领先。只看调亮图时，y = x 打平，y > x 严格翻转。等权均值分差为 d₀ + Δ/2，达到平局需要 y = 2x。此处跨界使用该提示词的 Δ。",
        table(["样本","提示词","比较终点","同向扩大","反向未跨零","恰好打平","严格翻转","原图同分","零响应"],
              [[GROUPS[k],q,term,*[t[f"{q}_{mode}_{name}"] for name in ("same_direction","toward_short","exact_tie","flip","initial_tie","zero_delta")]]
               for k,t in sums.items() for q in ("Q0","Q1") for mode,term in (("L1","调亮"),("TTA","等权均值"))]),
        table(["提示词（全体）","原领先幅度中位数","绝对亮度响应中位数","朝平局移动对数","这些对的原领先中位数","朝平局位移中位数","逐对覆盖比例中位数"],
              [[q,f(s[q+"_lead_median"]),f(s[q+"_abs_delta_median"]),s[q+"_n_toward"],f(s[q+"_toward_lead_median"]),f(s[q+"_toward_move_median"]),pct(s[q+"_toward_fraction_median"])] for q in ("Q0","Q1")]),
        "![原领先幅度与朝平局方向的亮度位移](figure2_choice_boundaries.png)",
        "图 2　上排显示全部有原始领先方的样本，下排放大虚框内的小分差区域。原图同分时朝向平局的方向未定义，另列在面板内。逐对覆盖比例取朝平局移动样本的 y/x 后求中位数。",
        "Q0 下，008 调亮后刚好打平；039 是唯一严格翻转，朝平局移动 0.00378418 分，超过原领先的 0.00219727 分，但未达其两倍。Q1 下，029 从原图同分变为调亮和均值均选 B，单列为原图同分。",
        "在固定 10% 调亮下，能直接判定的限制是方向与跨界距离。要区分这种距离主要源于亮度不足、任务本身的候选差异，还是评分器分辨率，需要改变亮度幅度，并在可比较的原始分差层内复测；精度作用另需较高精度输出作对照。Q0、Q1 的选择均指模型选择。",
        "## 4　精度与精确的零",
        "评分脚本将模型置于推理路径并校验 eval 状态，环境与实际 smoke/full.log 记录 CUDA AMP；已核对版本哈希一致的官方 img_score.py，前向和特征点积均在 autocast 中执行。每个算子及其内部累加使用的精确 dtype **无法从现有文件确认**。",
        f"512 行的 native_dtype 都为 float16。评分脚本用 float(v) 提升返回值，再写入缓存与 CSV。逐个验证每个 CSV 十进制数与其读回的 binary64 值精确相等，并且均能无损转回 float16。分数范围 [{prec['minimum']}, {prec['maximum']}]，共有 {prec['unique_values']} 个唯一值。按大小排序，相邻唯一值有 {prec['unique_values']-1} 个间隔、{prec['distinct_gap_sizes']} 种长度，最小 {prec['gap_min']}，最大 {prec['gap_max']}。最小间隔出现 {prec['adjacent_gap_1_steps']} 次，二倍间隔出现 {prec['adjacent_gap_2_steps']} 次；逐间隔频数保存在 summary.csv。",
        "该分数范围跨过 0.25。低于 0.25 时，相邻可表示 float16 分数的间隔为 0.0001220703125；从 0.25 起为 0.000244140625。观测到的更大间隔可跨过未取到的可表示值。dtype 的判断还依据 CSV 的原始返回类型记录；仅看步长，无法确定所有前向运算的精度。",
        f"全体 G 的两个中间值均为 {s['G_lower_middle']}，所以中位数就是这个值。无冲突组的两个中间值为 {clean['G_lower_middle']} 与 {clean['G_upper_middle']}，取平均得到 {clean['G_median']}，恰为半个最小输出步长的负值。C、D 也包含除以二，因此衍生量可以落在原始分数网格之间。",
        table(["样本","单图 h 精确为零","图对 I 精确为零","图对 G 精确为零"],
              [[GROUPS[k],f"{t['h_zero']}/{t['images']}",f"{t['I_zero']}/{t['pairs']}",f"{t['G_zero']}/{t['pairs']}"] for k,t in sums.items()]),
        "## 5　三个按固定规则选出的图对"]
    target,picks=cases
    report.append(f"总变化幅度取单图交互的均方根 √(C²+D²)，样本中位数为 {f(target)} 分。按图对 ID 升序处理并列。三个规则选中 {len(set(r['pair_id'] for _,r in picks))} 个不同图对。原始分数表保留完整精度，比较表中的 → 依次为原图、调亮图、等权均值。")
    for label,r in picks:
        i=r["pair_id"]
        report.extend([f"### {label} · 图对 {i}",prompts[i]["Q0"],
                       table(["提示词","A 原图","A 调亮","B 原图","B 调亮","原图 → 调亮 → 均值"],
                             [[q,*[r[f"{side}_{q}_{l}"] for side in ("A","B") for l in ("L0","L1")],
                               " → ".join(r[q+"_"+col].replace("tie","同分") for col in ("pick0","pick1","pick_tta"))] for q in ("Q0","Q1")]),
                       f"h_A = {r['h_A']}，h_B = {r['h_B']}，C = {r['C']}，D = {r['D']}，I = {r['I']}，总变化幅度 = {f(r['total_amplitude'])}。"])
    report.extend(["## 复算文件","[逐对分解](pair_decomposition.csv)保留全部八个原始分数、单图变化、分差、选择与留一结果。[汇总数据](summary.csv)使用长表保存所有计数、分位数和步长频数。[README](README.md)给出离线命令与字段说明。"])
    (ROOT/"report.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")

