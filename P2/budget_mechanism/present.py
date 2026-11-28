"""Render the report and its two figures using only the exported analysis tables."""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analyze import CELLS, FAMILIES, FOCUS, ROOT, read


UNIT = 1/8192
BLUE, RED, INK = "#176a85", "#ad5033", "#243642"


def source(name):
    return read(ROOT / (name + ".csv"))


def score(value):
    return float(value)/UNIT


def u(value):
    return f"{score(value):+.3f}".rstrip("0").rstrip(".") if float(value) else "0"


def table(headers, records):
    return ["", "| " + " | ".join(headers) + " |", "|" + "---|"*len(headers),
            *["| " + " | ".join(map(str,row)) + " |" for row in records], ""]


def make_figures(by_id, chains, examples):
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "PingFang SC"]
    cjk = next((f for f in candidates if f in fonts),None)
    if not cjk:
        raise RuntimeError("Install a Simplified Chinese font, e.g. Noto Sans CJK SC, before rendering.")
    plt.rcParams.update({"font.family":cjk,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.unicode_minus":False,"font.size":10})
    dest = ROOT / "figures"
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(10,4.4),sharey=True,layout="constrained")
    for ax, i in zip(axes,FOCUS):
        for f, color, offset, marker in [("B",BLUE,-0.10,"o"),("C",RED,0.10,"s")]:
            rs = sorted((r for r in by_id[i] if r["view"][0]==f),key=lambda r:float(r["factor"]))
            chain = next(r for r in chains if r["pair_id"]==i and r["strategy"]==f)
            y = [score(r["margin"]) for r in rs] + [score(chain["M5"])]
            ax.scatter(np.arange(5)+offset,y,c=color,marker=marker,s=55,zorder=3)
        ax.set_xticks(range(5),[f'{float(r["factor"]):.2f}' for r in rs]+["五次均值"])
        ax.axvline(3.5,color="#c7cdd1",lw=0.8)
        ax.axhline(score(rs[0]["m0"]),color="#777",ls=":",lw=1.3)
        ax.axhline(0,color=INK,lw=1)
        ax.set_title(i+"   人工选 "+rs[0]["human_winner_P0"],loc="left",fontweight="bold")
        ax.grid(axis="y",alpha=.12)
    axes[0].set_ylabel("人工赢家领先幅度（u）")
    axes[0].legend(handles=[Line2D([],[],ls="",marker="o",color=BLUE,label="亮度"),
                            Line2D([],[],ls="",marker="s",color=RED,label="对比度"),
                            Line2D([],[],ls=":",color="#777",label="原图")],loc="upper left",frameon=False)
    fig.suptitle("四档单视图的分差与加入原图后的五次均值",fontweight="bold")
    fig.savefig(dest/"parameter_margins.png",dpi=165)
    plt.close(fig)

    combos = source("complement_checks")
    fig, axes = plt.subplots(1,3,figsize=(11.4,4.4),sharey=True,layout="constrained")
    for ax, r in zip(axes,examples):
        values = sorted(score(row[key]) for row in combos if row["pair_id"]==r["pair_id"] for key in ("M","M_complement"))
        assert len(values)==36
        ax.axhline(0,color=INK,lw=1)
        ax.axhline(score(r["pure_B"]),color=BLUE,ls="--")
        ax.axhline(score(r["pure_C"]),color=RED,ls=":")
        for x,y in enumerate(values,1):
            ax.scatter(x,y,s=20,color=("#128365" if y>0 else "#8d979c" if y<0 else "#bd861b"),zorder=3)
        ax.set_title(f'{r["pair_id"]}   {r["correct"]}/36 选对',loc="left",fontweight="bold")
        ax.set_xticks([1,12,24,36])
        ax.grid(axis="y",alpha=.13)
    axes[0].set_ylabel("混合策略有向分差（u）")
    axes[0].legend(handles=[Line2D([],[],ls="--",color=BLUE,label="纯亮度"),Line2D([],[],ls=":",color=RED,label="纯对比度")],
                   loc="lower left",frameon=False)
    fig.suptitle("三个混合独有案例的全部 36 种搭配",fontweight="bold")
    fig.supxlabel("同一图对内按分差从小到大排列的参数组合",fontsize=10)
    fig.savefig(dest/"mixed_margins.png",dpi=165)
    plt.close(fig)


def main():
    cases = source("case_migration")
    chains = source("parameter_chains")
    examples = source("mixed_cases")
    probs = source("pair_probabilities")
    summary = source("factorial_summary")
    coverage = source("coverage")
    by_id = defaultdict(list)
    for r in source("parameter_margins"):
        by_id[r["pair_id"]].append(r)
    make_figures(by_id,chains,examples)
    lines = ["# 评分预算为什么会改变亮度与对比度的纠错覆盖关系", "",
             "① 三次评分时，N078 的对比度只有 0.90＋0.95 能纠正；N137 只有 0.95＋1.10 和 1.05＋1.10 能纠正。取齐四档后，前者加入的两个高档、后者尤其是 0.90 档拉低了分差，两对都严格选错。四档亮度中两者的 1.10 档都支持人工赢家，五次均值均保持正值。", "",
             "② 把原图权重固定在 1/3 或 1/5，从选两档改为取齐四档，纯 B、纯 C 共有的可纠正图对均从 5 对降到 0 对。权重影响临界案例；例如 N078 的四档亮度在 1/3 时同分，在 1/5 时选对。把门槛提高到至少 25% 的组合能纠正，三次评分时就已经出现 B 独有 3 对、C 独有 2 对。", "",
             "③ 五次混合在 N015、N100、N156 分别只以 11/36、6/36、2/36 的概率纠正。均匀选搭配在全体 256 对上期望纠正 2.167 对、改错 2.528 对，选对率比原图和同预算纯 C 都低 0.141 个百分点。这批图支持“个别搭配可纠正”。", "",
             "## 数据与判定", "",
             "使用 ImageRewardDB validation 已研究过的 256 个不同提示词，按 P0 和原始人工排序对齐 A/B。原图 175 对选对、81 对选错、0 对同分。从旧评分中读取双方原图和亮度、对比度各四档的 4,608 条分数。两个方法的实际因子均为 0.90、0.95、1.05、1.10。", "",
             "有向分差 m 为人工赢家分数减另一侧分数。下文用 **u = 1/8192 = 0.0001220703125 HPS 分**显示小数，分差为正、零、负分别对应严格选对、同分、严格选错。CSV 保留 HPS 分的 float64 数值；每个具体组合先用未除以分母的和判符号，再累计概率。", "",
             "## A　五个共有错例去了哪里", "",
             "下表格内依次为选对／同分／选错的组合数。三个数各除以该策略的搭配总数即为概率。三次评分的 B、C、BC 分母分别为 6、6、16；五次为 1、1、36。表中七对是两个等权预算下曾被任一策略纠正的全部原始错例。"]
    lines += table(["图对","人工赢家","原图 m₀（u）","三次 B","三次 C","三次 BC","五次 B","五次 C","五次 BC"],
                   [[r["pair_id"],r["human_winner_P0"],u(r["m0"]),
                     *["/".join(r[f"K{k}_{f}_{v}"] for v in ("correct","tie","wrong")) for k in (3,5) for f in FAMILIES]] for r in cases])
    lines += ["三次共有的 N078、N137 在五次时均只剩纯 B 和混合能纠正；N015、N100、N156 均只剩混合。两对纯 C 独有的 N020、N164 在两个预算下均保留。上述计数、独有 ID 和期望选对数均与上一份关系表一致；原表以计数保存三次共有组，本次从逐对分数找回全部五个 ID。", "",
              "![两个案例四档单视图及全池分差](figures/parameter_margins.png)", "",
              "图 1 左四列各自只使用一个增强版本。最右一列是原图与同一方法四个版本的均值。点在零线以上表示该策略选中人工赢家。", "",
              "### 从任选两档到取齐四档", "",
              "每行先用原图＋所列两档评分。表格列出全部六种搭配的三次均值，单位均为 u。"]
    pairspec = sorted({r["selected"][1:4]+"+"+r["selected"][-3:] for r in chains})
    lines += table(["图对／方法",*[f"{int(s[:3])/100:.2f}+{int(s[-3:])/100:.2f}" for s in pairspec],"五次均值"],
                   [[i+"／"+f,*[u(r["M3"]) for r in chains if r["pair_id"]==i and r["strategy"]==f],
                     u(next(r["M5"] for r in chains if r["pair_id"]==i and r["strategy"]==f))] for i in FOCUS for f in "BC"])
    lines += ["对每套三次能纠正的搭配，下表把剩余两档逐个加入原有三项均值。列中的变化量是单独加入该档时的四项均值减三项均值，两列共用同一起点。最后一列另算同时加入两档的五项均值。"]
    lines += table(["图对／方法","已选两档","三次领先（u）","余档一／单加变化（u）","余档二／单加变化（u）","最终分差（u）"],
                   [[r["pair_id"]+"／"+r["strategy"],r["selected"],u(r["M3"]),r["extra1"]+" / "+u(r["extra1_change_if_alone"]),
                     r["extra2"]+" / "+u(r["extra2_change_if_alone"]),u(r["M5"])] for r in chains if float(r["sum3"])>0])
    lines += ["N078 的纯 C 从低两档合计领先 2 u（三次均值 +⅔ u），遇到高两档的单视图分差 −9、−4 u，五次均值变成 −2.2 u。纯 B 的两档高亮度分别贡献 +6、+13 u，全部取齐后仍领先 1 u。", "",
              "N137 的 C 0.90 单视图分差为 −23 u。以 0.95＋1.10 起步时，余下 1.05 的 +10 u 未能补足这一损失；以 1.05＋1.10 起步时，补入的两个低档都降低均值。纯 C 最终为 −2.4 u，纯 B 的 1.05、1.10 则分别为 +7、+22 u，全池均值 +3 u。", "",
              "对照案例 N020 的四档纯 B 分差都为负，C 1.10 单档为 +29 u，纯 C 全池均值为 +5.8 u。N164 的纯 C 全池只领先 +0.2 u，对原图权重更敏感。所有七对的原文、逐档分差和位移列在文末附表。", "",
              "## B　固定权重与固定版本数", "",
              "用 M = w·m₀ + (1−w)·所选增强分差均值。两个等权格复现原策略。另两个诊断格保持指定的原图权重；两档纯策略无放回任选两档，四档则取齐，混合在两族平分槽位。每个提示词只占总体的 1/256。"]
    lines += table(["增强数","原图权重","设计","策略","严格选对率","期望纠正","期望改错","对→同分","错→同分"],
                   [[r["views"],r["weight"],"等权" if r["design"]=="equal" else "诊断",r["strategy"],f'{float(r["accuracy_pct"]):.4f}%',
                     *[f'{float(r[col]):.4f}' for col in ("fixes","harms","correct_to_tie","wrong_to_tie")]] for r in summary])
    lines += ["**固定权重看版本数。** 原图权重 1/3 时，取齐四档让纯 B／C 的可达集合从 5／7 对缩至 1／1 对；权重 1/5 时，从 5／8 缩至 2／2 对。两条对照中共有组都消失。四档纯策略只有一套搭配，原先有利的两档不能单独决定结果。", "",
              "**固定版本数看权重。** 只选两档时，降低原图权重后，纯 C 新增 N111，B 的可达集合不变。取齐四档时，权重从 1/3 降到 1/5，使 N078 的纯 B 由平到对、N164 的纯 C 由错到对；N137 的纯 B 和 N020 的纯 C 在两种权重下都选对。", "",
              "下表给出七个案例的逐格概率。每个格内是“严格选对数／同分数”，两项概率分别以同行的组合总数为分母。"]
    lookup = {(r["pair_id"],int(r["views"]),r["weight"],r["strategy"]):r for r in probs}
    ids = [r["pair_id"] for r in cases]
    for_table = []
    for n,q in CELLS:
        for f in FAMILIES:
            rs = [lookup[i,n,f"1/{q}",f] for i in ids]
            for_table.append([n,f"1/{q}",f,rs[0]["combinations"],*[r["correct"]+"/"+r["tie"] for r in rs]])
    lines += table(["增强数","原图权重","策略","分母",*ids],for_table)
    lines += ["### 覆盖门槛改变了哪些判断", "",
              "下表仅看 81 个原始错例。最后一列是达到相同门槛的混合独有图对数。无阈值时，前面的总体表直接给出严格选对概率在全部原始错例上的和（期望纠正列），逐对概率可读取明细。"]
    lines += table(["增强数","原图权重","纠正门槛","纯 B","纯 C","共有","仅 B","仅 C","仅混合"],
                   [[r["views"],r["weight"],">0" if r["criterion"]=="any" else "≥25%",*[r[key] for key in ("B","C","both","B_only","C_only","mix_only")]] for r in coverage])
    lines += ["三次等权时，把门槛从任一成功提高到四分之一，N015、N078、N100 变成纯 B 独有，N137 是唯一共有。五次等权纯策略各只有一个结果，两个门槛给出相同的纯策略覆盖；混合独有从三对缩到仅 N015。", "",
              "## C　为什么混合独有的成功率最多一半", "",
              "令 b、c 为两族各四个单视图有向分差之和，全池纯策略为 B₅=(m₀+b)/5 和 C₅=(m₀+c)/5。混合搭配 S 选两档 B、两档 C，S′ 取两族各自剩下的两档。两套参数不重叠且并集为八档，因此", "",
              "$$\\frac{M_S+M_{S'}}2=\\frac{2m_0+b+c}{10}=\\frac{B_5+C_5}2.$$",
              "", "36 套恰好分成 18 对互补搭配，每对均值相同，全部搭配的均值也等于右侧。若 B₅、C₅ 均不大于零，每对至多一个分差严格为正，成功数至多 18，概率至多 50%。两项同分也满足此界。这个上限使用了等权、每族取半池和剩余半池正好互补三个条件。", ""]
    checks = source("identity_summary")
    lines += table(["数值检验","比较次数","最大残差／超额","违反次数"],
                   [[label,r["comparisons"],f'{float(r["max_residual"]):.3g}',r["violations"]] for label,r in zip(
                       ("36 套均值与两纯均值","每对互补均值与两纯均值","两纯均未选对时，每对成功数超出 1 的部分"),checks)])
    lines += ["前两行残差单位为 HPS 分，float64 检验容差 10⁻¹⁵。第三行以严格符号计数，容差为零。逐项检查 256 个图对，其中 79 对满足两纯策略均未严格选对，进入 79 × 18 次条件核验。", "",
              "![三个案例的混合分差和两个全池纯策略](figures/mixed_margins.png)", "",
              "图 2 每个点代表从两族各取两档的一套组合，正值选对，零值同分。同一案例的分差已排序，虚线和点线各给出一种纯策略。下表分差单位为 u。"]
    lines += table(["图对","纯 B","纯 C","混合最小／均值／最大","选对／同分／选错","纠正概率"],
                   [[r["pair_id"],u(r["pure_B"]),u(r["pure_C"])," / ".join(u(r[k]) for k in ("minimum","mean","maximum")),
                     "/".join(r[k] for k in ("correct","tie","wrong")),f'{100*float(r["p_correct"]):.2f}%'] for r in examples])
    lines += table(["图对","一套选对搭配","分差（u）","其互补搭配","分差（u）"],
                   [[r["pair_id"],r["choice"],u(r["M"]),r["complement"],u(r["M_complement"])] for r in examples])
    lines += ["三个案例在均匀取搭配时合计贡献 19/36 个期望纠正。五次混合在全部 81 个原始错例上期望纠正 2.1667 对；175 个原始正确样本中期望改错 2.5278 对、转为同分 0 对。另有 0.0833 对原始错误转为同分。期望严格选对为 174.6389 对，低于原图和纯 C 的 175 对，高于纯 B 的 174 对。这些是冻结参数池和这批提示词上的结果。", "",
              "## 附表　全部曾纠正错例的参数与原始提示词", "",
              "每格为单视图有向分差／相对于 m₀ 的位移，单位均为 u。[逐参数 CSV](parameter_margins.csv) 同时保留 A/B 图像标识、原始分数和人工名次。"]
    lines += table(["图对","方法","0.90","0.95","1.05","1.10"],[[i,f,*[u(r["margin"])+" / "+u(r["shift"]) for r in by_id[i] if r["view"][0]==f]]
                                                           for i in ids for f in "BC"])
    lines += [f'- **{r["pair_id"]}**　{r["P0"].strip()}' for r in cases]
    lines += ["", "## 下一轮设计建议", "",
              "预先冻结主指标为“均匀选一套参数时，相对同预算最佳纯策略的严格选对率差”。它给每个提示词同样权重，也会把偶尔纠正、改错和转为同分的代价一并计入。"]
    (ROOT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
