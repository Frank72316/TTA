"""Build the Chinese report from the frozen outputs. Counts keep their full precision in CSV."""
from collections import defaultdict
import numpy as np
from common import ROOT,read_csv,file_json,FAMILIES

def load(name):return read_csv(ROOT/(name+".csv"))
def val(x):return f"{float(x):.4f}".rstrip("0").rstrip(".") if x!="" else "—"
def small(x):return f"{float(x):.7f}" if x!="" else "—"
def interval(ab):return "［"+ "，".join(f"{v:+.3f}" for v in ab)+"］"
def ids(rs):return "、".join(r["pair_id"] for r in rs) or "无"
def table(head,rows):return "\n".join(["| "+" | ".join(head)+" |","|"+"---|"*len(head)]+["| "+" | ".join(map(str,r))+" |" for r in rows])
def main():
    e=file_json(ROOT/"endpoint.json");n=e["n"];src=e["source_contrasts"];runs=file_json(ROOT/"runs.json");fr=file_json(ROOT/"freeze.json")
    tr={(r["strategy"],r["original_state"]):r for r in load("transitions")};sums={r["strategy"]:r for r in load("strategy_summary")}
    pc=load("pair_contributions");g=load("margin_groups");dc=load("direction_counts");old=load("old_case_checks");combos=load("combination_results")
    for r in pc:
        for k in r:
            if k.startswith(("BT_","BC_","TT_","BB_","CC_")):r[k]=int(r[k])
    extra=[r for r in pc if r["original_state"]=="wrong" and r["BT_correct"] and not r["BC_correct"]]
    photo=[r for r in extra if not (r["BB_correct"] or r["CC_correct"])]
    harm=[r for r in pc if r["original_state"]=="correct" and r["BC_correct"] and r["BT_wrong"]]
    both=[r for r in extra if r["TT_correct"]]
    by_id=defaultdict(list)
    for r in dc:by_id[r["pair_id"]].append(r)
    n_dir=lambda rs:sum(int(r["n_changed"])>0 for r in rs)
    num_dir_multi=sum(n_dir(rs)>=2 for rs in by_id.values());one_cell=sum(sum(int(r["n_changed"]) for r in rs)==1 for rs in by_id.values())
    repeat=sum(any(int(r["n_changed"])>=2 for r in rs) for rs in by_id.values());four=sum(any(int(r["n_changed"])==4 for r in rs) for rs in by_id.values())
    single_suffices=sum(any(int(r["O_plus_T_changes"]) for r in rs) for rs in by_id.values())
    any_changed=sum(int(x["any_BT_BC_change"]) for x in g);first=g[0];tie_g=g[-1]
    delta=e["bt_minus_bc_pp"];net=e["net_expected_correct"];ci=e["ci_pp"]
    go=lambda a,b:src[a+"_to_"+b]["expected_pairs"]
    order=["O","BT","BC","TT","BB","CC"];zh={"correct":"对","wrong":"错","tie":"同分"}
    de=lambda r:float(r["bt_minus_bc_expected_correct"])
    pos=[r for r in pc if de(r)>0];neg=[r for r in pc if de(r)<0]
    lines=["# 轻微平移为什么会纠正一些排序，也会破坏另一些？",
           f"① 在 {n} 个独立提示词上，BT 至少一次纠正了 {len(extra)} 对 BC 从未纠正的原始错例，其中 {len(photo)} 对连 BB、CC 也未能纠正，{len(both)} 对单用平移策略 TT 就能纠正。",
           f"② 均匀抽取搭配时，BT 比 BC 多纠正 {val(go('wrong','correct'))} 对原始错例，原本选对者的严格改错数之差为 {val(go('correct','wrong'))} 对；计入同分去向后，净变化为 {net:+.4f} 个期望正确图对，即 {delta:+.3f} 个百分点（配对 95% 区间 {interval(ci)}）。",
           f"③ 原始小分差和可重复的方向效应同时可见：BT 或 BC 的 {any_changed} 个改判案例中，{first['any_BT_BC_change']} 个落在原始领先幅度最小的四分组；BT 改判的 {len(by_id)} 对中，{num_dir_multi} 对由多个方向触发，{repeat} 对在同一方向跨亮度档重复。",
           "## 独立复验 · 三次评分能选对多少",
           f"排除旧提示词与图像后，validation 有 {fr['eligible_normalized']} 条合格规范化文本。以种子 2026 冻结其中 {n} 条，每条只保留一对严格人工排名。原图选对 {sums['O']['correct_cells']} 对、选错 {sums['O']['wrong_cells']} 对、同分 {sums['O']['tie_cells']} 对。B、C 分别为亮度与对比度，T 为平移。每种具体搭配先把原图和两个增强版本等权平均，严格按分差符号选图。人类赢家始终取自原文和原图。",
           table(["策略","选对 / 总搭配判定","期望选对数","严格选对率","期望选错数","期望同分数","至少一次改判的图对"],[
                [st,r["correct_cells"]+" / "+r["total_cells"],val(r["correct_expected"]),f"{float(r['correct_percent']):.3f}%",val(r["wrong_expected"]),val(r["tie_expected"]),r["changed_pairs"]] for st in order for r in [sums[st]]]),
           f"BT 与 BC 各有 {16*n:,} 次搭配判定，独立抽样单位有 {n} 个。每对先取选对比例之差，再按整对重采样 10,000 次，固定种子 2026。主终点区间换算成期望对数为 {interval(e['ci_expected_correct'])}。"+("区间跨过零，本次复验无法确定哪种策略的总体严格选对率更高。" if ci[0]<=0<=ci[1] else "本次固定设计的区间没有跨过零。"),
           "![各策略的纠正、改错与同分变化](figures/strategy_transitions.png)",
           "### 按原图判定对账",
           "每格依次列最终「选对 / 选错 / 同分」的期望对数。每格的三项加起来等于该行的图对数，小数来自在预设搭配中均匀取样。",
           table(["原图状态","图对数",*order[1:]],[[zh[base],tr["O",base]["pairs"],*[" / ".join(val(tr[st,base][x+"_expected"]) for x in ["correct","wrong","tie"]) for st in order[1:]]] for base in ["wrong","correct","tie"]]),
           f"对严格选对数的贡献，来自原始错例 {go('wrong','correct'):+.4f} 对、原始对例 {go('correct','correct'):+.4f} 对、原始同分 {go('tie','correct'):+.4f} 对，合计 {net:+.4f} 对。原始对例转成同分的 BT−BC 差为 {go('correct','tie'):+.4f} 对。原始错例新增纠正的配对区间为 {interval(src['wrong_to_correct']['ci_expected_pairs'])} 对；原始正确样本新增严格改错的区间为 {interval(src['correct_to_wrong']['ci_expected_pairs'])} 对。",
           "### 哪些样本带来收益和损失",
           "下表逐对列出 BT 补到、BC 从未纠正的原始错例，以及各策略严格选对的搭配数。这些病例支持平移提供了额外纠正；单用平移已能达到这一结果，因而缺少把收益归为亮度与平移协同的证据。",
           table(["图对","原始 A−B","人工赢家","BT /16","BC /16","TT /6","BB /6","CC /6"],[[r["pair_id"],small(r["d0"]),r["human"],*[r[s+"_correct"] for s in ["BT","BC","TT","BB","CC"]]] for r in extra]),
           f"原图正确、BC 至少一次保持正确而 BT 至少一次严格改错的有 {len(harm)} 对；其中 {sum(r['BT_wrong']<16 for r in harm)} 对仅部分搭配改错，{sum(r['BT_wrong']==16 for r in harm)} 对全部 16 种改错。BC 全部保持正确的严格子集为 {ids([r for r in harm if r['BC_correct']==16])}。",
           table(["图对","原始 A−B","人工赢家","BC 选对 /16","BT 改错 /16","BT 同分 /16","BT−BC 期望选对数"],[[r["pair_id"],small(r["d0"]),r["human"],r["BC_correct"],r["BT_wrong"],r["BT_tie"],val(de(r))] for r in harm]),
           f"BT−BC 逐对贡献中，{len(pos)} 对为正，合计 {sum(map(de,pos)):+.4f} 个期望正确图对；{len(neg)} 对为负，合计 {sum(map(de,neg)):+.4f}，余下 {n-len(pos)-len(neg)} 对贡献为零。"]
    for label,rr,func in [("正",pos,max),("负",neg,min)]:
        if rr:
            peak=func(map(de,rr));leaders=[r for r in rr if de(r)==peak]
            lines.append(f"最大{label}贡献为 {ids(leaders)}，各 {peak:+.4f} 对（对主终点各贡献 {100*peak/n:+.3f} 个百分点，占{label}向总量的 {100*peak/sum(map(de,rr)):.1f}%）。单独去掉其中一对，余下 {n-1} 对的 BT−BC 为 {100*(net-peak)/(n-1):+.3f} 个百分点。")
    lines+=["## 原始领先幅度与平移位移",
            "位移为单个增强版本的 A−B 分差减原图分差。表中位移取绝对值，再对每组的图对与四个版本取中位数。「改判」包括进入或离开同分，「翻转」只计严格异号。",
            table(["原始领先幅度组","图对数","领先幅度范围","领先中位数","亮度位移","对比度位移","平移位移","平移大 / 等 / 小¹","BT / BC 改判对数","BT / BC 翻转对数","BT−BC 百分点"],
                [["第"+r["group"][1:]+"组" if r["group"]!="original_tie" else "原同分",r["n"],(small(r["min_abs_d0"])+" 至 "+small(r["max_abs_d0"])) if int(r["n"]) else "—",small(r["median_abs_d0"]),
                  *[small(r["median_abs_delta_"+s]) for s in "BCT"]," / ".join(r["T_"+z+"_C"] for z in ["gt","eq","lt"]),r["BT_changed_pairs"]+" / "+r["BC_changed_pairs"],r["BT_strict_flip_pairs"]+" / "+r["BC_strict_flip_pairs"],val(r["BT_minus_BC_pp"])] for r in g]),
            "¹ 先在同一对图内分别取平移与对比度四个版本的绝对位移中位数，再比较两者。",
            "![按原始分差比较位移与改判](figures/margin_groups.png)",
            f"四个非平局组中，有 {sum(float(r['median_abs_delta_T'])>float(r['median_abs_delta_C']) for r in g[:4])} 组的典型平移位移大于对比度位移。在原本就有赢家的图对中，{first['any_BT_BC_change']}/{any_changed-int(tie_g['any_BT_BC_change'])} 个 BT 或 BC 改判案例集中在最低四分组。较大的分数位移描述模型的敏感性；准确性还取决于移动方向与人工赢家。",
            "## 同一图对能否跨参数重现",
            f"以下每个方向的分母都是四个亮度档。{len(by_id)-num_dir_multi} 对只在一个方向触发，{num_dir_multi} 对在多个方向触发；{one_cell} 对只在 16 格中的一格触发，{four} 对至少有一个方向在四档亮度下都触发。仅用原图和单方向平移等权平均，{single_suffices}/{len(by_id)} 对也会在至少一个方向改判。",
            table(["图对","原图","左 /4","右 /4","上 /4","下 /4","涉及亮度档数","原图＋单平移会改判的方向"],[
                [i,zh[rs[0]["original_state"]],*[r["n_changed"] for r in rs],len({v for r in rs for v in r["changed_brightness"].split()}),
                 "、".join(d for d,r in zip("左右上下",rs) if int(r["O_plus_T_changes"])) or "无"] for i,rs in by_id.items()]),
            "对每个搭配，三次评分后的分差等于原始分差加上两个增强位移之和的三分之一。原图有赢家时，位移之和须与原分差反向且幅度超过其三倍，才会严格换赢家。等于阈值时打平。以下各取最大正、负贡献样本中第一个实际改变选择的 BT 搭配。"]
    shift={(r["pair_id"],r["view"]):float(r["delta"]) for r in load("single_view_shifts")}
    for rr,func in [(pos,max),(neg,min)]:
        if not rr:continue
        i=next(r["pair_id"] for r in rr if de(r)==func(map(de,rr)))
        r=next((r for r in combos if r["pair_id"]==i and r["strategy"]=="BT" and int(r["changed"])),None)
        if r:
            b,t=r["view1"],r["view2"];d0=float(r["d0"])
            lines.append(f"- {i}（人工赢家 {r['human']}）在 {b}、{t} 下，原分差 {d0:+.7f}，亮度位移 {shift[i,b]:+.7f}，平移位移 {shift[i,t]:+.7f}，三分均值分差 {float(r['margin']):+.7f}。原图选 {r['original_pick']}，组合选 {r['pick']}。")
    lines+=["每个方向的严格翻转、最终选对、选错与同分计数见 [方向明细](direction_counts.csv)。同一图对的方向和亮度档共享两张候选图。数据足以描述这些案例，尚不足以判断哪类构图普遍受益。",
            "## 旧 64 对 · 案例复核",
            "从 P2 的 2,176 条原始分数独立枚举十项策略，每对 120 种搭配。仅 003、008、053、005、031 至少一次改变原始选择。003 原图选错，TT 纠正 4/6、BT 纠正 4/16；008 原图选对，BT 改错 16/16。下表显示原图与单一平移等权平均的选择，便于对照方向。"]
    op={r["pair_id"]:r for r in read_csv(ROOT.parent/"P1/baseline/pairs.csv")}
    lines.append(table(["旧图对","人工赢家","原始 A−B","原图选择","原图＋左","原图＋右","原图＋上","原图＋下"],[[i,op[i]["human_winner_P0"],small(rs[0]["d0"]),"A" if float(rs[0]["d0"])>0 else "B",*[r["pick"] for r in rs]] for i in ["003","008","053","005","031"] for rs in [[r for r in old if r["pair_id"]==i and r["method"]=="O+T"]]]))
    lines+=["[打开五个旧案例](old_cases/index.html)。每例含原文、原始图像、模型看到的 224 × 224 输入、BT 的 4 × 4 分差与判定，以及四种单方向结果。可直接比较边缘纹理和位置的改变；当前实验没有测量模型关注的图像区域。旧案例未并入独立样本估计。",
            "## 运行与验收",
            f"在 {runs[0]['gpu']['name']} 单卡上运行 HPS v2.1，官方验证变换和 CUDA fp16 autocast，原生分数为 float16，保存可精确往返的十进制，再用 float64 汇总。2 对冒烟新增 52 条并与官方评分入口逐项相等；全量阶段从检查点命中 {runs[1]['checkpoint_hits']} 条，独立样本从旧评分命中 {sum(r['old_score_hits'] for r in runs)} 条。新增唯一评分 {sum(r['unique_new_scores'] for r in runs):,} 条；冒烟验证另做 {sum(r['reference_check_extra_forwards'] for r in runs)} 次图像前向。",
            f"冒烟墙钟 {runs[0]['wall_seconds']:.1f} 秒，全量续跑 {runs[1]['wall_seconds']:.1f} 秒，合计 {sum(r['wall_seconds'] for r in runs[:2]):.1f} 秒。生产评分 CUDA event 累计 {sum(r['gpu_score_seconds'] for r in runs[:2]):.1f} 秒，含文本与图像编码及点积；额外官方数值核对计入墙钟。",
            f"独立核验 {n} 个提示词、{2*n} 张图、每图 13 种视图、{e['raw_scores']:,} 条唯一评分。{e['identity_checks']:,} 个三次组合均通过分差恒等式与整数精确判定；配对区间可独立复算。样本清单哈希为 `{fr['pairs_sha256']}`。结论限于本次 validation 独立提示词、HPS v2.1、3% 平移及预设的参数池。",
            "抽样与指标见 [冻结协议](protocol.md)，完整操作见 [README](README.md)。[原始评分](scores.csv) · [逐对策略](pair_strategy_results.csv) · [逐对贡献](pair_contributions.csv) · [全部搭配](combination_results.csv) · [验收记录](verification.json)。"]
    (ROOT/"report.md").write_text("\n\n".join(lines)+"\n",encoding="utf8")
if __name__=="__main__":main()

