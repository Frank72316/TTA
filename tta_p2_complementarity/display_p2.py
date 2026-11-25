from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from study import *
from views import original,make_view

def chart(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family=next((v for v in ("Noto Sans CJK SC","Microsoft YaHei","Noto Sans CJK JP","PingFang SC") if v in fonts),"DejaVu Sans")
    plt.rcParams.update({"font.family":[family,"DejaVu Sans"],"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False})
    fig,ax=plt.subplots(figsize=(9.3,5.8),layout="constrained")
    labels=[]
    for j,r in enumerate(summ):
        name=r["strategy"];x=float(r["delta_vs_O_pp"])
        lo,hi=[float(r[k]) for k in ("delta_vs_O_lo","delta_vs_O_hi")]
        color="#14766f" if name in PRIMARY else "#51627c" if name in PURE else "#958575"
        ax.hlines(j,lo,hi,color=color,lw=1.8)
        ax.scatter(x,j,s=50,color=color,zorder=3)
        labels.append(name+("　原图" if name=="O" else "　预设混合" if name in PRIMARY else "　探索混合" if name in MIX else "　单类"))
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.axvline(0,color="#586471",ls="--",lw=1)
    ax.grid(axis="x",alpha=.15)
    ax.set_xlabel("严格选对率相对原图的变化（百分点）")
    ax.set_title("同一三次预算下的人工赢家命中率",loc="left",fontsize=16,pad=24)
    ax.text(0,1.015,"点为版本组合上的平均差，线为按 64 个完整图对配对重采样的 95% 区间",transform=ax.transAxes,fontsize=9)
    dest=ROOT/"figures";dest.mkdir(exist_ok=True)
    fig.savefig(dest/"strategies.png",dpi=180)
    plt.close(fig)

def cases(per,combo,banks):
    index={(r["pair_id"],r["strategy"]):r for r in per}
    ps={r["pair_id"]:r for r in pairs()}
    summary=[]
    for event,orig,ending in [("纠正","wrong","correct"),("改错","correct","wrong")]:
        cands=[]
        for i,p in ps.items():
            if index[i,"O"]["base_outcome"]!=orig:continue
            support={s for s in PURE+MIX if int(index[i,s][ending+"_n"])>0}
            for s in set(MIX)&support:
                # Distinguish globally unique, unique vs any pure, and vs own components.
                rank=(0 if len(support)==1 else 1 if not support.intersection(PURE) else
                      2 if not support.intersection([s[0]*2,s[1]*2]) else
                      3 if not support.intersection(["BB","CC","BC"]) else 4)
                cands.append((rank,-float(index[i,s][ending+"_rate"]),s not in PRIMARY,i,s))
        if not cands:continue
        rank,_,_,i,st=min(cands)
        subset=[r for r in combo if r["pair_id"]==i and r["strategy"]==st and r["result"]==ending]
        mid=np.median([float(r["d_aug"]) for r in subset])
        row=min(subset,key=lambda r:(abs(float(r["d_aug"])-mid),r["view1"],r["view2"]))
        vs=["O",row["view1"],row["view2"]]
        canvas=Image.new("RGB",(3*234,2*237+26),"#f2f3f4");draw=ImageDraw.Draw(canvas)
        for col,v in enumerate(vs):
            draw.text((col*234+12,4),v,fill="black")
            for idx,side in enumerate(("A","B")):
                rgb=original(ps[i],side)
                im=ImageOps.contain(make_view(rgb,v)[0],(220,220))
                x=col*234+(234-im.width)//2;y=26+idx*237
                canvas.paste(im,(x,y));draw.text((col*234+4,y+220),side,fill="black")
        file=f"figures/case_{event}_{i}.png";canvas.save(ROOT/file)
        case=dict(event=event,case_type=["该混合独有（其余九种均无）","四种单类均无","两个组成单类均无","光度策略均无","与其他策略共享"][rank],
                  pair_id=i,strategy=st,P0=ps[i]["P0"],human=ps[i]["human_winner_P0"],view1=vs[1],view2=vs[2],
                  d0=row["d0"],d_aug=row["d_aug"],base_pick=row["base_pick"],pick=row["pick"],image=file)
        for side in ("A","B"):
            for j,v in enumerate(vs):case[side+str(j)]=banks[i,side,v]
        for s in PURE+MIX:case[s+"_count"]=index[i,s][ending+"_n"]+"/"+index[i,s]["n"]
        summary.append(case)
    return summary

def main():
    from analyze_p2 import input_scores
    _,bank,_=input_scores()
    per=read_csv(ROOT/"pair_strategy_results.csv"); combos=read_csv(ROOT/"combination_results.csv")
    chart(read_csv(ROOT/"strategy_summary.csv"))
    selected=cases(per,combos,bank)
    if selected:write_csv(ROOT/"cases.csv",selected)
    print([(c["event"],c["case_type"],c["pair_id"],c["strategy"]) for c in selected])

if __name__=="__main__":main()

