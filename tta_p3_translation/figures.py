from common import ROOT,read_csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main():
    out=ROOT/"figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":["Noto Sans CJK JP","DejaVu Sans"],"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False,
                        "font.size":11,"axes.labelcolor":"#213544","text.color":"#213544","axes.titlelocation":"left"})
    tr={(r["strategy"],r["original_state"]):r for r in read_csv(ROOT/"transitions.csv")}
    order=["BT","BC","TT","BB","CC"];pal=["#15786E","#B84C34","#60789A","#998196"]
    vals=lambda st,old,new:float(tr[st,old][new+"_expected"])
    series=[[vals(st,"wrong","correct") for st in order],[vals(st,"correct","wrong") for st in order],
            [sum(vals(st,s,"tie") for s in ["correct","wrong"]) for st in order],
            [sum(vals(st,"tie",s) for s in ["correct","wrong"]) for st in order]]
    fig,ax=plt.subplots(figsize=(10.6,5.3),layout="constrained")
    for j,(y,label,color) in enumerate(zip(series,["错 → 对","对 → 错","有赢家 → 同分","原同分 → 有赢家"],pal)):
        ax.bar(np.arange(5)+(j-1.5)*.18,y,.17,color=color,label=label,zorder=3)
    ax.set_xticks(np.arange(5),order);ax.set_ylabel("期望图对数");ax.set_ylim(0,max(map(max,series))*1.30+.1)
    ax.set_title("在相同三次评分预算下，纠正与改错各有多少？",fontsize=17,pad=14)
    ax.legend(ncol=4,frameon=False,loc="upper center",fontsize=10);ax.grid(axis="y",alpha=.18);fig.savefig(out/"strategy_transitions.png",dpi=160);plt.close(fig)
    groups=[r for r in read_csv(ROOT/"margin_groups.csv") if r["group"]!="original_tie"];ties=read_csv(ROOT/"margin_groups.csv")[-1]
    nc=[int(r["n"]) for r in groups];x=np.arange(len(groups));names=[f"第{j+1}组\n{n} 对" for j,n in enumerate(nc)]
    fig,axs=plt.subplots(2,1,sharex=True,figsize=(9,7.4),layout="constrained")
    for ax in axs:ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    for off,fam,label,col in [(-.18,"T","平移","#1C6E73"),(.18,"C","对比度","#A98B56")]:
        axs[0].bar(x+off,[1000*float(r["median_abs_delta_"+fam]) for r in groups],.35,label=label,color=col)
    axs[0].set_title("相近原始领先幅度下，平移和对比度的影响",fontsize=17,pad=12)
    axs[0].set_ylabel("分差绝对位移中位数\n（HPS 分数 × 1,000）");axs[0].legend(frameon=False,ncol=2)
    for off,st,col in [(-.18,"BT","#326997"),(.18,"BC","#B16B45")]:
        b=axs[1].bar(x+off,[int(r[st+"_changed_pairs"]) for r in groups],.35,label=st,color=col)
        axs[1].bar_label(b,padding=2,fontsize=10)
    axs[1].margins(y=.25);axs[1].yaxis.set_major_locator(MaxNLocator(integer=True));axs[1].legend(frameon=False,ncol=2)
    axs[1].set_ylabel("至少一组搭配改判的图对数");axs[1].set_xticks(x,names)
    axs[1].set_xlabel("按原图领先幅度从小到大四等分。原图同分另列，共 "+ties["n"]+" 对。")
    fig.savefig(out/"margin_groups.png",dpi=160);plt.close(fig)
if __name__=="__main__":main()

