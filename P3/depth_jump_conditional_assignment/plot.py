"""Render the three paired components and their total on a shared signed millimetre scale."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from common import *

def main():
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC","Noto Sans CJK JP","Microsoft YaHei","Droid Sans Fallback"):
        if name in fonts:
            plt.rcParams["font.family"]=[name,"DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"]=False
    rows={(r["batch"],r["method"],r["metric"]):r for r in read(OUT/"summary.csv")}
    fig,panels=plt.subplots(1,2,figsize=(10.5,4.0),sharex=True,sharey=True)
    col={"R":"#006f7a","T":"#a45918"};markers={"R":"o","T":"s"}
    fields=["axis_gain_mm","jump_bin_gain_mm","residual_mm","total_mm"]
    for ax,batch,title in zip(panels,["2014","2005-2006"],["2014 年 · 23 场 / 23 簇","2005/06 年 · 27 场 / 17 簇"]):
        ax.axvline(0,color="#606b75",lw=1.15);ax.axhline(.5,color="#afbac2",ls=":",lw=1)
        for i,field in enumerate(fields):
            for m,dy in (("R",.12),("T",-.12)):
                row=rows[batch,m,field];v,lo,hi=[float(row[k]) for k in ("mean","lo","hi")]
                y=3-i+dy
                ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=markers[m],color=col[m],ms=6,
                    mfc=col[m] if m=="R" else "white",ecolor=col[m],elinewidth=1.5,capsize=3,zorder=3)
        ax.set(title=title,ylim=(-.45,3.46),xlim=(-8,34),xticks=[-5,0,5,10,15,20,25,30],xlabel="落差 MAE 的减少量（mm）")
        ax.set_yticks([3,2,1,0],["扫描方向","进一步匹配预测落差","恢复原边界对应","总位置收益"])
        ax.grid(axis="x",alpha=.16)
        for edge in ("top","right","left"):ax.spines[edge].set_visible(False)
        ax.tick_params(axis="y",length=0)
    fig.legend([Line2D([0],[0],c=col[m],marker=markers[m],mfc=col[m] if m=="R" else "white") for m in METHODS],
               ["真实周边裁窗 R","原四窗 T"],ncol=2,loc="upper center",bbox_to_anchor=(.54,1.00),frameon=False)
    fig.subplots_adjust(left=.17,right=.98,top=.80,bottom=.18,wspace=.10)
    destination=ROOT/"figures";destination.mkdir(exist_ok=True)
    fig.savefig(destination/"decomposition.png",dpi=185,bbox_inches="tight",facecolor="white")
    plt.close(fig)
if __name__=="__main__":main()
