"""Plot the scene-cluster estimates and their strategy tie at zero."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from study import HERE, read

def main():
    paths=[Path("C:/Windows/Fonts/msyh.ttc"),Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for p in paths:
        if p.exists():
            fm.fontManager.addfont(str(p))
            plt.rcParams["font.family"]=fm.FontProperties(fname=str(p)).get_name()
            break
    rows={(r["arm"],r["metric"]):r for r in read(HERE/"summary.csv")}
    fig,ax=plt.subplots(figsize=(9.1,4.7),layout="constrained")
    ax.axvspan(-.55,1.5,color="#eaf6f4")
    ax.axvspan(1.5,3.55,color="#f2eef9")
    ax.axhline(0,color="#303841",lw=1)
    for arm,offset,color,marker,label in [("reflect",-.115,"#006d70","o","反射填边"),("real",.115,"#964509","s","真实边条")]:
        for k,name in enumerate(["G_11_25","G_17_31","G_11_31","G_17_25"]):
            r=rows[arm,name]
            m,lo,hi=[1000*float(r[f]) for f in ("estimate","low95","high95")]
            ax.errorbar(k+offset,m,yerr=[[m-lo],[hi-m]],fmt=marker,color=color,capsize=4,lw=1.6,ms=7,
                        label=label if k==0 else None)
    ax.set_xlim(-.55,3.55)
    ax.set_ylim(-2.0,3.75)
    ax.set_xticks(range(4),["11 ＋ 25","17 ＋ 31","11 ＋ 31","17 ＋ 25"])
    ax.set_xlabel("平移幅度组合（模型输入像素）")
    ax.set_ylabel(r"同向 MSE 减反向 MSE（$10^{-3}$ m²）")
    ax.text(.5,3.24,"预言反向更好（正值）",ha="center",color="#085558")
    ax.text(2.5,3.24,"预言同向更好（负值）",ha="center",color="#573768")
    ax.set_title("两种填边下，四组点估计均与事先预测同号",loc="left",weight="bold",pad=12)
    ax.legend(loc="lower left",ncol=2,frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.16)
    fig.savefig(HERE/"pairing_results.png",dpi=170)
    plt.close(fig)

if __name__=="__main__":
    main()
