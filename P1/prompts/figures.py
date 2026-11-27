import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageOps
from common import ROOT, PAIRS, read_csv, images, write_csv

def case_rows(rows):
    ordered=sorted(rows,key=lambda r:(-float(r["abs_I"]),r["pair_id"]))
    top=ordered[:3]
    controls=random.Random(42).sample(sorted(ordered[3:],key=lambda r:r["pair_id"]),3)
    return [("最大交互",r) for r in top]+[("随机对照",r) for r in controls]


def main():
    dest=ROOT/"figures"
    dest.mkdir(exist_ok=True)
    new=read_csv(ROOT/"pair_results.csv")
    old=read_csv(ROOT/"old_audit.csv")
    plt.rcParams.update({"font.family":["Noto Sans CJK JP","DejaVu Sans"],"axes.spines.top":False,
                         "axes.spines.right":False,"axes.unicode_minus":False,"font.size":11})
    fig,ax=plt.subplots(figsize=(6.4,6.0),layout="constrained")
    xs,ys=([abs(float(r[c+"_delta"])) for r in new] for c in ("Q0","Q1"))
    hi=max(xs+ys)*1.08
    ax.plot([0,hi],[0,hi],color="#31353a",linestyle="--",lw=1,label="响应幅度相等")
    ax.scatter(xs,ys,s=36,facecolor="#087b82",alpha=.58,edgecolor="white",linewidth=.45,zorder=3)
    ax.set(xlim=(-hi*.025,hi),ylim=(-hi*.025,hi),aspect="equal",xlabel="Q0 亮度响应幅度",ylabel="Q1 亮度响应幅度",
           title="同一图对在两种提示词下的亮度响应")
    ax.legend(loc="lower right",frameon=False)
    ax.grid(alpha=.15)
    fig.savefig(dest/"response.png",dpi=185)
    plt.close(fig)
    panels=[("P0",old,"旧原提示词"),("Q0",new,"新基础提示词"),("Q1",new,"新夜景提示词")]
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.1),sharex=True,sharey=True,layout="constrained")
    xmax=max(abs(float(r[c+"_d0"])) for c,rows,_ in panels for r in rows)*1.06
    ymax=max(abs(float(r[c+"_delta"])) for c,rows,_ in panels for r in rows)*1.12
    for ax,(c,rows,title) in zip(axes,panels):
        ax.axhline(0,lw=.7,color="#a0a8aa")
        ax.axvline(0,lw=.7,color="#a0a8aa")
        axis=np.array([-xmax,xmax])
        ax.plot(axis,-axis,color="#b46d23",lw=1.4,label="只调亮平局")
        ax.plot(axis,-2*axis,color="#714580",lw=1.4,ls="--",label="均值平局")
        ax.scatter([float(r[c+"_d0"]) for r in rows],[float(r[c+"_delta"]) for r in rows],color="#087b82",s=17,alpha=.65)
        ax.set(xlim=(-xmax,xmax),ylim=(-ymax,ymax),title=c+" · "+title,xlabel="原图分差 d₀")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if c=="P0":
            for i,offset in [("005",(-43,20)),("053",(18,-22))]:
                r=next(r for r in rows if r["pair_id"]==i)
                pt=float(r["P0_d0"]),float(r["P0_delta"])
                ax.scatter(*pt,color="#8c3032",s=24,zorder=4)
                ax.annotate(i,pt,xytext=offset,textcoords="offset points",fontsize=9,
                            bbox=dict(boxstyle="round,pad=.15",fc="white",ec="none",alpha=.9),
                            arrowprops=dict(arrowstyle="-",color="#8c3032"))
    axes[0].set_ylabel("差异漂移 Δ")
    axes[-1].legend(frameon=False,loc="upper right",fontsize=9)
    fig.savefig(dest/"flip_boundaries.png",dpi=185)
    plt.close(fig)
    selected=case_rows(new)
    write_csv(ROOT/"cases.csv",[dict(selection=group,pair_id=r["pair_id"],abs_I=r["abs_I"]) for group,r in selected])
    for _,r in selected:
        i=r["pair_id"]
        canvas=Image.new("RGB",(608,608),"#e1e4e5")
        for k,pic in enumerate(images(i)):
            x,y=k//2,k%2
            thumb=ImageOps.contain(pic,(300,300))
            canvas.paste(thumb,(x*308+(300-thumb.width)//2,y*308+(300-thumb.height)//2))
        canvas.save(dest/f"case_{i}.png")

if __name__=="__main__":
    main()

