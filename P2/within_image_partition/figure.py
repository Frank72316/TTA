"""Show the two registered examples using PNGs exported from their scored tensors."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from design import HERE,SOURCE,rows

def draw(out):
    installed={f.name for f in font_manager.fontManager.ttflist}
    cjk=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","WenQuanYi Micro Hei","SimHei") if f in installed),None)
    if cjk:plt.rcParams["font.family"]=cjk
    def tr(zh,en):return zh if cjk else en
    data=rows(out/"cases.csv"); info={r["image_id"]:r for r in rows(SOURCE/"images.csv")}
    fig=plt.figure(figsize=(12,6.65),layout="constrained")
    gs=fig.add_gridspec(len(data)*3,5,height_ratios=[.15,.14,1]*len(data),hspace=.04,wspace=.02)
    for i,r in enumerate(data):
        fid=r["image_id"];im=info[fid];y=3*i
        change=float(r["delta_true"]);rel=tr("左","left") if r["sign"]=="1" else tr("右","right")
        relation=(f'{r["x"]} 在 {r["y"]} {rel}侧' if cjk else f'{r["x"]} {rel} of {r["y"]}')
        title=fig.add_subplot(gs[y,:]);title.axis("off")
        title.text(0,.5,f'{r["pair_id"]} {r["side"]}   ·   {relation}    |    '+tr("分开后真名词证据变化","Change in signed true evidence")+f'  {change:+.5f}',fontsize=13.8,weight="bold",va="center")
        x0,y0,w,h=json.loads(im["valid_xywh"])
        for j,line in enumerate((0,25,75)):
            hax=fig.add_subplot(gs[y+1,0] if not j else gs[y+1,2*j-1:2*j+1]);hax.axis("off")
            if j:
                split=line==int(r["split_line"]);val=float(r["split_true" if split else "same_true"])
                label=tr("分开","split") if split else tr("同侧","same side")
                margin_label=tr("有向证据","signed evidence")
                hax.text(.5,.5,f'{line}%   ·   {label}   ·   {margin_label} {val:+.5f}',va="center",ha="center",fontsize=12)
            else:hax.text(.5,.5,tr("原图与标注框","Original and boxes"),va="center",ha="center",fontsize=12)
        for j,(line,half) in enumerate(((0,"O"),(25,"left"),(25,"right"),(75,"left"),(75,"right"))):
            path=out/"smoke"/(f"{fid}_O.png" if not j else f"{fid}_{line}_{half}.png")
            ax=fig.add_subplot(gs[y+2,j])
            with Image.open(path) as img:ax.imshow(img,interpolation="nearest")
            ax.axis("off")
            if j:
                ax.text(.5,-.027,tr("保留左侧","Left visible") if half=="left" else tr("保留右侧","Right visible"),
                        transform=ax.transAxes,va="top",ha="center",fontsize=10)
            else:
                for k,col in (("x","#ffce36"),("y","#46dced")):
                    l,t,rr,b=json.loads(im[k+"_input_xyxy"])
                    ax.add_patch(Rectangle((l-.5,t-.5),rr-l,b-t,edgecolor=col,facecolor="none",lw=1.6))
                for xp in (.25,.75):ax.plot([x0+w*xp-.5]*2,[y0-.5,y0+h-.5],"--",color="white",lw=1)
    fig.savefig(out/"cases.png",dpi=145,facecolor="white");plt.close(fig)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=HERE);draw(ap.parse_args().out)
