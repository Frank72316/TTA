"""Frozen examples, selected by scene-level change; common edge and orientation within each case."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data import *
from analysis import maps
from coverage import edge_path
from views import canvas,ref
from transformers import AutoImageProcessor

def main():
    pp=AutoImageProcessor.from_pretrained(old.MODEL,local_files_only=True)
    paired=read(ROOT/"paired_differences.csv")
    folder=ROOT/"figures";folder.mkdir(exist_ok=True)
    selected=[]
    fig,axs=plt.subplots(2,4,figsize=(15,7),gridspec_kw={"width_ratios":[1,1,1,1.48]},layout="constrained")
    for row,(label,condition) in enumerate([("improved",lambda g:g>0),("not improved",lambda g:g<=0)]):
        candidates=[r for r in paired if condition(float(r["T_minus_R_all_near_mae_mm"]))]
        candidates.sort(key=lambda r:r["year"]=="2014")
        if not candidates:
            for ax in axs[row]:ax.axis("off")
            continue
        result=candidates[0]
        r=next(r for r in included() if key(r)==key(result))
        z=metric(r);im=Image.open(sources(r)[0]).convert("RGB")
        with np.load(edge_path(r)) as f:e=dict(f)
        yy,xx=np.divmod(e["anchor"],z.shape[1]);i=np.argmin((xx-z.shape[1]*.5)**2+(yy-z.shape[0]*.5)**2)
        y,x=int(yy[i]),int(xx[i]);vertical=bool(e["axis"][i])
        j=get(cache(r)/"inference.json");ls=j["layouts"]
        choices=[k for k,s in enumerate(ls) if s["box"][0]<=x<s["box"][2] and s["box"][1]<=y<s["box"][3]]
        q=min(choices,key=lambda k:(x-(ls[k]["box"][0]+ls[k]["box"][2])/2)**2+(y-(ls[k]["box"][1]+ls[k]["box"][3])/2)**2)
        s=ls[q];core=ref.raster(pp,im.crop(s["box"])).transpose(1,2,0)
        ax=axs[row,0];ax.imshow(im);ax.plot(x,y,marker="+",ms=15,mew=2.5,color="red")
        a,b,u,v=s["box"];ax.add_patch(Rectangle((a,b),u-a,v-b,ec="deepskyblue",lw=1.4,fill=False))
        gain=float(result["T_minus_R_all_near_mae_mm"])
        ax.set_title(f'{key(r)} | {label}\nNear-side MAE gain: {gain:+.1f} mm',fontsize=11);ax.axis("off")
        for col,m in [(1,"R"),(2,"P")]:
            ax=axs[row,col];actual=canvas(im,core,s,m,pp.resample);ax.imshow(actual)
            a,b,u,v=s["net_box"];ax.add_patch(Rectangle((a,b),u-a-1,v-b-1,ec="deepskyblue",lw=2,fill=False))
            ax.set_title("R: real surround" if m=="R" else "P: reflected surround",fontsize=11);ax.axis("off")
        ax=axs[row,3];steps=np.arange(-32,33)
        sy=y+steps if vertical else np.full_like(steps,y)
        sx=x+steps if not vertical else np.full_like(steps,x)
        keep=(sy>=0)&(sx>=0)&(sy<z.shape[0])&(sx<z.shape[1])
        sy,sx,steps=sy[keep],sx[keep],steps[keep]
        line=z[sy,sx].copy();line[(line<=0)|(line>20)]=np.nan
        ax.plot(steps,line,lw=2,c="black",label="GT")
        for m,d in maps(r,("B","T","R")):
            ax.plot(steps,d[sy,sx],lw=1.6,color={"B":"#666666","T":"#d67808","R":"#067a78"}[m],label=m)
        ax.axvline(.5,c="#666666",lw=.8,ls=":")
        ax.set_xlabel(("Vertical" if vertical else "Horizontal")+" offset (source pixels)");ax.set_ylabel("Optical-axis depth (m)")
        ax.set_title("Profile at marked GT boundary",fontsize=11);ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
        selected.append(dict(year=r["year"],id=r["id"],category=label,near_gain_mm=gain,x=x,y=y,axis="vertical" if vertical else "horizontal",
                             corner=q,near_direction=int(e["near_direction"][i]),gt_jump_m=float(e["jump"][i])))
    fig.savefig(folder/"examples.jpg",dpi=170)
    write(ROOT/"cases.csv",selected)
if __name__=="__main__":main()
