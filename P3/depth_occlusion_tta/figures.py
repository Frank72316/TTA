"""Frozen registration audit (RGB/GT only) and three separately selected outcome cases."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from evaluate import fuse
from io_data import *

def valid(z):return np.where((z>0)&(z<=20),z,np.nan)
def choose_edge(e,shape):
    h,w=shape;idx=np.flatnonzero(e["width"]==1)
    if not len(idx):idx=np.arange(len(e["anchor"]))
    y,x=np.unravel_index(e["anchor"][idx],shape)
    return idx[np.argmin(((y-h/2)/h)**2+((x-w/2)/w)**2)]
def overlay(ax,img,e):
    w,h=img.size
    ax.imshow(img.resize((min(w,650),rounded(h*min(w,650)/w))),extent=(-.5,w-.5,h-.5,-.5))
    for group,color in [(0,"#ffbc57"),(2,"#14baf4"),(1,"#fa2777")]:
        inds=e["anchor"][e["width"]==group]
        yy,xx=np.divmod(inds,w)
        ax.scatter(xx,yy,s=1,linewidths=0,color=color,alpha=.8)
    ax.set(xlim=(-.5,w-.5),ylim=(h-.5,-.5))
def plot_audit(rows,folder):
    fig,axes=plt.subplots(len(rows),3,figsize=(10,2.25*len(rows)),squeeze=False,layout="constrained")
    for ax,r in zip(axes,rows):
        im=rgb(r);z=valid(depth(r));e=dict(np.load(work(r)/"edges.npz"))
        ax[0].imshow(im.resize((540,rounded(540*im.height/im.width))));ax[0].set_title(r["id"]+"  |  RGB",fontsize=10)
        ax[1].imshow(z[::5,::5],vmin=np.nanpercentile(z,2),vmax=np.nanpercentile(z,98),cmap="viridis")
        ax[1].set_title("Registered optical-axis depth",fontsize=10);overlay(ax[2],im,e)
        ax[2].set_title("Narrow: pink   wide: blue   other: gold",fontsize=9)
        for a in ax:a.axis("off")
    fig.savefig(folder/(rows[0]["dataset"]+"_registration.jpg"),dpi=125);plt.close(fig)
def plot_case(r,case,folder):
    im=rgb(r);z=valid(depth(r));e=dict(np.load(work(r)/"edges.npz"));d=work(r)
    info=get(d/"inference.json");bs=info["boxes"];he,wi=z.shape
    with np.load(d/"predictions.npz") as ck:
        b=ck["NB"];parts=ck["NW"];h=ck["H"] if "H" in ck else None
    t=(b+fuse(b,parts,bs,info["conditions"]["N"]["k"]))/2
    raw=(b+fuse(b,parts,bs,[1]*4))/2
    j=choose_edge(e,z.shape);y,x=divmod(int(e["anchor"][j]),wi);v=int(e["axis"][j])
    fig,axs=plt.subplots(4,4,figsize=(13.2,10.3),layout="constrained");ax=axs.ravel()
    thumb=im.resize((610,rounded(610*he/wi)))
    for a in ax[:2]:a.imshow(thumb,extent=(0,wi,he,0))
    for box,col in zip(bs,["#00aadb","#ffb900","#e23473","#99dc20"]):
        l,u,rr,dd=box;ax[1].add_patch(Rectangle((l,u),rr-l,dd-u,fill=False,edgecolor=col,lw=1.5))
    overlay(ax[2],im,e)
    ax[2].plot(x,y,marker="o",ms=9,mfc="none",mec="white")
    maps=[p for p in (z,b,t,h) if p is not None]
    lo,hi=np.nanpercentile(np.concatenate([p[::5,::5].ravel() for p in maps]),[2,98]);names=["RGB","Four 60% windows","Width labels + selected interface","GT (m)","B (m)","T (m)","H (m)","Local interface: near / far","B absolute error","T absolute error","H absolute error"]
    for idx,val in [(3,z),(4,b),(5,t),(6,h)]:
        if val is not None:vmap=ax[idx].imshow(val[::4,::4],vmin=lo,vmax=hi,cmap="viridis")
    fig.colorbar(vmap,ax=ax[3:7].tolist(),shrink=.55,label="depth (m)",location="right")
    rad=55;left=max(0,x-rad);top=max(0,y-rad);right=min(wi,x+rad+1);bottom=min(he,y+rad+1)
    ax[7].imshow(im.crop((left,top,right,bottom)),extent=(left-.5,right-.5,bottom-.5,top-.5))
    for side,c in [("near","#ff1170"),("far","#06d5f2")]:
        yy,xx=np.divmod(e[side][j],wi);ax[7].scatter(xx,yy,s=14,color=c)
    vmax=np.nanpercentile(np.concatenate([np.abs(p[::5,::5]-z[::5,::5]).ravel() for p in maps[1:]]),98)
    for aa,val in zip(ax[8:11],(b,t,h)):
        if val is not None:err=aa.imshow(np.abs(val[::4,::4]-z[::4,::4]),vmin=0,vmax=vmax,cmap="inferno")
    fig.colorbar(err,ax=ax[8:11].tolist(),shrink=.55,label="|error| (m)",location="right")
    pos=np.arange(max(0,(y if v else x)-rad),min(he if v else wi,(y if v else x)+rad+1))
    ind=(pos,x) if v else (y,pos)
    for val,name,c in [(z,"GT","black"),(b,"B","#2067a0"),(raw,"raw","#ac6f20"),(t,"T","#15836b")]:
        ax[11].plot(pos-(y if v else x)-.5,val[ind],label=name,color=c,lw=1.3)
    ax[11].axvline(0,ls=":",lw=1,color="gray");ax[11].legend(fontsize=7,ncol=2);ax[11].set(xlabel="native px from interface",ylabel="depth (m)",title=("Vertical" if v else "Horizontal")+" profile")
    for a,n in zip(ax,names):a.set_title(n,fontsize=10);a.axis("off")
    for a,p,n in zip(ax[12:],(z,b,t,h),("GT","B","T","H")):
        if p is not None:a.imshow(p[top:bottom,left:right],vmin=lo,vmax=hi,cmap="viridis")
        a.set_title("Same interface: "+n,fontsize=10);a.axis("off")
    fig.suptitle(f"{r['id']}   |   {case['case']}   |   boundary MAE gain {float(case['G_mm']):+.1f} mm   |   nominal zoom 1.67x")
    fig.savefig(folder/(r["dataset"]+"_"+case["case"]+".jpg"),dpi=130);plt.close(fig)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["audit","cases"]);mode=ap.parse_args().mode
    folder=HERE/"figures";folder.mkdir(exist_ok=True)
    rs=[r for r in read(HERE/"expected.csv") if (work(r)/"edges.npz").exists()]
    if mode=="audit":
        for ds in dict.fromkeys(r["dataset"] for r in rs):plot_audit([r for r in rs if r["dataset"]==ds][:6],folder)
    else:
        for c in read(HERE/"cases.csv"):plot_case(dict(dataset=c["dataset"],id=c["id"]),c,folder)
