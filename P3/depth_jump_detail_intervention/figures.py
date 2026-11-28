"""Paired intervals and pre-specified case profiles, generated without additional model calls."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from shared import *

FIG=HERE/"figures"
COLORS={"GT":"#17191b","B":"#808891","H":"#007b81","L":"#bd5d20"}

def setup():
    FIG.mkdir(exist_ok=True)
    options={f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC","Noto Sans CJK JP","Microsoft YaHei","Droid Sans Fallback"):
        if name in options:
            plt.rcParams["font.family"]=[name,"DejaVu Sans"];break
    plt.rcParams["axes.unicode_minus"]=False

def intervals_plot():
    rows={(r["batch"],r["metric"],r["scale"]):r for r in read(RESULTS/"summary.csv")}
    fig,axes=plt.subplots(1,2,figsize=(10,3.0))
    for ax,k,title in zip(axes,("delta_P_mm","G_mm"),("位置匹配收益差 ΔP","真实落差精度差 G")):
        ax.axvline(0,color="#606a74",lw=1)
        for y,batch in enumerate(("2005-2006","2014")):
            for scale,dy,marker,color in (("native_R",.10,"o","#007b81"),("low_core",-.10,"s","#75607c")):
                row=rows[batch,k,scale];v,l,h=[float(row[f]) for f in ("mean","lo","hi")]
                ax.errorbar(v,y+dy,xerr=[[v-l],[h-v]],fmt=marker,color=color,mfc=color if scale=="native_R" else "white",
                            capsize=3,elinewidth=1.7,ms=6,zorder=3)
        ax.set(yticks=[0,1],yticklabels=["2005/06","2014"],ylim=(-.45,1.45),title=title,xlabel="毫米（正值有利于原生核心）")
        ax.grid(axis="x",alpha=.20)
        for s in ("top","right","left"):ax.spines[s].set_visible(False)
        ax.tick_params(axis="y",length=0)
    fig.legend(handles=[Line2D([],[],color="#007b81",marker="o",label="主分析 · 共用 H 尺度"),
        Line2D([],[],color="#75607c",marker="s",mfc="white",label="敏感性 · 共用 L 尺度")],ncol=2,loc="upper center",frameon=False)
    fig.subplots_adjust(top=.72,bottom=.21,wspace=.35,left=.09,right=.98)
    fig.savefig(FIG/"paired_intervals.png",dpi=190,bbox_inches="tight");plt.close(fig)

def renderer():
    from importlib.util import module_from_spec,spec_from_file_location
    spec=spec_from_file_location("detail_intervention_score",HERE/"score.py")
    scoring=module_from_spec(spec);spec.loader.exec_module(scoring)
    from transformers import AutoImageProcessor
    # Match ref.setup(): CPU interpolation can round a few RGB values differently with one thread.
    scoring.torch.set_num_threads(2)
    pp=AutoImageProcessor.from_pretrained(old.MODEL,local_files_only=True)
    return pp,scoring

def audit_inputs():
    from PIL import Image
    pp,scoring=renderer()
    count=0
    for rec in included():
        source=Image.open(sources(rec)[0]).convert("RGB")
        _,rows,_,_=scoring.build_inputs(pp,source,get(cache(rec)/"inference.json"))
        saved=get(local_cache(rec)/"score.json")["window_rows"]
        assert rows==saved
        count+=len(rows)
    assert count==200
    save(RESULTS/"input_replay.json",dict(windows=count,matched_native_tensors=count,matched_low_tensors=count,
         cpu_threads=2,additional_visual_forwards=0))

def case_plots():
    from data import metric
    from analyze import load_low
    from edge_data import predict_map
    from coverage import edge_path
    from PIL import Image
    pp,scoring=renderer()
    rows=read(RESULTS/"per_scene.csv")
    samples={(r["year"],r["id"]):r for r in included()}
    cases=[];profiles=[]
    for batch,years in COHORTS.items():
        rr=[r for r in rows if r["scale"]=="native_R" and r["year"] in years]
        fig,axes=plt.subplots(2,5,figsize=(16,6.1),gridspec_kw={"width_ratios":[1,1,.82,.82,1.87]},layout="constrained")
        for row,(rank,selected) in enumerate((("最大",max(rr,key=lambda r:float(r["delta_P_mm"]))),("最小",min(rr,key=lambda r:float(r["delta_P_mm"]))))):
            rec=samples[selected["year"],selected["id"]];b,hi=signed_base_and_h(rec)
            low,mark=load_low(rec);j=get(cache(rec)/"inference.json");layouts=j["layouts"];boxes=[v["box"] for v in layouts]
            gt=metric(rec)
            frames={"GT":gt,"B":b,"H":predict_map(b,hi,boxes,j["k"]["R"]),"L":predict_map(b,low,boxes,j["k"]["R"])}
            with np.load(edge_path(rec)) as f:
                anchors=f["anchor"];axis=f["axis"];yy,xx=np.divmod(anchors,b.shape[1])
                xxmid=xx+.5*(axis==0);yymid=yy+.5*(axis==1)
                idx=np.argmin((xxmid-(b.shape[1]-1)/2)**2+(yymid-(b.shape[0]-1)/2)**2)
                x,y=int(xx[idx]),int(yy[idx]);xm,ym=xxmid[idx],yymid[idx];vertical=bool(axis[idx])
                near,far=f["near"][idx],f["far"][idx]
            options=[i for i,(a,c,d,e) in enumerate(boxes) if a<=xm<d and c<=ym<e]
            i=min(options,key=lambda t:(xm-(boxes[t][0]+boxes[t][2]-1)/2)**2+(ym-(boxes[t][1]+boxes[t][3]-1)/2)**2)
            source=Image.open(sources(rec)[0]).convert("RGB");_,hashes,high_inputs,low_inputs=scoring.build_inputs(pp,source,j)
            assert hashes[i]["low_sha256"]==mark["window_rows"][i]["low_sha256"]
            a,c,d,e=boxes[i];nx,ny,nu,nv=layouts[i]["net_box"]
            cx=(xm+.5-a)*(nu-nx)/(d-a)-.5;cy=(ym+.5-c)*(nv-ny)/(e-c)-.5
            size=80
            left=int(np.clip(round(cx-size/2),0,nu-nx-size));top=int(np.clip(round(cy-size/2),0,nv-ny-size))
            for col,(array,label) in enumerate(((high_inputs[i],"H 原生核心"),(low_inputs[i],"L 低通核心"))):
                core=array[ny:nv,nx:nu]
                ax=axes[row,col];ax.imshow(core);ax.add_patch(Rectangle((left-.5,top-.5),size,size,ec="#e94b32",fill=False,lw=1.4))
                ax.set_title(label,fontsize=11);ax.axis("off")
                ax=axes[row,col+2];ax.imshow(core[top:top+size,left:left+size],interpolation="nearest")
                ax.plot(cx-left,cy-top,"+",c="#e94b32",ms=10,mew=1.2)
                ax.set_title(("H" if col==0 else "L")+" · 同一处放大",fontsize=11);ax.axis("off")
            steps=np.arange(-32,33)
            sy=y+steps if vertical else np.full_like(steps,y)
            sx=x+steps if not vertical else np.full_like(steps,x)
            keep=(sy>=0)&(sx>=0)&(sy<b.shape[0])&(sx<b.shape[1]);sy,sx,steps=sy[keep],sx[keep],steps[keep]
            ax=axes[row,4];ax.axvline(.5,color="#848c94",lw=.8,ls=":")
            traces={}
            for m,data in frames.items():
                line=data[sy,sx].copy()
                if m=="GT":line[(line<=0)|(line>20)]=np.nan
                ax.plot(steps,line,label=m,color=COLORS[m],lw=2 if m=="GT" else 1.55);traces[m]=line
            ax.set_title(f'{rec["id"]} · ΔP {rank}（{float(selected["delta_P_mm"]):+.1f} mm）',fontsize=11)
            ax.set_xlabel(("纵向" if vertical else "横向")+"偏移（源像素）",fontsize=9);ax.set_ylabel("光轴深度（m）",fontsize=9)
            ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=4,loc="best",columnspacing=.7,handlelength=1.3)
            cases.append(dict(**ident(rec),rank=rank,delta_P_mm=selected["delta_P_mm"],G_mm=selected["G_mm"],edge_index=idx,
                anchor_x=x,anchor_y=y,axis=int(vertical),corner=i,zoom_x=left,zoom_y=top,zoom_size=size,
                **{m+"_edge_jump_mm":(float(np.median(arr.ravel()[far]))-float(np.median(arr.ravel()[near])))*1000 for m,arr in frames.items()}))
            for k,t in enumerate(steps):
                profiles.append(dict(**ident(rec),offset=t,x=sx[k],y=sy[k],**{m+"_m":line[k] if np.isfinite(line[k]) else "" for m,line in traces.items()}))
        fig.savefig(FIG/("cases_"+batch+".jpg"),dpi=170,bbox_inches="tight");plt.close(fig)
    write(RESULTS/"cases.csv",cases);write(RESULTS/"case_profiles.csv",profiles)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("which",choices=("intervals","cases","both","audit"),default="both",nargs="?")
    arg=p.parse_args().which;setup()
    if arg in ("intervals","both"):intervals_plot()
    if arg in ("cases","both"):case_plots()
    if arg=="audit":audit_inputs()
