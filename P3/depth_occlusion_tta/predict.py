"""FP32 from decoded RGB to native optical-axis depth, with a synthetic latency calibration."""
import argparse,os,platform,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import transformers
from io_data import *

def weight(w,h):
    x=np.minimum(np.arange(w)+.5,w-np.arange(w)-.5)/w
    y=np.minimum(np.arange(h)+.5,h-np.arange(h)-.5)/h
    return np.clip(np.minimum(y[:,None],x[None,:])/.1,.05,1).astype("f4")

def stitch(b,ws,bs,factors=None):
    total=np.zeros_like(b);den=np.zeros_like(b);ks=[]
    for i,((x,y,u,v),part) in enumerate(zip(bs,ws)):
        hh,ww=part.shape
        if factors is None:
            ys=np.flatnonzero((np.arange(hh)+.5>=.1*hh)&(np.arange(hh)+.5<=.9*hh))
            xs=np.flatnonzero((np.arange(ww)+.5>=.1*ww)&(np.arange(ww)+.5<=.9*ww))
            sel=np.s_[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]
            a=b[y:v,x:u][sel];p=part[sel];ok=(a>0)&(p>0)&np.isfinite(a)&np.isfinite(p)
            k=np.clip(np.median(a[ok]/p[ok]),.5,2) if ok.any() else 1.
        else:k=factors[i]
        wt=weight(ww,hh)
        total[y:v,x:u]+=wt*part*k;den[y:v,x:u]+=wt;ks.append(float(k))
    assert (den>0).all()
    return total/den,ks

def setup():
    base=get(HERE.parent/"depth_adaptive_budget/environment.json")
    assert all(sha(MODEL/f)==h for f,h in base["weights"].items())
    assert base["torch"]==torch.__version__ and base["transformers"]==transformers.__version__
    uuid=os.environ["CUDA_VISIBLE_DEVICES"];assert uuid.startswith("GPU-") and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    pp=transformers.AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    model=transformers.AutoModelForDepthEstimation.from_pretrained(MODEL,local_files_only=True).eval().cuda()
    assert model.config.max_depth==20 and model.config.depth_estimation_type=="metric"
    assert next(model.parameters()).dtype==torch.float32
    meta=dict(model=base["model"],revision=base["revision"],weights=base["weights"],torch=torch.__version__,
              transformers=transformers.__version__,numpy=np.__version__,pillow=Image.__version__,cuda=torch.version.cuda,
              uuid=uuid,hardware=torch.cuda.get_device_name(),driver=subprocess.check_output(["nvidia-smi","-i",uuid,
              "--query-gpu=driver_version","--format=csv,noheader"],text=True).strip(),host=platform.node())
    return pp,model,meta

def tensor(pp,images,factor=1):
    side=rounded(518*factor)
    return pp(images=images,size={"height":side,"width":side},return_tensors="pt").pixel_values
def raster(pp,image):
    return pp(images=image,return_tensors="np",do_normalize=False,do_rescale=False).pixel_values[0].astype("uint8")
def forward(pp,model,imgs,factor=1):
    x=tensor(pp,imgs,factor)
    with torch.inference_mode():
        p=model(pixel_values=x.cuda()).predicted_depth
        assert p.dtype==torch.float32 and tuple(p.shape[1:])==tuple(x.shape[-2:])
        h,w=imgs[0].height,imgs[0].width
        maps=F.interpolate(p[:,None],size=(h,w),mode="bicubic",align_corners=False)[:,0].cpu().numpy()
    return maps,tuple(x.shape[-2:])
def timed(fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();s=time.perf_counter()
    val=fn();torch.cuda.synchronize()
    return val,(time.perf_counter()-s)*1000,torch.cuda.max_memory_allocated()/2**20
def all_views(pp,model,img):
    (arr,hw),tb,mb=timed(lambda:forward(pp,model,[img]))
    b=arr[0];bs=boxes(*img.size)
    def extra():
        parts,wh=forward(pp,model,[img.crop(box) for box in bs])
        loc,ks=stitch(b,parts,bs)
        return parts,wh,(loc+b)*.5,ks
    (parts,wh,t,ks),extra_ms,extra_mem=timed(extra)
    return b,parts,hw,wh,ks,dict(B_ms=tb,T_ms=tb+extra_ms,B_peak_MiB=mb,T_peak_MiB=max(mb,extra_mem))
def calibration(pp,model,meta):
    dims=[Image.open(paths(r)[0]).size for r in read(HERE/"expected.csv") if r["dataset"]=="middlebury"]
    dims.sort(key=lambda z:z[0]/z[1])
    rows=[];chosen={}
    for ds,(w,h) in [("ibims",(1500,1000)),("middlebury",dims[len(dims)//2])]:
        im=Image.fromarray(np.random.default_rng(2026).integers(256,size=(h,w,3),dtype="u1"))
        bsz=tensor(pp,[im]).shape[-2:]
        seen=set();pick=0
        for method,factor in [("T",1)]+[("H",f) for f in SCALE]:
            h1,w1=tensor(pp,[im],factor).shape[-2:]
            row=dict(dataset=ds,method=method,factor=factor,source_w=w,source_h=h,net_h=h1,net_w=w1,
                     patches_h=h1//14,patches_w=w1//14,status="ok",warmup=0,repetitions=0,completed_image_forwards=0)
            if max(h1/h,w1/w)>1 or (method=="H" and (h1,w1) in seen):
                row["status"]="above_native" if max(h1/h,w1/w)>1 else "duplicate";rows.append(row);continue
            ts=[]
            try:
                for j in range(9):
                    if method=="T":
                        *_,tt=all_views(pp,model,im);ms=tt["T_ms"]
                    else:_,ms,_=timed(lambda:forward(pp,model,[im],factor))
                    row["completed_image_forwards"]+=5 if method=="T" else 1
                    if j>=2:ts.append(ms);row["repetitions"]+=1
                    else:row["warmup"]+=1
                row["median_ms"]=float(np.median(ts));row["p90_ms"]=float(np.percentile(ts,90));row["measured_ms"]=" ".join(map(str,ts))
                if method=="T":target=row["median_ms"]
                else:
                    seen.add((h1,w1))
                    if row["median_ms"]<=target and (h1,w1)!=tuple(bsz):pick=factor
            except torch.cuda.OutOfMemoryError:
                row["status"]="OOM";torch.cuda.empty_cache()
            rows.append(row)
        chosen[ds]=pick
        write(HERE/"calibration.csv",rows)
    save(HERE/"calibration.json",dict(selected=chosen,reference=meta,cache_code=sha(Path(__file__)),
          protocol=sha(HERE/"protocol.md"),freeze=sha(HERE/"freeze.json"),calibration_sha=sha(HERE/"calibration.csv")))
def execute(pp,model,meta,args):
    fr=get(HERE/"freeze.json");assert sha(HERE/"expected.csv")==fr["expected_sha256"] and sha(HERE/"protocol.md")==fr["protocol_sha256"]
    cal=get(HERE/"calibration.json");assert cal["cache_code"]==sha(Path(__file__)) and cal["freeze"]==sha(HERE/"freeze.json")
    env_sig=sha(HERE/"calibration.json")
    inputs=[r for r in read(HERE/"expected.csv") if paths(r)[0].exists() and paths(r)[1].exists() and paths(r)[2].exists()]
    rows=[];start=time.perf_counter();count=0
    for r in inputs[args.shard::args.shards]:
        im=rgb(r);w,h=im.size
        if r["dataset"]=="ibims":assert (w,h)==(1500,1000)
        d=work(r);d.mkdir(parents=True,exist_ok=True)
        p=d/"predictions.npz";audit=d/"inference.json"
        signature=dict(calibration=env_sig,rgb_sha256=sha(paths(r)[0]),common_impl=sha(HERE/"io_data.py"))
        if audit.exists():
            record=get(audit);assert record["signature"]==signature and record["prediction_sha256"]==sha(p)
            rows.append(dict(id=named(r),hits=record["forwards"],new=0));continue
        cache={};conditions={};ref=None
        for cond in ("N","L"):
            if cond=="N":image=im
            else:image=degrade(im,hw)
            grid=raster(pp,image)
            if cond=="N":ref=grid
            b,parts,hw,wh,ks,t=all_views(pp,model,image)
            delta=np.abs(grid.astype("i2")-ref.astype("i2"))
            assert grid.shape[1:]==hw
            cache[cond+"B"]=b;cache[cond+"W"]=parts
            conditions[cond]=dict(**t,base_hw=hw,window_hw=wh,k=ks,grid_sha=ah(grid),grid_mean_abs=float(delta.mean()),
                grid_max_abs=int(delta.max()),window_scale_x=wh[1]/parts.shape[2]/(hw[1]/w),
                window_scale_y=wh[0]/parts.shape[1]/(hw[0]/h))
        requested=cal["selected"][r["dataset"]]
        actual=0
        for f in reversed([x for x in SCALE if 1<x<=requested]):
            hh,ww=tensor(pp,[im],f).shape[-2:]
            if hh<=h and ww<=w:
                try:
                    (hp,shape),ms,mem=timed(lambda:forward(pp,model,[im],f))
                    cache["H"]=hp[0];conditions["H"]=dict(ms=ms,peak_MiB=mem,hw=shape,factor=f);actual=f;break
                except torch.cuda.OutOfMemoryError:
                    conditions.setdefault("H_oom",[]).append(f);torch.cuda.empty_cache()
        for v in cache.values():assert np.isfinite(v).all()
        tmp=p.with_suffix(".tmp")
        with tmp.open("wb") as file:np.savez(file,**cache)
        tmp.replace(p)
        s=dict(signature=signature,meta=meta,conditions=conditions,boxes=boxes(w,h),size=[w,h],
               B_prediction_MAD_m=float(np.abs(cache["NB"]-cache["LB"]).mean(dtype=np.float64)),
               H_actual_factor=actual,forwards=10+int(actual>0),prediction_sha256=sha(p))
        save(audit,s);count+=s["forwards"];rows.append(dict(id=named(r),new=s["forwards"],hits=0))
    save(HERE/f"gpurun_{args.shard}.json",dict(meta=meta,shards=args.shards,rows=rows,fresh=count,wall_s=time.perf_counter()-start))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["calibrate","score"])
    ap.add_argument("--shard",type=int,default=0);ap.add_argument("--shards",type=int,default=1)
    args=ap.parse_args();pp,model,meta=setup()
    if args.mode=="calibrate":calibration(pp,model,meta)
    else:execute(pp,model,meta,args)
