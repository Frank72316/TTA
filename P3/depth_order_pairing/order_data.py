"""Read the five verified aligned arrays. No model packages or calls are needed."""
import argparse
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"depth_translation_pairing"
sys.path.insert(0,str(SOURCE))
from metrics import VIEWS,valid_truth,read,write,sha,save

RECIPES={"O":("O",),"LL":("O","L2","L4"),"RR":("O","R2","R4"),
         "LR":("O","L2","R4"),"RL":("O","R2","L4")}
CHANGES=(("LL","LR"),("RR","RL"))

def frozen():
    source=json.loads((HERE/"source.json").read_text())
    assert source["commit"]=="3e0ea7c340a5bf27faf784b6e16a45a27571a7f0"
    for k,h in source["sha256"].items():assert sha(SOURCE/k)==h,k
    return source

def stratum(mm_num,den):
    n=abs(mm_num)
    if not n:return "equal"
    if n<100*den:return "near"
    if n<300*den:return "small"
    if n<1000*den:return "medium"
    return "large"

def judgement(q,d):
    if d==0:return "undefined"
    if q==0:return "tie"
    return "correct" if (q>0)==(d>0) else "wrong"

def window_records(frame,millimetres,g,valid,p,smoke):
    windows=[];pixel_recipes={}
    height,width=g.shape
    assert (width,height)==(588,480)
    for row in range(4):
        for col in range(4):
            cx=((2*col+1)*width+4)//8;cy=((2*row+1)*height+4)//8
            x,y=cx-16,cy-16;cut=np.s_[y:y+32,x:x+32];mask=valid[cut];n=int(mask.sum())
            w=dict(frame_id=frame["frame_id"],scene=frame["scene"],window=4*row+col+1,x0=x,y0=y,x1=x+32,y1=y+32,
                   valid_pixels=n,eligible=int(10*n>=9*32*32),mm_sum=int(millimetres[cut][mask].sum(dtype=np.int64)))
            if n:
                w["truth_mean"]=w["mm_sum"]/(1000*n)
                assert abs(w["truth_mean"]-g[cut][mask].mean())<1e-12
                vals={v:p[v][cut][mask].astype(np.float64) for v in VIEWS}
                for v in VIEWS:w[v]=float(vals[v].mean())
                if smoke:
                    pixel_recipes[w["window"]]={k:float(np.mean([vals[v] for v in vs],axis=0).mean()) for k,vs in RECIPES.items()}
            else:
                w.update(dict.fromkeys(("truth_mean",*VIEWS),""))
            windows.append(w)
    assert [w["x0"] for w in windows[:4]]==[58,205,352,499]
    assert [w["y0"] for w in windows[::4]]==[44,164,284,404]
    return windows,pixel_recipes

def pair_record(a,b,pixel_recipes):
    den=a["valid_pixels"]*b["valid_pixels"]
    num=a["mm_sum"]*b["valid_pixels"]-b["mm_sum"]*a["valid_pixels"]
    d=num/(1000*den);x={v:a[v]-b[v] for v in VIEWS}
    assert abs(d-(a["truth_mean"]-b["truth_mean"]))<1e-12
    r=dict(frame_id=a["frame_id"],scene=a["scene"],A=a["window"],B=b["window"],stratum=stratum(num,den),d=d)
    r.update({"x_"+v:value for v,value in x.items()})
    for k,vs in RECIPES.items():
        q=sum(x[v] for v in vs)/len(vs)
        r["q_"+k]=q;r["sqerr_"+k]=(q-d)**2;r["state_"+k]=judgement(q,d)
        if pixel_recipes:
            independent=pixel_recipes[a["window"]][k]-pixel_recipes[b["window"]][k]
            assert abs(q-independent)<1e-12
    r["mse_same"]=(r["sqerr_LL"]+r["sqerr_RR"])/2;r["mse_opposite"]=(r["sqerr_LR"]+r["sqerr_RL"])/2
    r["G"]=r["mse_same"]-r["mse_opposite"]
    r["G_product"]=(x["L2"]-x["R2"])*(x["L4"]-x["R4"])/9
    r["identity_error"]=abs(r["G"]-r["G_product"])
    assert r["identity_error"]<1e-12
    return r

def extract(args):
    start=perf_counter();out=args.out;out.mkdir(parents=True,exist_ok=True)
    source=frozen()
    env=json.loads((SOURCE/"environment.json").read_text())
    assert sha(args.archive)==env["archive_sha256"]
    sig=dict(source=source["commit"],old_environment=sha(SOURCE/"environment.json"),protocol=sha(HERE/"protocol.md"),
             source_manifest=sha(HERE/"source.json"),script=sha(Path(__file__)),archive=env["archive_sha256"])
    frame_list=read(SOURCE/"frame_results.csv");assert len(frame_list)==len({r["frame_id"] for r in frame_list})==128
    audit={(r["frame_id"],r["view"]):r for r in read(SOURCE/"view_audit.csv")}
    assert len(audit)==640
    all_windows=[];all_pairs=[];verified=0
    if args.smoke:
        target=frame_list[:2]
    else:
        gate=json.loads((out/"smoke.json").read_text());assert gate["signature"]==sig
        for k in ("smoke_windows.csv","smoke_region_pairs.csv"):assert sha(out/k)==gate["hashes"][k]
        all_windows=read(out/"smoke_windows.csv");all_pairs=read(out/"smoke_region_pairs.csv")
        target=frame_list[2:]
    with ZipFile(args.archive) as z:
        for frame in target:
            fid=frame["frame_id"];scene=frame["scene"]
            with Image.open(BytesIO(z.read(frame["truth"]))) as gt:mm=np.asarray(gt)
            g,valid=valid_truth(mm);assert valid.sum()==int(frame["valid_pixels"])
            assert np.array_equal(g,mm[:,26:614].astype(np.float64)/1000)
            with np.load(args.predictions/(fid+".npz"),allow_pickle=False) as f:
                assert str(f["signature"])==sig["old_environment"]
                embedded=json.loads(str(f["log"]))
                assert {r["view"] for r in embedded}==set(VIEWS) and len(embedded)==5
                p={v:f[v] for v in VIEWS}
                for entry in embedded:
                    v=entry["view"];a=audit[fid,v];arr=p[v]
                    assert entry["frame_id"]==fid and arr.dtype==np.float32 and arr.shape==g.shape and np.isfinite(arr).all()
                    assert (entry["input_tensor_sha256"],entry["aligned_prediction_sha256"])==(a["input_tensor_sha256"],a["aligned_prediction_sha256"])
                    assert sha256(arr.tobytes()).hexdigest()==a["aligned_prediction_sha256"]
                    verified+=1
            windows,pixel_recipes=window_records(frame,mm[:,26:614],g,valid,p,args.smoke)
            pairs=[pair_record(a,b,pixel_recipes) for a,b in combinations(windows,2) if a["eligible"] and b["eligible"]]
            assert all(r["frame_id"]==fid and r["scene"]==scene and r["A"]<r["B"] for r in pairs)
            all_windows.extend(windows);all_pairs.extend(pairs)
    prefix="smoke_" if args.smoke else ""
    assert len(all_windows)==16*(2 if args.smoke else 128)
    assert len(all_pairs)==len({(r["frame_id"],int(r["A"]),int(r["B"])) for r in all_pairs})
    write(out/(prefix+"windows.csv"),all_windows);write(out/(prefix+"region_pairs.csv"),all_pairs)
    log=dict(signature=sig,hashes={prefix+n:sha(out/(prefix+n)) for n in ("windows.csv","region_pairs.csv")},
             frames_newly_read=len(target),source_arrays_hashed_this_run=verified,smoke_arrays_reused=0 if args.smoke else 10,
             window_rows=len(all_windows),pair_rows=len(all_pairs),max_identity_residual=max(float(r["identity_error"]) for r in all_pairs),
             smoke_pixel_first_recipe_checks=5*len(all_pairs) if args.smoke else 0,
             model_forwards=0,device="CPU",hostname=platform.node(),numpy=np.__version__,python=platform.python_version(),
             wall_seconds=perf_counter()-start)
    save(out/("smoke.json" if args.smoke else "run.json"),log)
    print({k:log[k] for k in ("frames_newly_read","source_arrays_hashed_this_run","pair_rows","max_identity_residual","wall_seconds")})

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--predictions",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE);ap.add_argument("--smoke",action="store_true");extract(ap.parse_args())
