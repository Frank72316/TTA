"""Pixel-level decomposition using the original aligned depth cache, CPU only."""
import argparse, csv, hashlib, json, platform
from pathlib import Path
from io import BytesIO
from time import perf_counter
from zipfile import ZipFile
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"depth_adaptive_budget"
def sha(p):
    obj=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): obj.update(chunk)
    return obj.hexdigest()

def zones(truth,valid):
    defined=valid.copy()
    defined[[0,-1],:]=False
    defined[:,[0,-1]]=False
    jumps=np.zeros_like(valid)
    logs=np.log(np.where(valid,truth,1))
    for axis in (0,1):
        for shift in (-1,1):
            defined &= np.roll(valid,shift,axis)
            jumps |= abs(logs-np.roll(logs,shift,axis))>np.log(1.1)
    y,x=np.ogrid[-3:4,-3:4]
    boundary=binary_dilation(jumps&defined,structure=x*x+y*y<=9)&defined
    return dict(boundary=boundary,interior=defined&~boundary,undefined=valid&~defined)

def evaluate(args):
    started=perf_counter()
    env=json.loads((OLD/"environment.json").read_text())
    assert sha(args.zip)==env["archive_sha"]
    cache=args.cache/sha(OLD/"environment.json")
    rows=list(csv.DictReader((OLD/"frame_results.csv").open()))
    results=[]
    mosaic=np.arange(35,dtype=float).reshape(5,7)/8
    synthetic_four=mosaic+.75
    shifted=mosaic+np.mean(synthetic_four-mosaic)
    synthetic_truth=np.fliplr(mosaic)
    synthetic_space=float(np.mean((shifted-synthetic_truth)**2)-np.mean((synthetic_four-synthetic_truth)**2))
    assert np.array_equal(shifted,synthetic_four) and synthetic_space==0
    with ZipFile(args.zip) as z:
        for i,r in enumerate(rows):
            with Image.open(BytesIO(z.read(r["truth"]))) as im: gt=np.array(im,dtype="float32")/1000
            w=gt.shape[1]; s=(w*3+50)//100; gt=gt[:,s:w-s]
            valid=(gt>0)&(gt<=10)
            masks=zones(gt.astype(float),valid)
            pred=[]
            for v in "OFLR":
                with np.load(cache/(r["frame_id"]+"-"+v+".npz")) as f: pred.append(f["p"])
            assert all(p.shape==gt.shape for p in pred)
            o,f,l,right=[p[valid].astype(float) for p in pred]
            p2=(o+f)/2; p4=(o+f+l+right)/4
            shift=(p4-p2).mean()
            # Match depth.py's FP32 subtraction followed by FP64 reductions.
            o,f,l,right=[(p[valid]-gt[valid]).astype(float) for p in pred]
            e2=(o+f)/2; e4=(o+f+l+right)/4; es=e2+shift
            mse2=np.mean(e2**2); mse4=np.mean(e4**2); ms=np.mean(es**2)
            a=mse2-ms; b=ms-mse4
            d=dict(frame_id=r["frame_id"],scene=r["scene"],pixels=int(valid.sum()),shift_m=shift,
                   mse2=mse2,mse_shift=ms,mse4=mse4,G_mean=a,G_space=b,G_old=float(r["G"]),
                   telescope_error=a+b-float(r["G"]),mean_depth_check=abs((p2+shift).mean()-p4.mean()),
                   direct_space_error=abs(b-np.mean((p2+shift-gt[valid])**2-(p4-gt[valid])**2)))
            assert mse2==float(r["MSE2"]) and mse4==float(r["MSE4"]) and abs(d["telescope_error"])<1e-14
            for name,m in masks.items():
                subset=m[valid]; n=int(subset.sum()); fraction=n/len(e2)
                gain=np.mean(es[subset]**2-e4[subset]**2) if n else 0
                d.update({name+"_pixels":n,name+"_fraction":fraction,name+"_gain":gain if n else "",
                      name+"_mse2":np.mean(e2[subset]**2) if n else "",name+"_mse_shift":np.mean(es[subset]**2) if n else "",
                      name+"_weighted":gain*fraction})
            assert sum(d[k+"_pixels"] for k in masks)==len(e2)
            assert abs(sum(d[k+"_weighted"] for k in masks)-b)<1e-13
            results.append(d)
            if i==1: print("two frames passed alignment, metric replay and region checks",flush=True)
    with (HERE/"frame_results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0],lineterminator="\n");w.writeheader();w.writerows(results)
    (HERE/"environment.json").write_text(json.dumps(dict(source_environment=env,python=platform.python_version(),
        numpy=np.__version__,frame_results_sha=sha(OLD/"frame_results.csv"),protocol_sha=sha(HERE/"protocol.md"),
        native_cache_hits=4*len(results),forwards=0,wall_seconds=perf_counter()-started,
        smoke=[r["frame_id"] for r in results[:2]],constant_offset_space_gain=synthetic_space,script_sha=sha(Path(__file__))),indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k in ("zip","cache"): ap.add_argument("--"+k,type=Path,required=True)
    evaluate(ap.parse_args())
