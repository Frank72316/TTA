"""Independent CPU replay from per-frame FP32 caches and unsigned millimetre truth."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from metrics import HERE,OLD,VIEWS,read,save,sha
import hashlib

def verify(args):
    out=args.out
    env=json.loads((out/"environment.json").read_text())
    assert sha(args.archive)==env["archive_sha256"] and sha(OLD/"environment.json")==env["old_environment_sha256"]
    prev=args.old_cache/env["old_environment_sha256"]
    audit={(r["frame_id"],r["view"]):r for r in read(out/"view_audit.csv")}
    results=read(out/"frame_results.csv");worst=worst_product=0.
    with ZipFile(args.archive) as z:
        for r in results:
            fid=r["frame_id"]
            with Image.open(BytesIO(z.read(r["truth"]))) as gt:
                mm=np.array(gt)
            w=mm.shape[1];border=int(w*.04+.5);oldborder=int(w*.03+.5)
            truth=mm[:,border:w-border].astype(np.float64)*.001;mask=(truth>0)&(truth<=10)
            assert mask.sum()==int(r["valid_pixels"])
            with np.load(out/"predictions"/(fid+".npz"),allow_pickle=False) as f:
                assert str(f["signature"])==sha(out/"environment.json")
                d={v:f[v] for v in VIEWS}
            oldfile=prev/(fid+"-O.npz")
            assert sha(oldfile)==audit[fid,"O"]["source_old_npz_sha256"]
            with np.load(oldfile,allow_pickle=False) as f:old=f["p"]
            trim=border-oldborder;assert np.array_equal(d["O"],old[:,trim:old.shape[1]-trim])
            for v in VIEWS:
                assert hashlib.sha256(d[v].tobytes()).hexdigest()==audit[fid,v]["aligned_prediction_sha256"]
            residual={v:d[v][mask].astype(np.float64)-truth[mask] for v in VIEWS}
            for k,views in (("O",("O",)),("LL",("O","L2","L4")),("RR",("O","R2","R4")),
                            ("LR",("O","L2","R4")),("RL",("O","R2","L4"))):
                estimate=np.mean(np.square(np.mean([residual[v] for v in views],axis=0)))
                worst=max(worst,abs(estimate-float(r["MSE_"+k])))
            c=(residual["L2"]-residual["R2"])*(residual["L4"]-residual["R4"])/9
            for k,val in [("G",c.mean()),("positive_contribution",c[c>0].sum()/c.size),
                          ("negative_contribution",c[c<0].sum()/c.size)]:
                worst_product=max(worst_product,abs(val-float(r[k])))
    assert len(audit)==640 and len(results)==128 and max(worst,worst_product)<1e-12
    obj=dict(frames=len(results),old_original_arrays_checked=len(results),aligned_arrays_hashed=len(audit),
             mse_values_replayed=5*len(results),max_abs_mse_difference=float(worst),
             max_abs_gain_or_contribution_difference=float(worst_product))
    save(out/"pixel_replay.json",obj);print(obj)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE)
    for k in ("archive","old-cache"):p.add_argument("--"+k,type=Path,required=True)
    verify(p.parse_args())
