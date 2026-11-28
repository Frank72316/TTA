"""Recover region means on the original RGB-D pixel grid and audit all cached maps (CPU)."""
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from study import HERE, TRIM, SHIFTS, ARMS, PAIRS, key,read,write,load,save,sha

def main(args):
    out=args.out
    frames=read(HERE/"frames.csv")
    if args.smoke:
        frames=frames[:2]
    else:
        assert load(out/"smoke_regions.json")["pixel_first_checks"]>0
    env=load(out/"environment.json")
    assert sha(args.archive)==env["archive_sha"]
    signature=sha(out/"environment.json")
    windows=[]
    hashes=[]
    checks=0
    max_resid=0.
    names=["O"]+[arm+"_"+key(dx) for arm in ARMS for dx in SHIFTS if dx]
    with ZipFile(args.archive) as z:
        for f in frames:
            with Image.open(io.BytesIO(z.read(f["truth"]))) as im:
                mm=np.asarray(im)[:,TRIM:-TRIM]
            assert mm.shape==(480,500) and np.issubdtype(mm.dtype,np.integer)
            valid=(mm>0)&(mm<=10000)
            path=out/"predictions"/(f["frame_id"]+".npz")
            hashes.append(dict(frame_id=f["frame_id"],sha256=sha(path)))
            with np.load(path,allow_pickle=False) as ck:
                assert str(ck["signature"])==signature
                arr={k:ck[k] for k in names}
                logs=json.loads(str(ck["log"]))
            for row in logs:
                k="O" if row["view"]=="O" else row["arm"]+"_"+row["view"]
                p=arr[k]
                assert p.shape==mm.shape and p.dtype==np.float32 and np.isfinite(p).all()
                assert hashlib.sha256(p.tobytes()).hexdigest()==row["aligned_sha256"]
            for i,j in itertools.product(range(4),repeat=2):
                x=((2*j+1)*500+4)//8-16
                y=((2*i+1)*480+4)//8-16
                cut=np.s_[y:y+32,x:x+32]
                mask=valid[cut]
                n=int(mask.sum())
                w=dict(frame_id=f["frame_id"],scene=f["scene"],window=4*i+j+1,source_x=x+TRIM,source_y=y,
                       valid=n,eligible=int(10*n>=9*1024),mm_sum=int(mm[cut][mask].sum(dtype=np.int64)))
                for k in names:
                    w[k]=float(arr[k][cut][mask].mean(dtype=np.float64)) if n else ""
                if args.smoke and w["eligible"]:
                    for arm,(a,b),(s,t) in itertools.product(ARMS,PAIRS,itertools.product((-1,1),repeat=2)):
                        ka=arm+"_"+key(s*a)
                        kb=arm+"_"+key(t*b)
                        first=((arr["O"][cut][mask].astype(np.float64)+arr[ka][cut][mask]+arr[kb][cut][mask])/3).mean()
                        max_resid=max(max_resid,abs(first-(w["O"]+w[ka]+w[kb])/3))
                        checks+=1
                windows.append(w)
    prefix="smoke_" if args.smoke else ""
    assert len(windows)==len(frames)*16 and max_resid<1e-12
    write(out/(prefix+"windows.csv"),windows)
    if not args.smoke:
        write(out/"prediction_files.csv",hashes)
    save(out/(prefix+"regions.json"),dict(frames=len(frames),windows=len(windows),verified_predictions=17*len(frames),
                                         eligible_windows=sum(w["eligible"] for w in windows),pixel_first_checks=checks,
                                         max_pixel_first_residual=max_resid,windows_sha256=sha(out/(prefix+"windows.csv"))))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE)
    ap.add_argument("--smoke",action="store_true")
    main(ap.parse_args())
