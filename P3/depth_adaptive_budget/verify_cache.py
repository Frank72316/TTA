"""Independent CPU replay from the aligned per-view predictions."""
import argparse
import csv
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from freeze import HERE, sha


def verify(archive, root):
    env=json.loads((HERE/"environment.json").read_text())
    assert sha(archive)==env["archive_sha"]
    folder=root/sha(HERE/"environment.json")
    with (HERE/"measurements.csv").open() as f: rows=list(csv.DictReader(f))
    errors={k:0. for k in ("D","MSE2","MSE3L","MSE3R","MSE4")}
    recomputed=[]
    with ZipFile(archive) as z:
        for r in rows:
            aligned=[]
            for v in "OFLR":
                with np.load(folder/(r["frame_id"]+"-"+v+".npz")) as data:
                    aligned.append(data["p"].astype(np.float64))
            o,f,l,right=aligned
            observed={"D":np.square(o-f).mean()}
            with Image.open(BytesIO(z.read(r["truth"]))) as im:
                truth=np.asarray(im,dtype=np.float32)/1000
            w=truth.shape[1]; border=(w-o.shape[1])//2
            assert border==int(w*.03+.5)
            truth=truth[:,border:w-border].astype(np.float64)
            valid=(truth>0)&(truth<=10)
            assert valid.sum()==int(r["valid_pixels"])
            for k,parts in (("MSE2",[o,f]),("MSE3L",[o,f,l]),("MSE3R",[o,f,right]),("MSE4",aligned)):
                pred=np.mean(parts,axis=0)
                observed[k]=np.square(pred[valid]-truth[valid]).mean()
            for k,v in observed.items(): errors[k]=max(errors[k],abs(v-float(r[k])))
            recomputed.append((observed["D"],r["frame_id"]))
    selected={fid for _,fid in sorted(recomputed,key=lambda x:(-x[0],x[1]))[:64]}
    assert selected=={r["frame_id"] for r in rows if r["selected"]=="1"}
    assert errors["D"]==0 and max(errors.values())<1e-6
    proof=dict(frames=len(rows),predictions=4*len(rows),reproduced_selection=len(selected),
        max_abs_error_m2=errors,method="Average aligned float64 predictions, then subtract metric truth")
    (HERE/"cache_audit.json").write_text(json.dumps(proof,indent=2))
    print(proof)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",type=Path,required=True); ap.add_argument("--cache",type=Path,required=True)
    a=ap.parse_args(); verify(a.archive,a.cache)
