"""Score predictions first, freeze the scheduler, then evaluate against depth."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
import torch
import transformers
from scipy.io import loadmat

from freeze import HERE, OLD, sha, csv_bytes, immutable, freeze
sys.path.insert(0, str(HERE.parent))
from depth import REVISION, MODEL, read_image, make_translation

VIEWS = ("O", "F", "L", "R")
UUID = "GPU-15f75a68-cf82-1b07-45c5-8c805812b8b4"


def transformed(rgb, view):
    if view == "F":
        return ImageOps.mirror(rgb)
    return make_translation(rgb, {"O":"O","L":"Tleft","R":"Tright"}[view], 3)[0]


def align(p, view):
    w = p.shape[1]
    edge = (3*w + 50)//100
    if view == "F":
        p = p[:, ::-1]
    dx = {"L":-edge, "R":edge}.get(view, 0)
    return p[:, edge+dx:w-edge+dx].copy()


def check(rgb):
    reference = align(np.asarray(rgb), "O")
    for v in VIEWS:
        assert np.array_equal(align(np.asarray(transformed(rgb,v)),v), reference), v


def predict(model, processor, rgb, v):
    start = perf_counter()
    derived = transformed(rgb,v)
    t1 = perf_counter()
    inputs = processor(images=derived, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t2 = perf_counter()
    with torch.inference_mode():
        output = model(**inputs).predicted_depth[:,None]
        torch.cuda.synchronize()
        t3 = perf_counter()
        array = torch.nn.functional.interpolate(output,size=(rgb.height,rgb.width),
                    mode="bicubic",align_corners=False)[0,0].cpu().numpy()
    t4 = perf_counter()
    p = align(array,v)
    t5 = perf_counter()
    assert p.dtype == np.float32 and np.isfinite(p).all()
    labels = ("view_ms","processor_h2d_ms","forward_ms","resize_d2h_ms","align_ms")
    return p, dict(zip(labels,1000*np.diff([start,t1,t2,t3,t4,t5])))


def run(a):
    start = perf_counter()
    freeze(a.archive)
    meta = json.loads((OLD/"depth_environment.json").read_text())
    assert {k:sha(a.model/k) for k in meta["model_sha256"]} == meta["model_sha256"]
    assert meta["model_revision"] == REVISION and meta["model"] == MODEL
    gpu = subprocess.check_output(["nvidia-smi","-i","3","--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    assert gpu.split(",")[0] == os.environ["CUDA_VISIBLE_DEVICES"] == UUID and torch.cuda.device_count()==1
    assert torch.__version__ == meta["torch_build"] and transformers.__version__ == meta["transformers"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    proc = transformers.AutoImageProcessor.from_pretrained(a.model,local_files_only=True)
    model = transformers.AutoModelForDepthEstimation.from_pretrained(a.model,local_files_only=True).eval().cuda()
    assert model.config.depth_estimation_type == "metric" and next(model.parameters()).dtype == torch.float32
    env = dict(model=MODEL,revision=REVISION,weights=meta["model_sha256"],torch=torch.__version__,
        transformers=transformers.__version__,numpy=np.__version__,python=platform.python_version(),
        gpu=gpu,host=platform.node(),precision="float32, TF32 disabled",split_sha=sha(a.split),
        frames_sha=sha(HERE/"frames.csv"),archive_sha=sha(a.archive),script_sha=sha(Path(__file__)),
        legacy_sources={p:sha(HERE.parent/p) for p in ("depth.py","score_gpu.py")},views_sha=sha(HERE.parents[1]/"P2/views.py"))
    immutable(HERE/"environment.json",json.dumps(env,indent=2).encode())
    cache = a.cache / sha(HERE/"environment.json")
    cache.mkdir(parents=True,exist_ok=True)
    with (HERE/"frames.csv").open() as f: rows=list(csv.DictReader(f))
    for w in (100,157):
        yy,xx=np.indices((95,w))
        check(Image.fromarray(np.stack((xx,yy,(xx+yy)%256),axis=-1).astype("uint8")))
    times=[]; metrics=[]; schedule=[]; new=hits=0
    with ZipFile(a.archive) as z:
        all_ids = {int(Path(n).stem.split("_")[-1]) for n in z.namelist() if "/images/val/" in n and n.endswith(".jpg")}
        test = set(map(int,loadmat(a.split)["testNdxs"].ravel()))
        assert len(test)==654 and {i+1 for i in all_ids}==test, "Frame numbering or split mismatch"
        for row in rows[:2]:
            check(read_image(z,row["frame"]).convert("RGB"))
        for phase in ("OF","LR"):
            for index,row in enumerate(rows):
                fid=row["frame_id"]
                rgb=read_image(z,row["frame"]).convert("RGB")
                for view in phase:
                    target=cache/f"{fid}-{view}.npz"
                    if target.exists():
                        hits+=1
                    else:
                        p,ms=predict(model,proc,rgb,view)
                        with target.with_suffix(".tmp").open("wb") as f:
                            np.savez(f,p=p,ms=json.dumps(ms))
                        target.with_suffix(".tmp").replace(target)
                        new+=1
                    with np.load(target,allow_pickle=False) as entry:
                        times.append(dict(frame_id=fid,view=view,warmup=int(phase=="OF" and index<2),
                                          **json.loads(str(entry["ms"]))))
                if phase=="OF":
                    with np.load(cache/f"{fid}-O.npz") as o, np.load(cache/f"{fid}-F.npz") as f:
                        diff=o["p"].astype(np.float64)-f["p"]
                        schedule.append(dict(frame_id=fid,D=np.mean(diff**2),geometry_pixels=diff.size))
                if (index+1)%32==0: print(phase,index+1,flush=True)
            if phase=="OF":
                order=sorted(schedule,key=lambda r:(-r["D"],r["frame_id"]))
                for rank,r in enumerate(order,1): r.update(rank_D=rank,selected=int(rank<=64))
                immutable(HERE/"schedule.csv",csv_bytes(schedule))
        # Ground-truth PNGs are opened only after the prediction-only schedule has been written.
        for row,sched in zip(rows,schedule):
            fid=row["frame_id"]
            gt=align(np.array(read_image(z,row["truth"]),dtype=np.float32)/1000,"O")
            valid=(gt>0)&(gt<=10)
            assert gt.size==sched["geometry_pixels"] and valid.any()
            residual=[]
            for view in VIEWS:
                with np.load(cache/f"{fid}-{view}.npz") as entry:
                    pred=entry["p"]
                    assert pred.shape==gt.shape
                    residual.append((pred[valid]-gt[valid]).astype(np.float64))
            o,f,l,r=residual
            mse=lambda err:float(np.mean(err**2))
            m2=mse((o+f)/2); m4=mse((o+f+l+r)/4)
            metrics.append(dict(**row,**{k:v for k,v in sched.items() if k!="frame_id"},
                valid_pixels=int(valid.sum()),MSE2=m2,MSE3L=mse((o+f+l)/3),MSE3R=mse((o+f+r)/3),MSE4=m4,G=m2-m4))
    immutable(HERE/"measurements.csv",csv_bytes(metrics))
    immutable(HERE/"timings.csv",csv_bytes(times))
    assert len(times)==len({(r["frame_id"],r["view"]) for r in times})==512 and len(metrics)==128
    log=HERE/"runs.json"
    runs=json.loads(log.read_text()) if log.exists() else []
    runs.append(dict(new_forwards=new,checkpoint_hits=hits,wall_seconds=perf_counter()-start,
        schedule_sha=sha(HERE/"schedule.csv"),common_geometry="x in [s,W-s), full height",matched_official_test_ids=len(test),
        smoke_frames=[r["frame_id"] for r in rows[:2]],coordinate_test_widths=[100,157]))
    log.write_text(json.dumps(runs,indent=2))
    print(runs[-1],flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    for key in ("archive","model","split","cache"): ap.add_argument("--"+key,type=Path,required=True)
    run(ap.parse_args())
