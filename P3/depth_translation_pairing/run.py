"""Collect four new aligned translations and reuse verified original depth predictions."""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import transformers

from metrics import HERE,OLD,VIEWS,sha,read,write,save,frames,step,displacement,align,valid_truth,evaluate
sys.path.insert(0,str(HERE.parent))
from depth import MODEL,REVISION,read_image,make_translation

def arrayhash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def translated(rgb,v):
    if v=="O":return make_translation(rgb,"O",0)[0]
    image,note=make_translation(rgb,{"L":"Tleft","R":"Tright"}[v[0]],int(v[1:]))
    assert note==f"dx={displacement(rgb.width,v)};dy=0"
    return image

def check_geometry(rgb):
    w,h=rgb.size;arr=np.asarray(rgb);xx=np.arange(w);edge=step(w,4)
    masks=[]
    for v in VIEWS:
        dx=displacement(w,v)
        # Independently fold destination x back to its original coordinate (reflect, no repeated edge).
        folded=(xx-dx)%(2*(w-1));source=np.minimum(folded,2*(w-1)-folded)
        actual=np.asarray(translated(rgb,v))
        assert np.array_equal(actual,arr[:,source])
        assert np.array_equal(align(actual,dx,edge),arr[:,edge:w-edge])
        masks.append((xx+dx>=0)&(xx+dx<w))
    assert np.array_equal(np.flatnonzero(np.logical_and.reduce(masks)),xx[edge:w-edge])
    return dict(width=w,height=h,dx={v:displacement(w,v) for v in VIEWS},x_start=edge,x_stop=w-edge)

def forward(model,inputs,size):
    torch.cuda.synchronize();t=perf_counter()
    with torch.inference_mode():
        pred=model(**inputs).predicted_depth
        assert pred.dtype==torch.float32
        torch.cuda.synchronize();ms=(perf_counter()-t)*1000
        full=torch.nn.functional.interpolate(pred[:,None],size=size,mode="bicubic",align_corners=False)[0,0].cpu().numpy()
    assert full.dtype==np.float32 and np.isfinite(full).all()
    return full,ms

def setup(args,root):
    env=json.loads((OLD/"environment.json").read_text())
    assert env["model"]==MODEL and env["revision"]==REVISION
    assert {k:sha(args.model/k) for k in env["weights"]}==env["weights"]
    for f in env["legacy_sources"]:assert sha(HERE.parent/f)==env["legacy_sources"][f]
    assert sha(HERE.parents[1]/"P2/views.py")==env["views_sha"] and sha(OLD/"measure.py")==env["script_sha"]
    assert sha(args.archive)==env["archive_sha"] and sha(OLD/"frames.csv")==env["frames_sha"]
    assert json.loads((OLD/"freeze.json").read_text())["frames_sha256"]==env["frames_sha"]
    assert (torch.__version__,transformers.__version__,np.__version__)==(env["torch"],env["transformers"],env["numpy"])
    dev=os.environ["CUDA_VISIBLE_DEVICES"]
    assert dev.startswith("GPU-") and torch.cuda.device_count()==1
    gpu=subprocess.check_output(["nvidia-smi","-i",dev,"--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    assert gpu.split(", ")[0]==dev
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    processor=transformers.AutoImageProcessor.from_pretrained(args.model,local_files_only=True)
    model=transformers.AutoModelForDepthEstimation.from_pretrained(args.model,local_files_only=True).eval().cuda()
    assert model.config.depth_estimation_type=="metric" and next(model.parameters()).dtype==torch.float32
    spec=dict(model=MODEL,revision=REVISION,weights=env["weights"],archive_sha256=env["archive_sha"],
              frames_sha256=sha(OLD/"frames.csv"),scene_source_sha256=sha(OLD/"frame_results.csv"),old_environment_sha256=sha(OLD/"environment.json"),
              historical_gpu=env["gpu"],gpu=gpu,hostname=socket.gethostname(),torch=torch.__version__,cuda=torch.version.cuda,
              transformers=transformers.__version__,numpy=np.__version__,python=platform.python_version(),processor=type(processor).__name__,
              dtype="FP32 model, preprocessing and cached predictions; FP64 evaluation; TF32 off",hashes={
                  f:sha(HERE/f) for f in ("protocol.md","metrics.py","run.py")})
    ep=root/"environment.json"
    if ep.exists():assert json.loads(ep.read_text())==spec
    else:save(ep,spec)
    return model,processor,sha(ep)

def process(args):
    start=perf_counter();root=args.out;cache=root/"predictions"
    cache.mkdir(parents=True,exist_ok=True)
    rows=frames();target=rows[:2] if args.smoke else rows
    model,processor,signature=setup(args,root)
    legacy=args.old_cache/sha(OLD/"environment.json")
    smoke_ids={r["frame_id"] for r in rows[:2]}
    if not args.smoke:
        proof=json.loads((root/"smoke_proof.json").read_text())
        assert proof["signature"]==signature and {k["frame_id"] for k in proof["original_replays"]}==smoke_ids
        assert all((cache/(fid+".npz")).exists() for fid in smoke_ids)
    shapes=[]
    for w in (125,157,640):
        y,x=np.indices((29,w))
        photo=Image.fromarray(np.stack((x%256,x//256,y),axis=-1).astype(np.uint8))
        shapes.append(check_geometry(photo))
    assert step(125,2)==3
    results=[];timings=[];replays=[];real_geometry=[]
    hits=legacy_hits=new=calibration=0
    with ZipFile(args.archive) as z:
        for r in target:
            fid=r["frame_id"];target_file=cache/(fid+".npz")
            rgb=read_image(z,r["frame"]).convert("RGB");w,h=rgb.size
            if fid in smoke_ids:real_geometry.append(dict(frame_id=fid,**check_geometry(rgb)))
            g,valid=valid_truth(np.asarray(read_image(z,r["truth"])))
            edge=step(w,4);old_edge=step(w,3)
            assert g.shape==(h,w-2*edge)
            if target_file.exists():
                with np.load(target_file,allow_pickle=False) as f:
                    assert str(f["signature"])==signature
                    arrays={v:f[v] for v in VIEWS};log=json.loads(str(f["log"]))
                    if args.smoke:replays.append(json.loads(str(f["original_check"])))
                hits+=len(VIEWS)
            else:
                rgb_sha=arrayhash(np.asarray(rgb));arrays={};log=[];check=None
                old_file=legacy/(fid+"-O.npz")
                with np.load(old_file,allow_pickle=False) as f:old=f["p"]
                assert old.dtype==np.float32 and old.shape==(h,w-2*old_edge) and np.isfinite(old).all()
                start_x=edge-old_edge
                arrays["O"]=old[:,start_x:start_x+g.shape[1]].copy();legacy_hits+=1
                for v in VIEWS:
                    dx=displacement(w,v);view=translated(rgb,v)
                    inputs=processor(images=view,return_tensors="pt")
                    t=inputs["pixel_values"]
                    assert t.dtype==torch.float32 and tuple(inputs.keys())==("pixel_values",)
                    digest=arrayhash(t.numpy())
                    ms=0.0
                    if v!="O" or fid in smoke_ids:
                        full,ms=forward(model,inputs.to("cuda"),(h,w))
                        if v=="O":
                            maxdiff=float(np.max(abs(align(full,0,old_edge)-old)))
                            assert maxdiff==0,f"{fid} fresh O differs from old cache by {maxdiff}"
                            check=dict(frame_id=fid,max_abs_difference=maxdiff,shape=list(old.shape),dtype=str(old.dtype),
                                       original_cache_sha256=sha(old_file),processor_tensor_sha256=digest)
                            replays.append(check);calibration+=1
                        else:
                            arrays[v]=align(full,dx,edge);new+=1
                    log.append(dict(frame_id=fid,view=v,dx=dx,rgb_sha256=rgb_sha,input_tensor_sha256=digest,
                                    aligned_prediction_sha256=arrayhash(arrays[v]),origin="verified_old_O" if v=="O" else "new_translation",
                                    source_old_npz_sha256=sha(old_file) if v=="O" else "",forward_ms=ms))
                temp=target_file.with_suffix(".tmp")
                with temp.open("wb") as f:np.savez(f,**arrays,signature=signature,log=json.dumps(log),original_check=json.dumps(check))
                temp.replace(target_file)
            vals=evaluate(arrays,g,valid)
            results.append(dict(**r,width=w,height=h,shift2_px=step(w,2),shift4_px=edge,**vals))
            timings.extend(log)
    prefix="smoke_" if args.smoke else ""
    assert len(timings)==len({(r["frame_id"],r["view"]) for r in timings})==5*len(target)
    if args.smoke:
        assert len(replays)==2 and all(c and c["max_abs_difference"]==0 for c in replays)
        save(root/"smoke_proof.json",dict(signature=signature,original_replays=replays,coordinate_tests=shapes,real_frames=real_geometry))
    write(root/(prefix+"frame_results.csv"),results);write(root/(prefix+"view_audit.csv"),timings)
    stats=dict(frames=len(results),new_translation_forwards=new,validation_original_forwards=calibration,
               old_original_cache_hits=legacy_hits,local_checkpoint_view_hits=hits,wall_seconds=perf_counter()-start)
    log=root/"runs.json";history=json.loads(log.read_text()) if log.exists() else []
    history.append(dict(stage="smoke" if args.smoke else "full",**stats));save(log,history);print(stats,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("archive","model","old-cache"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE);p.add_argument("--smoke",action="store_true")
    process(p.parse_args())
