"""Fresh five-view bank with pinned upstream model, transforms, inverse alignment and evaluation."""
import argparse
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile
import numpy as np
import torch
from holdout_lib import HERE,P3,ORDER,TRANS,VIEWS,sha,read,write,save,frozen_rows,valid_truth,window_records,pair_record
from run import setup,translated,check_geometry,forward,arrayhash,step,displacement,align,read_image
from PIL import Image

def predict(args):
    started=perf_counter();out=args.out;cache=out/"predictions";cache.mkdir(exist_ok=True,parents=True)
    rows=frozen_rows()
    dev={r["metric"]:r for r in read(HERE/"development_replay.csv")}
    assert abs(float(dev["G"]["estimate"])-.000745)<1e-6 and abs(float(dev["A"]["estimate"])-.0651)<1e-4
    pinned=json.loads((HERE/"sources.json").read_text())["source_files"]
    for name in ("P3/depth_translation_pairing/run.py","P3/depth_translation_pairing/metrics.py",
                 "P3/depth_order_pairing/order_data.py","P3/depth_order_pairing/summarize.py"):
        assert sha(P3.parent/name)==pinned[name],name
    # Load the same offline snapshot using the upstream setup; keep the new cohort metadata explicit.
    with TemporaryDirectory() as tmp:
        model,processor,_=setup(args,Path(tmp))
        meta=json.loads((Path(tmp)/"environment.json").read_text())
    env={k:meta[k] for k in ("model","revision","weights","archive_sha256","gpu","hostname","torch","cuda","transformers","numpy","python","processor","dtype")}
    env.update(frames_sha256=sha(HERE/"frames.csv"),freeze_sha256=sha(HERE/"freeze.json"),source_evaluation_sha256=sha(ORDER/"order_data.py"),
               implementation_hashes={f:sha(HERE/f) for f in ("predict_holdout.py","holdout_lib.py","evaluate_holdout.py")},
               dev_replay_sha256=sha(HERE/"development_replay.csv"),batch=1)
    ep=out/"environment.json"
    if ep.exists():assert json.loads(ep.read_text())==env
    else:save(ep,env)
    sig=sha(ep);smoke_ids=[r["frame_id"] for r in rows[:2]]
    if not args.smoke:
        proof=json.loads((out/"smoke.json").read_text())
        assert proof["frame_ids"]==smoke_ids and proof["signature"]==sig
        for fid,h in proof["npz_sha256"].items():assert sha(cache/(fid+".npz"))==h
    target=rows[:2] if args.smoke else rows
    audit=[];files=[];checks=[];new=hits=0
    if args.smoke:
        for w in (125,157,640):
            y,x=np.indices((29,w));check_geometry(Image.fromarray(np.stack((x%256,x//256,y),axis=-1).astype(np.uint8)))
    with ZipFile(args.archive) as z:
        for r in target:
            fid=r["frame_id"];path=cache/(fid+".npz")
            rgb=read_image(z,r["frame"]).convert("RGB");width,height=rgb.size;edge=step(width,4)
            raw=arrayhash(np.asarray(rgb));assert (width,height)==(640,480)
            if path.exists():
                with np.load(path,allow_pickle=False) as a:
                    assert str(a["signature"])==sig and str(a["frame_id"])==fid
                    log=json.loads(str(a["log"]));pred={v:a[v] for v in VIEWS}
                    assert all(l["rgb_sha256"]==raw and l["aligned_sha256"]==arrayhash(pred[l["view"]]) for l in log)
                hits+=len(VIEWS)
            else:
                log=[];pred={}
                for v in VIEWS:
                    dx=displacement(width,v);image=translated(rgb,v)
                    x=processor(images=image,return_tensors="pt")
                    tensor=x["pixel_values"];assert tensor.dtype==torch.float32
                    ih=arrayhash(tensor.numpy());p,ms=forward(model,x.to("cuda"),(height,width))
                    pred[v]=align(p,dx,edge)
                    log.append(dict(frame_id=fid,view=v,dx=dx,rgb_sha256=raw,input_tensor_sha256=ih,
                                    aligned_sha256=arrayhash(pred[v]),forward_ms=ms))
                    new+=1
                tmp=path.with_suffix(".tmp")
                with tmp.open("wb") as f:np.savez(f,**pred,signature=sig,frame_id=fid,log=json.dumps(log))
                tmp.replace(path)
            assert {l["view"] for l in log}==set(VIEWS) and len(log)==5
            audit.extend(log);files.append(dict(frame_id=fid,scene=r["scene"],file="predictions/"+path.name,npz_sha256=sha(path)))
            if args.smoke:
                coords=check_geometry(rgb);assert coords["dx"]=={"O":0,"L2":-13,"R2":13,"L4":-26,"R4":26}
                mm=np.array(read_image(z,r["truth"]))
                g,valid=valid_truth(mm);assert np.array_equal(g,mm[:,26:614].astype(np.float64)/1000)
                assert all(v.shape==g.shape and v.dtype==np.float32 and np.isfinite(v).all() for v in pred.values())
                windows,px=window_records(r,mm[:,26:614],g,valid,pred,True)
                pairs=[pair_record(a,b,px) for a,b in combinations(windows,2) if a["eligible"] and b["eligible"]]
                checks.append(dict(frame_id=fid,coordinates=coords,valid_pixels=int(valid.sum()),eligible_windows=sum(w["eligible"] for w in windows),
                                   region_pairs=len(pairs),independent_recipe_checks=5*len(pairs),
                                   max_identity_residual=max(p["identity_error"] for p in pairs)))
    assert len(audit)==len({(r["frame_id"],r["view"]) for r in audit})==5*len(target)
    prefix="smoke_" if args.smoke else ""
    write(out/(prefix+"view_audit.csv"),audit);write(out/(prefix+"prediction_files.csv"),files)
    if args.smoke:save(out/"smoke.json",dict(signature=sig,frame_ids=smoke_ids,checks=checks,npz_sha256={f["frame_id"]:f["npz_sha256"] for f in files}))
    prev=out/"runs.json";history=json.loads(prev.read_text()) if prev.exists() else []
    stats=dict(stage="smoke" if args.smoke else "full",frames=len(target),new_forwards=new,checkpoint_view_hits=hits,wall_seconds=perf_counter()-started)
    history.append(stats);save(prev,history);print(stats)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--archive",type=Path,required=True);ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE);ap.add_argument("--smoke",action="store_true");predict(ap.parse_args())
