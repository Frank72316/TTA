"""Inference at integer translations of one preprocessed tensor. Keeps aligned metric depth on Asus."""
import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import time
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from study import HERE, PRIOR, SHIFTS, BORDER, TRIM, key, read, save, load, sha, write

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def views(parent,dx):
    w=parent.shape[-1]-2*BORDER
    center=parent[...,BORDER:BORDER+w]
    idx=torch.arange(w)-dx
    idx=idx.remainder(2*(w-1))
    reflected=center.index_select(-1,torch.minimum(idx,2*(w-1)-idx))
    real=parent[...,BORDER-dx:BORDER-dx+w]
    keep=torch.arange(w)-dx
    keep=(keep>=0)&(keep<w)
    assert torch.equal(real[...,keep],reflected[...,keep])
    assert torch.equal(real[...,keep],center[...,torch.arange(w)[keep]-dx])
    diff=(real!=reflected).any(dim=1)[0]
    assert not diff[:,keep].any() and (~keep).sum()==abs(dx)
    return reflected.contiguous(),real.contiguous(),int(diff.sum())

def align(pred,dx):
    # Embed the intersection back into the complete preprocessed photo before bicubic resampling.
    assert pred.shape==(518,602) and pred.dtype==torch.float32
    margin=max(SHIFTS)
    canvas=pred.new_zeros((1,1,518,686))
    canvas[0,0,:,BORDER+margin:686-BORDER-margin]=pred[:,margin+dx:602-margin+dx]
    native=F.interpolate(canvas,size=(480,640),mode="bicubic",align_corners=False)
    return native[0,0,:,TRIM:640-TRIM].contiguous()

def check_coordinates():
    x=torch.arange(686,dtype=torch.float32).repeat(518,1)
    ramp=x[None,None]
    ref=F.interpolate(ramp,size=(480,640),mode="bicubic",align_corners=False)[0,0,:,TRIM:-TRIM]
    for dx in SHIFTS:
        mirrored,real,_=views(ramp,dx)
        for t in (mirrored,real):
            assert torch.equal(align(t[0,0],dx),ref)
    # Independent interpolation-support check in align_corners=False pixel centres.
    u=(np.arange(TRIM,640-TRIM)+.5)*686/640-.5
    low=int(np.floor(u.min())-1)
    high=int(np.floor(u.max())+2)
    assert low>=BORDER+31 and high<686-BORDER-31
    return dict(source_x=[TRIM,640-TRIM],parent_support_minmax=[low,high],
                ramp_checks=2*len(SHIFTS),max_error=0)

def process(args):
    t0=time.perf_counter()
    out=args.out
    cache=out/"predictions"
    cache.mkdir(parents=True,exist_ok=True)
    frozen=load(HERE/"freeze.json")
    assert (sha(HERE/"frames.csv"),sha(HERE/"protocol.md"))==(frozen["frame_sha256"],frozen["protocol_sha256"])
    base=load(PRIOR/"environment.json")
    assert sha(args.archive)==base["archive_sha"]
    assert {f:sha(args.model/f) for f in base["weights"]}==base["weights"]
    assert (torch.__version__,transformers.__version__,np.__version__)==(base["torch"],base["transformers"],base["numpy"])
    uuid=os.environ["CUDA_VISIBLE_DEVICES"]
    assert uuid.startswith("GPU-") and torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    processor=transformers.AutoImageProcessor.from_pretrained(args.model,local_files_only=True)
    model=transformers.AutoModelForDepthEstimation.from_pretrained(args.model,local_files_only=True).eval().cuda()
    proj=model.backbone.embeddings.patch_embeddings.projection
    assert proj.kernel_size==proj.stride==(14,14) and proj.padding==(0,0)
    assert model.config.depth_estimation_type=="metric" and next(model.parameters()).dtype==torch.float32
    coordinates=check_coordinates()
    spec=dict(model=base["model"],revision=base["revision"],weights=base["weights"],archive_sha=base["archive_sha"],
              freeze_sha=sha(HERE/"freeze.json"),code={f:sha(HERE/f) for f in ("study.py","run.py")},
              hardware=subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip(),
              host=platform.node(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
              cuda=torch.version.cuda,dtype="FP32, TF32 off",processor=type(processor).__name__,
              parent_hw=[518,686],canvas_hw=[518,602],kernel=list(proj.kernel_size),stride=list(proj.stride),padding=list(proj.padding))
    ep=out/"environment.json"
    if ep.exists():
        assert spec==load(ep)
    else:
        save(ep,spec)
    signature=sha(ep)
    frames=read(HERE/"frames.csv")
    if args.smoke:
        frames=frames[:2]
    else:
        gate=load(out/"smoke_proof.json")
        assert gate["signature"]==signature and gate["ids"]==[f["frame_id"] for f in frames[:2]]
        assert all((cache/(f["frame_id"]+".npz")).is_file() for f in frames[:2])
    counts={"reflect":0,"real_extra":0,"hits":0,"preprocess":0}
    audit=[]
    grid_shapes=set()
    def hook(_m,_args,out_tensor):
        assert tuple(out_tensor.shape)==(1,384,37,43)
        grid_shapes.add(tuple(out_tensor.shape))
    handle=proj.register_forward_hook(hook)
    with ZipFile(args.archive) as z,torch.inference_mode():
        for f in frames:
            fid=f["frame_id"]
            file=cache/(fid+".npz")
            if file.exists():
                with np.load(file,allow_pickle=False) as old:
                    assert str(old["signature"])==signature
                    log=json.loads(str(old["log"]))
                counts["hits"]+=17
            else:
                with Image.open(io.BytesIO(z.read(f["frame"]))) as im:
                    assert im.size==(640,480)
                    parent=processor(images=im.convert("RGB"),return_tensors="pt").pixel_values
                assert tuple(parent.shape)==(1,3,518,686) and parent.dtype==torch.float32
                counts["preprocess"]+=1
                parent_sha=tensor_hash(parent)
                arrays={}
                log=[]
                for dx in SHIFTS:
                    refl,real,changed=views(parent,dx)
                    if dx!=0:
                        assert changed>0
                    for arm,t in ([("shared",refl)] if dx==0 else [("reflect",refl),("real",real)]):
                        x=t.to("cuda")
                        torch.cuda.synchronize()
                        t1=time.perf_counter()
                        pred=model(pixel_values=x).predicted_depth
                        torch.cuda.synchronize()
                        ms=(time.perf_counter()-t1)*1000
                        assert pred.shape==(1,518,602) and pred.dtype==torch.float32
                        aligned=align(pred[0].cpu(),dx)
                        assert aligned.shape==(480,500) and torch.isfinite(aligned).all()
                        name="O" if dx==0 else arm+"_"+key(dx)
                        arrays[name]=aligned.numpy()
                        counts["real_extra" if arm=="real" else "reflect"]+=1
                        log.append(dict(frame_id=fid,arm=arm,view=key(dx),dx=dx,parent_tensor_sha256=parent_sha,
                                        input_sha256=tensor_hash(t),aligned_sha256=tensor_hash(aligned),
                                        exposed_pixels=518*abs(dx),differing_pixels=changed,
                                        differing_overlap_pixels=0,feature_grid="37x43",forward_ms=ms))
                tmp=file.with_suffix(".tmp")
                with tmp.open("wb") as o:
                    np.savez(o,**arrays,signature=signature,log=json.dumps(log))
                tmp.replace(file)
            assert len(log)==17
            audit.extend(log)
    handle.remove()
    stage="smoke" if args.smoke else "full"
    write(out/("smoke_views.csv" if args.smoke else "view_audit.csv"),audit)
    if args.smoke:
        save(out/"smoke_proof.json",dict(signature=signature,ids=[r["frame_id"] for r in frames],coordinates=coordinates,
            feature_grid=[37,43],unique_predictions=len(audit),all_alignment_checks_passed=True))
    if counts["preprocess"]:
        assert grid_shapes=={(1,384,37,43)}
    logpath=out/"runs.json"
    history=load(logpath) if logpath.exists() else []
    history.append(dict(stage=stage,**counts,wall_seconds=time.perf_counter()-t0))
    save(logpath,history)
    print(history[-1],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE)
    ap.add_argument("--smoke",action="store_true")
    process(ap.parse_args())
