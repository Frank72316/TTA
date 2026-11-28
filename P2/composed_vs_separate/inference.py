import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter
import numpy as np
from PIL import ImageEnhance

from study import BRIGHT, SHIFT, HERE, UUID, VIEWS, pixels, read, write
from geometry import original, PARAMS, AXES
from score import digest, init, save_json


def smoke_pixels(rgb,changed,b,t):
    axis=t.split("_")[1]
    ax,ay=AXES[axis]
    w,h=rgb.size
    dx=ax*((3*w+50)//100); dy=ay*((3*h+50)//100)
    def indices(n,delta):
        u=(np.arange(n)-delta)%(2*n-2)
        return np.minimum(u,2*n-2-u)
    bright=np.asarray(ImageEnhance.Brightness(rgb).enhance(float(b[1:])))
    assert np.array_equal(np.asarray(changed),bright[indices(h,dy)[:,None],indices(w,dx)])


def run(a):
    import torch
    start=perf_counter()
    gpu=subprocess.check_output(["nvidia-smi","-i","2","--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    assert gpu.split(",")[0]==os.environ["CUDA_VISIBLE_DEVICES"]==UUID
    model,tf,tok,env=init(a.pilot,"cuda")
    prior=json.loads((a.prior/"environment.json").read_text())
    # Same historical model loader, package versions, hardware, implementation,
    # transform, native output dtype and single-image batch.
    assert all(prior[k]==v for k,v in env.items()) and prior["image_batch_size"]==1
    assert prior["pairs_sha"]==digest(a.pairs)
    assert prior["validation_archives"]=={k:digest(a.pilot/"data"/k) for k in prior["validation_archives"]}
    env.update(gpu=gpu,visible_uuid=UUID,host=os.uname().nodename,batch=1,pairs_sha=digest(a.pairs),
               prior_scores_sha=digest(a.prior/"scores.csv"),script_sha=digest(Path(__file__)),
               study_sha=digest(HERE/"study.py"))
    assert env["host"].lower()=="asus4090x8" and gpu.endswith("580.159.04")
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=a.out/"environment.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())==env
    else:
        save_json(manifest,env)
    signature=digest(manifest)
    bank={(r["pair_id"],r["side"],r["view"]):r for r in read(a.prior/"scores.csv") if r["view"] in ("O",*BRIGHT,*SHIFT)}
    pairs=read(a.pairs)
    assert len(pairs)==len({p["prompt_id"] for p in pairs})==256 and len(bank)==3584
    allrows=[]; new=hits=resume=checks=0; gpu_s=0
    cache=a.out/"cache"; cache.mkdir(exist_ok=True)
    for p in pairs[:a.limit]:
        pid=p["pair_id"]; ck=cache/(pid+".json")
        if ck.exists():
            entry=json.loads(ck.read_text())
            assert entry["signature"]==signature and len(entry["rows"])==30
            resume+=30; allrows.extend(entry["rows"]); continue
        assert int(p["rank_A"])!=int(p["rank_B"])
        assert p["human_winner_P0"]==("A" if int(p["rank_A"])<int(p["rank_B"]) else "B")
        token=tok([p["P0"]]); token_sha=sha256(token.numpy().astype("<i8").tobytes()).hexdigest()
        rows=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            text=model.encode_text(token.cuda(),normalize=True)
            for side in "AB":
                rgb=original(a.pilot,p,side)
                raw_sha=sha256(rgb.tobytes()).hexdigest()
                for v in VIEWS:
                    image=pixels(rgb,v)
                    if a.limit==2 and "+" in v:
                        smoke_pixels(rgb,image,*v.split("+")); checks+=1
                    tensor=tf(image).contiguous()
                    pix=sha256(image.tobytes()).hexdigest(); key=sha256(tensor.numpy().tobytes()).hexdigest()
                    priorrow=bank.get((pid,side,v))
                    if priorrow:
                        assert (priorrow["original_rgb_sha"],priorrow["view_rgb_sha"],priorrow["tensor_sha"],priorrow["token_sha"])==(raw_sha,pix,key,token_sha)
                        assert (priorrow["P0"],priorrow["image_id"],priorrow["human"],priorrow["prompt_id"])==(p["P0"],p["image_"+side],p["human_winner_P0"],p["prompt_id"])
                        assert priorrow["family"]==PARAMS[v][0] and json.loads(priorrow["parameter"])==json.loads(json.dumps(PARAMS[v][1]))
                    if priorrow and a.limit!=2:
                        value=float(priorrow["score"]); hits+=1; origin="geometry_specificity/dev256"
                    else:
                        t=perf_counter()
                        features=model.encode_image(tensor[None].cuda(),normalize=True)
                        raw=(features@text.T).flatten()
                        assert raw.dtype==torch.float16 and raw.numel()==1
                        value=float(raw.item()); gpu_s+=perf_counter()-t
                        assert np.isfinite(value)
                        if priorrow: assert value==float(priorrow["score"]), (pid,side,v,value,priorrow["score"])
                        new+=1; origin="new"
                    rows.append(dict(pair_id=pid,prompt_id=p["prompt_id"],P0=p["P0"],side=side,image_id=p["image_"+side],
                        human=p["human_winner_P0"],view=v,original_rgb_sha=raw_sha,view_rgb_sha=pix,tensor_sha=key,
                        token_sha=token_sha,score=value,source=origin))
        save_json(ck,dict(signature=signature,rows=rows)); allrows.extend(rows)
        if len(allrows)%960==0: print(pid,flush=True)
    assert len(allrows)==len({(r["pair_id"],r["side"],r["view"]) for r in allrows})==a.limit*30
    write(a.out/("smoke_scores.csv" if a.limit==2 else "scores.csv"),allrows)
    log=a.out/"runs.json"; info=dict(limit=a.limit,new=new,historical_hits=hits,checkpoint_hits=resume,
        checked_composed_pixels=checks,gpu_seconds=gpu_s,wall_seconds=perf_counter()-start)
    save_json(log,(json.loads(log.read_text()) if log.exists() else [])+[info]); print(info,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("pairs","pilot","prior"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE)
    p.add_argument("--limit",type=int,choices=[2,256],required=True)
    run(p.parse_args())
