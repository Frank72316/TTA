import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
import torch
from engine import Engine,HERE,AXES,original,view,digest,save_json,hashed,write

def moved(a,dx,dy,mode):
    h,w=a.shape[:2]; x,y=abs(dx),abs(dy)
    return np.pad(a,((y,y),(x,x),(0,0)),mode=mode)[y-dy:y-dy+h,x-dx:x-dx+w]

def run(root,out):
    start=perf_counter(); engine=Engine(root,2)
    env=dict(**engine.meta,script=digest(Path(__file__)),engine=digest(HERE/"engine.py"),protocol=digest(HERE/"protocol.md"))
    out.mkdir(exist_ok=True,parents=True); cache=out/"cache";cache.mkdir(exist_ok=True)
    if (out/"environment.json").exists(): assert json.loads((out/"environment.json").read_text())==env
    else: save_json(out/"environment.json",env)
    signature=digest(out/"environment.json"); allrows=[]; audits=[]; masks={}; repeats=[]; resumed=0
    std=np.array(engine.tf.transforms[-1].normalize.std,dtype=float)[:,None,None]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for i,p in enumerate(engine.pairs):
            pid=p["pair_id"]; ck=cache/(pid+".json"); pack=cache/(pid+".npz")
            if ck.exists():
                entry=json.loads(ck.read_text()); assert entry["key"]==signature and pack.exists();resumed+=1
            else:
                engine.prompt(p); rows=[]; checks=[]; repeated=[]; tensors={}; vals={}; mask_rows={}
                for side in "AB":
                    rgb=original(engine.pilot,p,side); a=np.asarray(rgb); h,w=a.shape[:2]
                    ot=engine.tf(rgb).contiguous()
                    assert np.array_equal(moved(a,0,0,"reflect"),moved(a,0,0,"edge")) and np.array_equal(moved(a,0,0,"edge"),a)
                    val=engine.historical(p,side,"O",rgb,rgb,ot,i<2)
                    def record(k,v,t):
                        vals[side,k]=v;tensors[side,k]=t
                        rows.append(dict(pair_id=pid,side=side,image_id=p["image_"+side],human=p["human_winner_P0"],
                            view=k,score=v,tensor_sha=hashed(t.numpy()),token_sha=engine.token))
                    record("O",val,ot)
                    for name,(sx,sy) in AXES.items():
                        dx=sx*((3*w+50)//100);dy=sy*((3*h+50)//100)
                        y,x=np.ogrid[:h,:w]
                        overlap=(x-dx>=0)&(x-dx<w)&(y-dy>=0)&(y-dy<h)
                        ref=view(rgb,"T_"+name+"_3"); edge=moved(a,dx,dy,"edge")
                        assert np.array_equal(np.asarray(ref)[overlap],edge[overlap])
                        assert np.array_equal(np.asarray(ref)[overlap],a[(y+np.zeros_like(x)-dy)[overlap],(x+np.zeros_like(y)-dx)[overlap]])
                        rt=engine.tf(ref).contiguous(); et=engine.tf(Image.fromarray(edge)).contiguous()
                        rv=engine.historical(p,side,"T_"+name+"_3",rgb,ref,rt,i<2)
                        record("ref_"+name,rv,rt); record("edge_"+name,engine.score(et),et)
                        source=np.abs(np.asarray(ref).astype(float)-edge)/255
                        delta=np.abs(rt.numpy().astype(float)-et.numpy())*std
                        mask=(delta!=0).any(axis=0); key=pid+"_"+side+"_"+name;mask_rows[key]=np.packbits(mask,axis=None)
                        checks.append(dict(pair_id=pid,side=side,direction=name,dx=dx,dy=dy,
                            border_fraction=(~overlap).mean(),border_rgb_mae=source[~overlap].mean(),source_mae=source.mean(),
                            source_changed=(source!=0).any(axis=2).mean(),input_mae=delta.mean(),input_changed=mask.mean(),
                            input_changed_mae=delta[:,mask].mean() if mask.any() else 0,mask_key=key))
                for name in AXES:
                    keys=["O","ref_"+name,"edge_"+name]
                    margins=[vals["A",k]-vals["B",k] for k in keys]
                    agg=(np.array(margins[1:])+margins[0])/2
                    step=max(float(np.spacing(np.float16(vals[s,k]))) for s in "AB" for k in keys)
                    if np.sign(agg[0])!=np.sign(agg[1]) and min(abs(agg))<=step:
                        for s in "AB":
                            for k in keys:
                                val=engine.score(tensors[s,k]); repeated.append(dict(pair_id=pid,direction=name,side=s,view=k,
                                    original_score=vals[s,k],repeat_score=val,ulp=step))
                np.savez_compressed(pack,**mask_rows)
                entry=dict(key=signature,rows=rows,checks=checks,repeats=repeated);save_json(ck,entry)
            allrows.extend(entry["rows"]);audits.extend(entry["checks"]);repeats.extend(entry["repeats"])
            with np.load(pack) as f: masks.update({k:f[k] for k in f.files})
            if i==1 or (i+1)%64==0:print(i+1,flush=True)
    assert len(allrows)==4608 and len(masks)==2048
    write(out/"scores.csv",allrows);write(out/"pixels.csv",audits)
    if repeats:write(out/"repeated.csv",repeats)
    np.savez_compressed(out/"input_masks.npz",**masks)
    save_json(out/"run.json",dict(wall_seconds=perf_counter()-start,visual_forwards=engine.visual,
        historical_hits=engine.hits,resumed_pairs=resumed,repeated_records=len(repeats),smoke_pairs=[p["pair_id"] for p in engine.pairs[:2]]))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--out",type=Path,default=HERE)
    a=ap.parse_args();run(a.root,a.out)
