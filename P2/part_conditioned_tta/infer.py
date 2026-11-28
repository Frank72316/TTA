"""Offline HPS v2.1. Only inputs.csv (no attribute labels) enters the scoring loop."""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
import torch
from prepare import ROOT, as_json, digest, read, write

sys.path.insert(0, str(ROOT.parent/"within_image_partition"))
from score_views import init, displayed, hash_tensor

QUERIES = ("crown","wing")
VIEWS = ("original","crown","wing","center")
SMOKE = ("C001","C002")
GRAY = (128,128,128)

def prompt(color,q):
    return f"The bird has {color} plumage on its {q}."

def bounds(point,side):
    x,y=(math.floor(v-side/2+0.5) for v in point)
    return x,y,x+side,y+side

def intersection(a,b):
    return (max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]))

def area(b):
    return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def make_views(photo,r):
    w,h=photo.size
    # Exact integer form of floor(0.30*min + 0.5).
    s=max(1,(3*min(w,h)+5)//10)
    points={q:(float(r[q+"_x"]),float(r[q+"_y"])) for q in QUERIES}
    points["center"]=(w/2,h/2)
    boxes={q:bounds(p,s) for q,p in points.items()}
    cross_area=area(intersection(boxes["crown"],boxes["wing"]))
    output={"original":photo};geometry=[]
    for key,box in boxes.items():
        x0,y0,x1,y1=box
        inside=intersection(box,(0,0,w,h))
        patch=Image.new("RGB",(s,s),GRAY)
        patch.paste(photo.crop(inside), (inside[0]-x0,inside[1]-y0))
        # Independently compare every real and padded pixel against the source geometry.
        ys,xs=np.indices((s,s));xs+=x0;ys+=y0
        valid=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
        rgb=np.asarray(photo);pix=np.asarray(patch)
        assert np.array_equal(pix[valid],rgb[ys[valid],xs[valid]]) and (pix[~valid]==128).all()
        output[key]=patch
        geom=dict(view=key,x0=x0,y0=y0,x1=x1,y1=y1,source_width=w,source_height=h,side_px=s,
            center_x=points[key][0],center_y=points[key][1],padding_pixels=s*s-area(inside),
            padding_fraction=1-area(inside)/(s*s),crown_wing_iou=cross_area/(2*s*s-cross_area))
        for q,(x,y) in points.items():
            if q!="center":geom["contains_"+q]=int(x0<=x<x1 and y0<=y<y1)
        geometry.append(geom)
    return output,geometry

def smoke_panel(r,photo,geom,tiles,out):
    marked=photo.copy();draw=ImageDraw.Draw(marked)
    for k,col in [("crown","yellow"),("wing","cyan")]:
        g=next(i for i in geom if i["view"]==k)
        x,y=g["center_x"],g["center_y"]
        draw.rectangle((g["x0"],g["y0"],g["x1"]-1,g["y1"]-1),outline=col,width=2)
        draw.ellipse((x-3,y-3,x+3,y+3),fill=col)
    marked.thumbnail((224,224))
    sheet=Image.new("RGB",(5*224,252),"white");d=ImageDraw.Draw(sheet)
    titles=["source: crown/wing","O: full input","H: crown","W: "+("left" if r["wing_id"]=="9" else "right")+" wing","C: center"]
    for j,im in enumerate([marked]+[tiles[v] for v in VIEWS]):
        sheet.paste(im,(j*224,25));d.text((j*224+3,6),titles[j],fill="black")
    sheet.save(out/(r["pair_id"]+"_"+r["side"]+".png"))

def run(a):
    begin=perf_counter();out=a.out;out.mkdir(exist_ok=True,parents=True)
    cache=out/"checkpoints";images=out/"display_cache";smoke=out/"smoke"
    for p in (cache,images,smoke):p.mkdir(exist_ok=True)
    audit=json.loads((ROOT/"data_audit.json").read_text())
    assert digest(ROOT/"inputs.csv")==audit["inputs_sha256"]
    # Labels, species and confidence are intentionally absent from these inputs.
    rows=read(ROOT/"inputs.csv")
    assert all("_present" not in k and "_certainty" not in k and k!="species" for k in rows[0])
    selected=[r for r in rows if not a.smoke or r["pair_id"] in SMOKE]
    dev=os.environ["CUDA_VISIBLE_DEVICES"]
    assert dev.startswith("GPU-") and torch.cuda.device_count()==1
    gpu=subprocess.check_output(["nvidia-smi","-i",dev,"--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    assert gpu.split(", ")[1]==dev
    if not a.smoke:
        assert (out/"smoke_run.json").is_file()
        assert all((cache/(r["image_id"]+".json")).exists() for r in rows if r["pair_id"] in SMOKE)
    model,tf,tokenizer,env=init(a.pilot,"cuda")
    original_env=json.loads((ROOT.parent/"within_image_partition"/"environment.json").read_text())
    assert all(original_env[k]==v for k,v in env.items()) and not model.training
    env.update(gpu=gpu,hostname=socket.gethostname(),inputs_sha256=audit["inputs_sha256"],runner_sha256=digest(Path(__file__)),
               helper_sha256=digest(ROOT.parent/"within_image_partition"/"score_views.py"),batch_image=1,batch_text=1,score_dtype="float16")
    ep=out/"environment.json"
    if ep.exists():assert json.loads(ep.read_text())==env
    else:as_json(ep,env)
    signature=digest(ep);texts={};scores=[];geometry=[];hits=0;visual=0
    loop=perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for r in selected:
            fid=r["image_id"];file=cache/(fid+".json")
            source=a.data/"images"/r["path"];assert digest(source)==r["jpeg_sha256"]
            if file.exists():
                record=json.loads(file.read_text());assert record["signature"]==signature and len(record["scores"])==8
                scores.extend(record["scores"]);geometry.extend(record["geometry"]);hits+=8;continue
            with Image.open(source) as im:photo=im.convert("RGB")
            assert photo.size==(int(r["width"]),int(r["height"]))
            views,geom=make_views(photo,r)
            for g in geom:g.update(image_id=fid,pair_id=r["pair_id"],candidate=r["side"],wing_id=r["wing_id"])
            current=[];tiles={}
            for view in VIEWS:
                t=tf(views[view]).contiguous()
                assert t.shape==(3,224,224)
                im=tiles[view]=displayed(t,tf);im.save(images/f"{fid}_{view}.png")
                th=hash_tensor(t)
                feat=model.encode_image(t[None].cuda(),normalize=True);visual+=1
                for q in QUERIES:
                    wording=prompt(r["color"],q)
                    if wording not in texts:
                        tok=tokenizer([wording]);assert 0<tok.count_nonzero()<tok.numel()
                        texts[wording]=(model.encode_text(tok.cuda(),normalize=True),hash_tensor(tok))
                    tx,toksha=texts[wording];native=(feat@tx.T).flatten()
                    assert native.numel()==1 and native.dtype==torch.float16
                    val=float(native.item());assert math.isfinite(val)
                    current.append(dict(image_id=fid,view=view,query=q,score=val,pair_id=r["pair_id"],side=r["side"],
                                        prompt=wording,tensor_sha256=th,token_sha256=toksha))
            if r["pair_id"] in SMOKE:smoke_panel(r,photo,geom,tiles,smoke)
            tmp=file.with_suffix(".tmp");as_json(tmp,dict(signature=signature,scores=current,geometry=geom));tmp.replace(file)
            scores.extend(current);geometry.extend(geom)
    assert len(scores)==8*len(selected)==len({(s["image_id"],s["view"],s["query"]) for s in scores})
    torch.cuda.synchronize()
    stats=dict(photos=len(selected),new_visual=visual,new_scores=2*visual,checkpoint_score_hits=hits,
               new_text=len(texts),loop_seconds=perf_counter()-loop,wall_seconds=perf_counter()-begin)
    prefix="smoke_" if a.smoke else ""
    write(out/(prefix+"scores.csv"),scores);write(out/(prefix+"geometry.csv"),geometry)
    as_json(out/(prefix+"run.json"),stats);print(stats,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT);p.add_argument("--smoke",action="store_true");run(p.parse_args())
