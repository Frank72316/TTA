"""Freeze continuous COCO geometry before reading any score cache."""
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"prompt_aware_flip"
NOUN=HERE.parent/"crop_noun_controls"
sys.path.insert(0,str(SOURCE))
from common import rows,save_rows,save_json,sha,valid_rect

LINES=(25,50,75)
ROLES=("x","y","alt_x","alt_y")
KEPT=("left","right")
GRAY=128

def digest(path):return sha(path.read_bytes())

def clipped(box,w,h):
    x,y,bw,bh=json.loads(box,parse_float=F,parse_int=F)
    x0,y0,x1,y1=max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
    assert x0<x1 and y0<y1
    return x0,y0,x1,y1

def fractions(box,t):
    x0,y0,x1,y1=box
    return max(0,min(t,x1)-x0)*(y1-y0)/((x1-x0)*(y1-y0))

def group(x,y):
    a,b=F(3,4),F(1,4)
    if (x>=a and y<=b) or (x<=b and y>=a):return "split"
    if (x>=a and y>=a) or (x<=b and y<=b):return "same"
    return "partial"

def noun_prompt(noun):return "A photo of the "+("TV" if noun=="tv" else noun)+"."

def freeze():
    old=json.loads((SOURCE/"freeze.json").read_text())
    nf=json.loads((NOUN/"freeze.json").read_text())
    assert all(digest(SOURCE/n)==old["hashes"][n] for n in ("images.csv","pairs.csv"))
    assert digest(NOUN/"mappings.csv")==nf["mappings"] and nf["pairs"]==old["hashes"]["pairs.csv"]
    mapping={r["pair_id"]:r for r in rows(NOUN/"mappings.csv") if r["seed"]=="42"}
    pairs={r["pair_id"]:r for r in rows(SOURCE/"pairs.csv")}
    images=rows(SOURCE/"images.csv")
    assert set(pairs)==set(mapping) and len(pairs)==64 and len(images)==len({r["image_id"] for r in images})==128
    detail=[];photos=[]; smoke=[];diff=[]
    for im in images:
        pid,side=im["pair_id"],im["side"];p=pairs[pid];m=mapping[pid]
        assert m["donor"] and p["x"]==m["x"] and p["y"]==m["y"] and p["image_"+side]==im["image_id"]
        w,h=int(im["width"]),int(im["height"]);ox,oy,dw,dh=valid_rect(w,h)
        assert [ox,oy,dw,dh]==json.loads(im["valid_xywh"])
        boxes=[clipped(im[k+"_original_xywh"],w,h) for k in ("x","y")]
        left=boxes[0][2]<=boxes[1][0];assert left or boxes[1][2]<=boxes[0][0]
        assert left==(p["winner_left"]==side) and p["winner_left"]!=p["winner_right"]
        # Cross-check the source-to-input map on all eight original box edges.
        for k,box in zip(("x","y"),boxes):
            projected=[ox+float(box[0])*dw/w,oy+float(box[1])*dh/h,ox+float(box[2])*dw/w,oy+float(box[3])*dh/h]
            assert max(abs(a-b) for a,b in zip(projected,json.loads(im[k+"_input_xyxy"])))<1e-9
        lines={};intlines={};intact=set()
        for pct in LINES:
            t=F(w*pct,100);k=int(t+F(1,2))
            x,y=[fractions(b,t) for b in boxes];u,v=[fractions(b,F(k)) for b in boxes]
            label=group(x,y); lines[pct]=label;intlines[pct]=group(u,v)
            intact.update(j for j,f in enumerate((u,v)) if f in (0,1))
            detail.append(dict(pair_id=pid,side=side,image_id=im["image_id"],line=pct,x_continuous=float(t),
                x_pixel=k,x_left=float(x),y_left=float(y),group=label,x_left_raster=float(u),y_left_raster=float(v),raster_group=intlines[pct]))
            if label!=intlines[pct]:diff.append([pid,side,pct,label,intlines[pct]])
        main={lines[25],lines[75]}=={"same","split"};split=(25 if lines[25]=="split" else 75) if main else 0
        photos.append(dict(pair_id=pid,side=side,image_id=im["image_id"],file=im["file"],jpeg_sha256=im["jpeg_sha256"],
            width=w,height=h,true_relation="left" if left else "right",sign=1 if left else -1,
            **{k:m[k] for k in ROLES},donor=m["donor"],main=int(main),split_line=split,
            same_line=100-split if main else 0,license_id=im["license_id"]))
        if len(smoke)<2 and im["license_id"]=="4" and len(intact)==2:smoke.append(im["image_id"])
    keep=[p for p in photos if p["main"]];counts=Counter(p["split_line"] for p in keep)
    assert len(keep)==64 and len({p["pair_id"] for p in keep})==48 and counts=={25:28,75:36},(len(keep),counts)
    assert len(smoke)==2
    save_rows(HERE/"groups.csv",detail);save_rows(HERE/"photos.csv",photos)
    save_json(HERE/"freeze.json",dict(source_pairs_sha=old["hashes"]["pairs.csv"],source_images_sha=old["hashes"]["images.csv"],
        mapping_sha=nf["mappings"],hashes={n:digest(HERE/n) for n in ("protocol.md","groups.csv","photos.csv","design.py")},
        main_images=len(keep),main_pairs=48,split_counts=dict(counts),raster_group_differences=diff,
        smoke_ids=smoke,gray=GRAY,lines=LINES,bootstrap_seed=42,bootstrap_replicates=10000))
if __name__=="__main__":freeze()
