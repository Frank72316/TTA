"""Aligned, common-mask metrics and frozen frame metadata. CPU only."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"depth_adaptive_budget"
VIEWS=("O","L2","R2","L4","R4")

def sha(path):
    d=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):d.update(chunk)
    return d.hexdigest()

def read(path):
    with path.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)

def save(path,obj):
    path.write_bytes((json.dumps(obj,indent=2,ensure_ascii=False)+"\n").encode("utf8"))

def frames():
    source=read(OLD/"frames.csv")
    scenes={r["frame_id"]:r for r in read(OLD/"frame_results.csv")}
    assert len(source)==len({r["frame_id"] for r in source})==len(scenes)==128
    out=[]
    for r in source:
        s=scenes[r["frame_id"]];assert s["frame"]==r["frame"] and s["truth"]==r["truth"]
        out.append(dict(**r,scene=s["scene"]))
    assert len({r["scene"] for r in out})==95
    return out

def step(w,percent):
    return (w*percent+50)//100

def displacement(w,v):
    return 0 if v=="O" else (-1 if v[0]=="L" else 1)*step(w,int(v[1:]))

def align(arr,dx,edge):
    w=arr.shape[1]
    return arr[:,edge+dx:w-edge+dx].copy()

def valid_truth(mm):
    assert np.issubdtype(mm.dtype,np.integer) and mm.ndim==2
    edge=step(mm.shape[1],4)
    g=align(mm,0,edge).astype(np.float64)/1000
    return g,(g>0)&(g<=10)

def fields(p):
    d2=p["L2"].astype(np.float64)-p["R2"].astype(np.float64)
    d4=p["L4"].astype(np.float64)-p["R4"].astype(np.float64)
    return d2,d4,d2*d4/9

def evaluate(p,g,valid):
    assert g.dtype==np.float64 and valid.any()
    for v in VIEWS:
        assert p[v].shape==g.shape and p[v].dtype==np.float32 and np.isfinite(p[v]).all()
    o,l2,r2,l4,r4=[p[v][valid].astype(np.float64) for v in VIEWS]
    gt=g[valid]
    ll=(o+l2+l4)/3;rr=(o+r2+r4)/3
    lr=(o+l2+r4)/3;rl=(o+r2+l4)/3
    moe=lambda fitted:float(np.mean((fitted-gt)**2))
    m_o,m_ll,m_rr,m_lr,m_rl=map(moe,(o,ll,rr,lr,rl))
    same=(m_ll+m_rr)/2;opp=(m_lr+m_rl)/2;gain=same-opp
    _,_,contrib=fields(p);c=contrib[valid]
    pos=np.maximum(c,0).mean();neg=np.minimum(c,0).mean()
    mean_prediction_error=np.max(abs((ll+rr)/2-(lr+rl)/2))
    assert mean_prediction_error<1e-12
    assert abs(gain-c.mean())<1e-12 and abs(pos+neg-gain)<1e-12
    return dict(geometry_pixels=g.size,valid_pixels=int(valid.sum()),MSE_O=m_o,MSE_LL=m_ll,MSE_RR=m_rr,
                MSE_LR=m_lr,MSE_RL=m_rl,MSE_same=same,MSE_opposite=opp,G=gain,O_minus_same=m_o-same,
                O_minus_opposite=m_o-opp,product_mean=float(c.mean()),positive_contribution=float(pos),
                negative_contribution=float(neg),positive_pixels=int((c>0).sum()),negative_pixels=int((c<0).sum()),
                zero_pixels=int((c==0).sum()),identity_abs_error=abs(gain-c.mean()),
                signed_sum_abs_error=abs(pos+neg-gain),mean_prediction_max_abs_error=float(mean_prediction_error))
