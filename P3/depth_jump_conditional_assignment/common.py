"""Numerics shared by the cache reader and the conditional assignment analysis."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
CACHE=Path(os.environ.get("EDGE_CACHE","/home/data/users/shenc/tta_edge_conditional_assignment_v1"))
OUT=ROOT/"results"
PREVIOUS=ROOT.parent/"depth_jump_assignment"/"results"
COHORTS={"2005-2006":("2005","2006"),"2014":("2014",)}
METHODS=("R","T")
SEED=2026

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def save_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<22),b""):
            h.update(block)
    return h.hexdigest()

def random_mae(e,u):
    """Each receiver sees the complete correction pool, including itself."""
    n=len(u)
    ordered=np.sort(u)
    prefix=np.r_[0.,np.cumsum(ordered,dtype=np.float64)]
    k=ordered.searchsorted(-e,side="left")
    return float(np.mean(((n-2*k)*e+prefix[-1]-2*prefix[k])/n))

def cluster_stats(rows,fields):
    """Two cohorts share one RNG as in the pinned analysis; cluster order is alphabetical."""
    rng=np.random.default_rng(SEED)
    results=[]
    for batch,years in COHORTS.items():
        subset=[r for r in rows if r["year"] in years]
        names=sorted({r["cluster"] for r in subset})
        n=len(names)
        pick=rng.integers(n,size=(10000,n))
        for method in METHODS:
            method_rows=[r for r in subset if r["method"]==method]
            means=np.stack([np.mean([[float(r[k]) for k in fields] for r in method_rows if r["cluster"]==name],axis=0) for name in names])
            lo,hi=np.percentile(means[pick].mean(axis=1),[2.5,97.5],axis=0)
            for i,k in enumerate(fields):
                results.append(dict(batch=batch,scenes=len(method_rows),clusters=n,method=method,metric=k,
                    mean=means[:,i].mean(),lo=lo[i],hi=hi[i]))
    return results

def array_path(rec):
    return CACHE/rec["year"]/(rec["id"]+".npz")
