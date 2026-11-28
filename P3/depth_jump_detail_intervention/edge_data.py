"""Depth-map fusion and the exact frozen edge strata."""
import numpy as np
from shared import *
from coverage import edge_path
from evaluate import fuse

def jump(depth,near,far):
    p=depth.ravel()
    return np.median(p[far],axis=1).astype(np.float64)-np.median(p[near],axis=1).astype(np.float64)

def predict_map(b,windows,boxes,factors):
    p=(b+fuse(b,windows,boxes,factors))*.5
    assert np.isfinite(p).all()
    return p

def pinned_fields(record):
    year,name=record["year"],record["id"]
    def rows(path):
        return [r for r in read(path) if r["year"]==year and r["id"]==name]
    idx=rows(PREVIOUS/"results/input_index.csv")[0]
    info=get(cache(record)/"inference.json")
    ep=edge_path(record)
    assert sha(ep)==idx["edge_sha256"]
    assert sha(cache(record)/"predictions.npz")==idx["prediction_sha256"]
    if year=="2014":
        prior=next(r for r in read(CONTEXT.parent/"depth_occlusion_tta/cache_index.csv") if r["id"]==name)
        assert sha(old.work(legacy_row(record))/"predictions.npz")==prior["raw_prediction_sha"]
    arr=array_path(record)
    assert sha(arr)==idx["array_sha256"]
    with np.load(ep) as f:
        near,far,truth,axis=(f[k] for k in ("near","far","jump","axis"))
    with np.load(arr) as f:
        assert np.array_equal(f["axis"],axis)
        old_mag,old_e,old_u=(f[k] for k in ("abs_B_jump_mm","e","R"))
    b,native=signed_base_and_h(record)
    bjump=jump(b,near,far)
    mag=np.abs(bjump)*1000
    np.testing.assert_array_equal(mag,old_mag)
    np.testing.assert_array_equal((bjump-truth)*1000,old_e)
    groups=[]
    bin_sizes={(int(r["axis"]),int(r["bin"])):int(r["count"]) for r in rows(PREVIOUS/"results/bins.csv")}
    for entry in rows(PREVIOUS/"results/axes.csv"):
        a=int(entry["axis"]);ii=np.flatnonzero(axis==a)
        assert len(ii)==int(entry["n"])
        if not len(ii):continue
        cuts=np.fromstring(entry["cutpoints_mm"],sep=" ")
        labels=cuts.searchsorted(mag[ii],side="left")
        for label in range(len(cuts)+1):
            take=ii[labels==label]
            assert len(take)==bin_sizes[a,label+1]
            if len(take):groups.append((a,label+1,take))
    np.testing.assert_array_equal(np.sort(np.concatenate([g[2] for g in groups])),np.arange(len(axis)))
    return b,native,info,near,far,truth.astype("f8"),bjump,old_u,groups

def mae_pair(required,update,groups):
    e=-required*1000
    u=update*1000
    actual=np.abs(e+u).mean()
    shuffled=sum((len(ii)/len(u))*random_mae(e[ii],u[ii]) for _,_,ii in groups)
    return actual,shuffled,shuffled-actual

def intervals(items,fields):
    rng=np.random.default_rng(2026)
    result=[]
    for batch,years in COHORTS.items():
        selected=[r for r in items if r["year"] in years]
        clusters=sorted({r["cluster"] for r in selected})
        take=rng.integers(len(clusters),size=(10000,len(clusters)))
        for scale in sorted({r["scale"] for r in selected}):
            scene=[r for r in selected if r["scale"]==scale]
            vals=np.stack([np.mean([[float(r[k]) for k in fields] for r in scene if r["cluster"]==c],axis=0) for c in clusters])
            lo,hi=np.quantile(vals[take].mean(axis=1),[.025,.975],axis=0)
            for i,k in enumerate(fields):
                result.append(dict(batch=batch,scale=scale,metric=k,mean=vals[:,i].mean(),lo=lo[i],hi=hi[i],
                    scenes=len(scene),clusters=len(clusters),positive=sum(float(r[k])>0 for r in scene),
                    negative=sum(float(r[k])<0 for r in scene),zero=sum(float(r[k])==0 for r in scene)))
    return result
