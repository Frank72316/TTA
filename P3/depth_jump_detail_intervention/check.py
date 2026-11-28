"""CPU-only independent pointwise fusion, CDF-based MAE, and paired bootstrap verification."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from shared import *
from coverage import edge_path
from verify import sample_T

def expected_abs(r,u):
    knots=np.unique(np.r_[r,u])
    f=np.searchsorted(np.sort(r),knots[:-1],side="right")/len(r)
    g=np.searchsorted(np.sort(u),knots[:-1],side="right")/len(u)
    return np.dot(np.diff(knots),f*(1-g)+g*(1-f))

def check_scene(rec):
    basic=ident(rec);year,name=rec["year"],rec["id"]
    b,high=signed_base_and_h(rec);j=get(cache(rec)/"inference.json")
    low=np.load(local_cache(rec)/"low.npy")
    rect=[v["box"] for v in j["layouts"]]
    scales=[r for r in read(RESULTS/"input_windows.csv") if (r["year"],r["id"])==(year,name)]
    assert [int(r["corner"]) for r in scales]==list(range(4))
    for w,(x,y,u,v),s in zip(low,rect,scales):
        ih,iw=w.shape;sy=(np.arange(ih)+.5)/ih;sx=(np.arange(iw)+.5)/iw
        indices=np.ix_((sy>=.1)&(sy<=.9),(sx>=.1)&(sx<=.9))
        a=b[y:v,x:u][indices];p=w[indices];ok=np.isfinite(a)&np.isfinite(p)&(a>0)&(p>0)
        k=np.clip(np.median(a[ok]/p[ok]),.5,2.) if ok.any() else 1.
        assert abs(float(s["k_L"])-k)<1e-7
    with np.load(edge_path(rec)) as f:
        near,far,truth,axis=[f[k] for k in ("near","far","jump","axis")]
    n=len(near)
    positions,inverse=np.unique(np.concatenate((near.ravel(),far.ravel())),return_inverse=True)
    n_idx,f_idx=inverse.reshape(2,n,3)
    def jumps(values):
        return np.sort(values[f_idx],axis=1)[:,1].astype(float)-np.sort(values[n_idx],axis=1)[:,1]
    jb=jumps(b.ravel()[positions]);need=truth-jb
    groups=[]
    for a in np.unique(axis):
        ids=np.where(axis==a)[0];mag=np.abs(jb[ids])*1000
        cuts=np.unique(np.quantile(mag,[.2,.4,.6,.8]))
        for l,h in zip(np.r_[-np.inf,cuts],np.r_[cuts,np.inf]):
            members=ids[(mag>l)&(mag<=h)]
            if len(members):groups.append(members)
    results=[]
    for mode,ks in (("native_R",[float(r["k_H"]) for r in scales]),("low_core",[float(r["k_L"]) for r in scales])):
        row=dict(**basic,scale=mode)
        for label,parts in (("H",high),("L",low)):
            q=jumps(sample_T(positions,b,parts,rect,ks));update=q-jb
            mae=np.abs(q-truth).mean()*1000
            conditional=sum(len(g)*expected_abs(need[g],update[g]) for g in groups)*1000/n
            row.update({label+"_actual_mm":mae,label+"_cond_mm":conditional,label+"_P_mm":conditional-mae})
        row.update(delta_P_mm=row["H_P_mm"]-row["L_P_mm"],G_mm=row["L_actual_mm"]-row["H_actual_mm"])
        results.append(row)
    return results

if __name__=="__main__":
    with ProcessPoolExecutor(4) as pool:rows=[r for scene in pool.map(check_scene,included()) for r in scene]
    ref={(r["year"],r["id"],r["scale"]):r for r in read(RESULTS/"per_scene.csv")}
    comparisons=[]
    for r in rows:
        rr=ref[r["year"],r["id"],r["scale"]]
        comparisons.extend(abs(r[k]-float(rr[k])) for k in r if k.endswith("_mm"))
    assert max(comparisons)<.001
    fields=("delta_P_mm","G_mm")
    report={(r["batch"],r["scale"],r["metric"]):r for r in read(RESULTS/"summary.csv")}
    bootstrap_errors=[];rng=np.random.default_rng(2026)
    for batch,years in COHORTS.items():
        names=sorted({r["cluster"] for r in rows if r["year"] in years});n=len(names)
        draws=np.array([rng.choice(n,n,replace=True) for _ in range(10000)])
        for mode in ("native_R","low_core"):
            vals=np.array([[np.mean([r[k] for r in rows if r["cluster"]==c and r["scale"]==mode]) for k in fields] for c in names])
            estimates=np.vstack([vals.mean(axis=0),np.quantile(vals[draws].mean(axis=1),[.025,.975],axis=0)])
            for i,k in enumerate(fields):
                oldrow=report[batch,mode,k]
                bootstrap_errors.extend(abs(estimates[j,i]-float(oldrow[s])) for j,s in enumerate(("mean","lo","hi")))
    assert max(bootstrap_errors)<.001
    save(RESULTS/"verification.json",dict(independent_scene_values=len(comparisons),max_scene_error_mm=max(comparisons),
        independently_checked_L_scales=200,bootstrap_checks=len(bootstrap_errors),max_paired_stats_error_mm=max(bootstrap_errors)))
