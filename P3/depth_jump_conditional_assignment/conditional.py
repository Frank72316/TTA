"""Exact within-bin assignment and the within-axis, bin-size-preserving label control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import time
from common import *

QUANTILES=[.2,.4,.6,.8]
REPEATS=100
FIELDS=["all_random_mm","axis_random_mm","conditional_mm","actual_mm","total_mm","axis_gain_mm","jump_bin_gain_mm",
        "group_gain_mm","residual_mm","size_null_mm","null_gain_vs_axis_mm","size_null_minus_conditional_mm",
        "analytic_size_null_mm","analytic_size_null_minus_conditional_mm"]

def independent_mae(e,u):
    """Symmetric empirical CDF integral used only as a check on the prefix-sum algorithm."""
    x=np.sort(e);y=np.sort(-u)
    knots=np.unique(np.r_[x,y])
    a=x.searchsorted(knots[:-1],side="right")/len(x)
    b=y.searchsorted(knots[:-1],side="right")/len(y)
    return float(np.dot(np.diff(knots),a*(1-b)+b*(1-a)))

def checked_mae(e,u,errors):
    value=random_mae(e,u)
    errors.append(abs(value-independent_mae(e,u)))
    return value

def analyze_scene(rec):
    ident={k:rec[k] for k in ("year","id","cluster")}
    path=array_path(rec);assert digest(path)==rec["array_sha256"]
    with np.load(path) as f:
        axis=f["axis"];mag=f["abs_B_jump_mm"];e=f["e"];updates=[f[k] for k in METHODS]
    n=len(e)
    errors=[];direction_loss=np.zeros(2);conditional_loss=np.zeros(2);analytic_null=np.zeros(2)
    null_draws=np.zeros((REPEATS,2));bin_rows=[];axis_rows=[]
    for direction in (0,1):
        ix=np.flatnonzero(axis==direction);h=len(ix)
        if not h:
            axis_rows.append(dict(**ident,axis=direction,n=0,nominal_bins=5,distinct_cutpoints=0,populated=0,singletons=0,
                singleton_edge_fraction=0.,cutpoints_mm="",raw_quintiles_mm=""))
            continue
        # The label depends only on B and axis. Values on a cut stay in the lower bin.
        x=mag[ix]
        q=np.quantile(x,QUANTILES,method="linear");cuts=np.unique(q)
        labels=np.searchsorted(cuts,x,side="left")
        sizes=np.bincount(labels,minlength=len(cuts)+1)
        order=np.argsort(x,kind="stable")
        assert np.all(np.diff(labels[order])>=0)
        assert not np.any((np.diff(x[order])==0)&(np.diff(labels[order])!=0))
        k=np.count_nonzero(sizes)
        axis_rows.append(dict(**ident,axis=direction,n=h,nominal_bins=5,distinct_cutpoints=len(cuts),populated=k,
            singletons=int(np.sum(sizes==1)),singleton_edge_fraction=float(np.sum(sizes==1)/h),
            cutpoints_mm=" ".join(map(repr,cuts.tolist())),raw_quintiles_mm=" ".join(map(repr,q.tolist()))))
        sube=e[ix];subu=[u[ix] for u in updates]
        axis_mae=np.array([checked_mae(sube,u,errors) for u in subu])
        direction_loss+=h/n*axis_mae
        actual=np.array([np.abs(sube+u).mean() for u in subu])
        analytic_null+=h/n*(axis_mae+((k-1)/(h-1)*(actual-axis_mae) if h>1 else 0))
        for label,size in enumerate(sizes):
            keep=labels==label
            row=dict(**ident,axis=direction,bin=label+1,count=size,axis_edges=h,scene_edges=n,
                axis_fraction=size/h,scene_fraction=size/n,singleton=int(size==1),
                lower_open_mm=cuts[label-1] if label else "",upper_closed_mm=cuts[label] if label<len(cuts) else "",
                observed_min_mm=x[keep].min() if size else "",observed_max_mm=x[keep].max() if size else "")
            bin_rows.append(row)
            if size:
                local=[checked_mae(sube[keep],u[keep],errors) for u in subu]
                conditional_loss+=size/n*np.array(local)
        rng=np.random.default_rng(np.random.SeedSequence([SEED,int(rec["seq"]),direction]))
        for rep in range(REPEATS):
            perm=rng.permutation(labels)
            assert np.array_equal(np.bincount(perm,minlength=len(sizes)),sizes)
            for label in np.flatnonzero(sizes):
                ii=np.flatnonzero(perm==label)
                # R and T use exactly the same shuffled group on this replicate.
                null_draws[rep]+=len(ii)/n*np.array([random_mae(sube[ii],u[ii]) for u in subu])
    assert sum(int(r["count"]) for r in bin_rows)==n
    scenes=[];replicates=[]
    for mi,(method,u) in enumerate(zip(METHODS,updates)):
        full=checked_mae(e,u,errors);actual=float(np.mean(np.abs(e+u)))
        direct=direction_loss[mi];cond=conditional_loss[mi];null=null_draws[:,mi].mean()
        axis_gain=full-direct;jump_gain=direct-cond;residual=cond-actual;total=full-actual
        balance=abs(total-(axis_gain+jump_gain+residual));assert balance<1e-8
        scenes.append(dict(**ident,method=method,n_edges=n,all_random_mm=full,axis_random_mm=direct,
            conditional_mm=cond,actual_mm=actual,total_mm=total,axis_gain_mm=axis_gain,jump_bin_gain_mm=jump_gain,
            group_gain_mm=full-cond,residual_mm=residual,size_null_mm=null,null_gain_vs_axis_mm=direct-null,
            size_null_minus_conditional_mm=null-cond,analytic_size_null_mm=analytic_null[mi],
            analytic_size_null_minus_conditional_mm=analytic_null[mi]-cond,control_mc_se_mm=null_draws[:,mi].std(ddof=1)/np.sqrt(REPEATS),
            max_cdf_check_mm=max(errors),decomposition_residual_mm=balance))
        for rep,loss in enumerate(null_draws[:,mi]):
            replicates.append(dict(**ident,method=method,replicate=rep+1,size_null_mm=loss,
                axis_minus_size_null_mm=direct-loss,size_null_minus_conditional_mm=loss-cond))
    assert max(errors)<1e-7
    return scenes,axis_rows,bin_rows,replicates

def main(workers):
    start=time.perf_counter()
    records=read(OUT/"input_index.csv")
    assert len(records)==50 and sum(int(r["n_edges"]) for r in records)==243818
    with ProcessPoolExecutor(workers) as pool:
        results=list(pool.map(analyze_scene,records))
    scenes,axes,bins,trials=([r for res in results for r in res[col]] for col in range(4))
    replay=read(PREVIOUS/"edge_assignment_by_scene.csv")
    prev={(r["year"],r["id"],r["method"]):r for r in replay}
    for r in scenes:
        ref=prev[r["year"],r["id"],r["method"]]
        assert max(abs(r[x]-float(ref[y])) for x,y in (("all_random_mm","permuted_mae_mm"),("actual_mm","actual_mae_mm"),("total_mm","permuted_minus_actual_mm")))<1e-7
    summary=cluster_stats(scenes,FIELDS)
    cover=[]
    for batch,years in COHORTS.items():
        aa=[r for r in axes if r["year"] in years]
        bb=[r for r in bins if r["year"] in years]
        total=sum(r["n"] for r in aa);singles=sum(r["count"]==1 for r in bb)
        positive=[int(r["count"]) for r in bb if r["count"]]
        cover.append(dict(batch=batch,scenes=len(aa)//2,clusters=len({r["cluster"] for r in aa}),edges=total,
            horizontal=sum(r["n"] for r in aa if r["axis"]==0),vertical=sum(r["n"] for r in aa if r["axis"]==1),
            axes_nonempty=sum(r["n"]>0 for r in aa),axes_with_duplicate_cuts=sum(r["distinct_cutpoints"]<4 for r in aa if r["n"]),
            occupied_bins=len(positive),empty_after_cut_dedup=sum(r["count"]==0 for r in bb),min_occupied_size=min(positive),
            median_occupied_size=np.median(positive),max_occupied_size=max(positive),singleton_bins=singles,
            singleton_edges=singles,singleton_edge_fraction=singles/total))
    for name,data in (("per_scene",scenes),("axes",axes),("bins",bins),("size_control_repeats",trials),
                      ("summary",summary),("coverage",cover)):
        write(OUT/(name+".csv"),data)
    save_json(OUT/"assignment_check.json",dict(scenes=len(records),edges=sum(r["edges"] for r in cover),
        worker_processes=workers,elapsed_s=time.perf_counter()-start,new_model_forwards=0,repetitions=REPEATS,scheme="PCG64 SeedSequence([2026, frozen_scene_index, axis])",
        max_prefix_vs_independent_CDF_difference_mm=max(r["max_cdf_check_mm"] for r in scenes),
        max_decomposition_residual_mm=max(r["decomposition_residual_mm"] for r in scenes),
        scene_method_comparisons=len(scenes),new_vs_previous_A_checks=2*len(scenes),all_shuffled_bin_counts_verified=True))
    for r in summary:
        if r["metric"] in ("total_mm","axis_gain_mm","jump_bin_gain_mm","residual_mm","size_null_minus_conditional_mm","null_gain_vs_axis_mm"):
            print(r["batch"],r["method"],r["metric"],*[round(r[k],4) for k in ("mean","lo","hi")])
    print(cover)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=4)
    main(p.parse_args().workers)
