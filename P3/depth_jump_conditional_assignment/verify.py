"""Independent full-sample binning, direct pair enumeration and cluster accounting."""
import time
from collections import defaultdict
from common import *

def checked_loss(e,u):
    # Distances to e targets, integrate using sorted -e instead of sorting supplier updates.
    target=np.sort(-e)
    sums=np.r_[0.,np.cumsum(target)]
    k=target.searchsorted(u,side="right")
    losses=(2*k-len(u))*u+sums[-1]-2*sums[k]
    avg=losses.mean()/len(e)
    step=np.unique(np.linspace(0,len(e)-1,min(19,len(e)),dtype=int))
    brute=np.abs(u[step,None]+e[None,:]).mean(axis=1)
    np.testing.assert_allclose(losses[step]/len(e),brute,atol=1e-8,rtol=0)
    return avg,len(step)

def main():
    start=time.perf_counter()
    cols=read(OUT/"per_scene.csv")
    values={(r["year"],r["id"],r["method"]):r for r in cols}
    published=read(OUT/"summary.csv");lookup={(r["batch"],r["method"],r["metric"]):r for r in published}
    bins={(r["year"],r["id"],int(r["axis"]),int(r["bin"])):int(r["count"]) for r in read(OUT/"bins.csv")}
    seen=set()
    repeats=defaultdict(list);all_checks=[];n_brute=0
    for r in read(OUT/"size_control_repeats.csv"):
        repeats[r["year"],r["id"],r["method"]].append(r)
    for rec in read(OUT/"input_index.csv"):
        sid=rec["year"],rec["id"]
        with np.load(array_path(rec)) as f:
            e=f["e"];direction=f["axis"];mag=f["abs_B_jump_mm"];updates={m:f[m] for m in METHODS}
        counts=0
        losses={m:0. for m in METHODS}
        for axis in np.unique(direction):
            ii=np.flatnonzero(direction==axis);x=mag[ii]
            cuts=np.unique(np.percentile(x,(20,40,60,80),method="linear"))
            # Direct interval inequalities implement (-infinity,q20], ... , (q80,infinity).
            bounds=np.r_[-np.inf,cuts,np.inf]
            for j,(low,high) in enumerate(zip(bounds,bounds[1:]),start=1):
                take=ii[(x>low)&(x<=high)]
                bkey=(*sid,axis,j)
                assert len(take)==bins[bkey]
                seen.add(bkey);counts+=len(take)
                if len(take):
                    for m,u in updates.items():
                        loss,num=checked_loss(e[take],u[take])
                        losses[m]+=len(take)*loss/len(e);n_brute+=num
        assert counts==len(e)
        for m in METHODS:
            ref=values[(*sid,m)]
            actual=np.abs(e+updates[m]).mean()
            all_checks.extend([abs(losses[m]-float(ref["conditional_mm"])),
                               abs(losses[m]-actual-float(ref["residual_mm"]))])
            rr=repeats[(*sid,m)]
            assert sorted(int(x["replicate"]) for x in rr)==list(range(1,101))
            avg=np.mean([float(x["size_null_mm"]) for x in rr])
            all_checks.append(abs(avg-float(ref["size_null_mm"])))
    assert seen==set(bins) and max(all_checks)<1e-8
    shuffle_by_cohort=[];gen=np.random.default_rng(2026);stat_err=0.
    fields=("total_mm","axis_gain_mm","jump_bin_gain_mm","residual_mm","size_null_minus_conditional_mm")
    for batch,years in COHORTS.items():
        rs=[r for r in cols if r["year"] in years]
        cids=sorted({r["cluster"] for r in rs})
        sampled=np.array([gen.choice(len(cids),size=len(cids)) for _ in range(10000)])
        for m in METHODS:
            by_cluster=[[r for r in rs if r["cluster"]==c and r["method"]==m] for c in cids]
            cluster_means=np.array([np.array([[float(r[k]) for k in fields] for r in group]).mean(axis=0) for group in by_cluster])
            stats=np.vstack([cluster_means.mean(0),np.quantile(cluster_means[sampled].mean(axis=1),[.025,.975],axis=0)])
            for i,k in enumerate(fields):
                ref=lookup[batch,m,k]
                stat_err=max(stat_err,max(abs(stats[j,i]-float(ref[stat])) for j,stat in enumerate(("mean","lo","hi"))))
            # Variation over 100 relabelings, conditional on these fixed scenes (not a scene-level CI).
            repeated=[]
            for group in by_cluster:
                diff=[[float(x["axis_minus_size_null_mm"]) for x in sorted(repeats[r["year"],r["id"],m],key=lambda x:int(x["replicate"]))] for r in group]
                repeated.append(np.mean(diff,axis=0))
            scores=np.mean(repeated,axis=0);real=float(lookup[batch,m,"jump_bin_gain_mm"]["mean"])
            expected=float(lookup[batch,m,"axis_random_mm"]["mean"])-float(lookup[batch,m,"analytic_size_null_mm"]["mean"])
            shuffle_by_cohort.append(dict(batch=batch,method=m,replicates=len(scores),real_bin_gain_mm=real,
                random_bin_gain_mm=scores.mean(),random_gain_analytic_mm=expected,monte_carlo_se_mm=scores.std(ddof=1)/10,
                random_gain_p025_mm=np.quantile(scores,.025),random_gain_p975_mm=np.quantile(scores,.975),
                random_reps_ge_observed=int(np.sum(scores>=real)),cluster_pairing="fixed for these 100 relabelings"))
    assert stat_err<1e-8
    write(OUT/"size_control_summary.csv",shuffle_by_cohort)
    save_json(OUT/"verification.json",dict(bin_rows_rebuilt=len(seen),conditional_and_primary_recalculations=2*len(values),
        direct_rows_against_all_within_bin_suppliers_or_receivers=n_brute,max_bin_replay_or_control_mean_difference_mm=max(all_checks),
        bootstrap_means_and_limits_checked=3*len(fields)*len(METHODS)*len(COHORTS),max_bootstrap_difference_mm=stat_err,
        scene_method_relabels_verified=len(values)*100,seconds=time.perf_counter()-start))
if __name__=="__main__":main()
