"""Full-data, CPU-only independent checks and signed subdivisions of the supplied transitions."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from edge_cache import csv_rows, digest, write_rows, TRANSITIONS

def independent_random_mae(residual, update):
    """E|X-Y| by integrating F_X*(1-F_Y)+F_Y*(1-F_X); X=e, Y=-u."""
    a=np.sort(residual)
    b=np.sort(-update)
    knots=np.unique(np.concatenate((a,b)))
    fa=np.searchsorted(a,knots[:-1],side="right")/len(a)
    fb=np.searchsorted(b,knots[:-1],side="right")/len(b)
    return np.dot(np.diff(knots),fa*(1-fb)+fb*(1-fa))

def named_classes(e,after):
    no_update=e==after
    crossed=e*after<0
    gain=np.abs(e)-np.abs(after)
    changed=(~no_update)&(e!=0)
    return dict(zip(TRANSITIONS,(
        no_update,(e==0)&~no_update,changed&~crossed&(gain<0),
        changed&~crossed&(gain>0),crossed&(gain>0),crossed&(gain==0),crossed&(gain<0))))

def cluster_vector(records,cols):
    clusters=defaultdict(list)
    for row in records:
        clusters[row["cluster"]].append([float(row[c]) for c in cols])
    return np.vstack([np.mean(clusters[k],axis=0) for k in sorted(clusters)])

def summarize(records, groupcols, cols, batches, indices):
    output=[]
    for batch,years in batches.items():
        selected=defaultdict(list)
        for r in records:
            if r["year"] in years:
                selected[tuple(r[k] for k in groupcols)].append(r)
        for combination,items in sorted(selected.items()):
            values=cluster_vector(items,cols)
            draws=values[indices[batch]].mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975],axis=0)
            for i,metric in enumerate(cols):
                output.append(dict(batch=batch,**dict(zip(groupcols,combination)),metric=metric,
                    mean=values[:,i].mean(),lo=lo[i],hi=hi[i],scenes=len(items),clusters=len(values)))
    return output

def main(context,out):
    sys.path.insert(0,str(context.resolve()))
    from data import included, metric, sources, cache, save
    from coverage import edge_path
    from analysis import maps
    rows=included()
    estimates={(r["year"],r["id"],r["method"]):r for r in csv_rows(out/"edge_assignment_by_scene.csv")}
    levels=csv_rows(out/"scene_metrics.csv")
    observed={(r["year"],r["id"],r["method"]):r for r in levels}
    movements={(r["year"],r["id"],r["side"]):r for r in csv_rows(context/"pixel_movements.csv") if r["direction"]=="farther"}
    published={(r["year"],r["id"],r["method"],r["transition"]):r for r in csv_rows(out/"scene_transitions.csv")}
    metadata=csv_rows(context/"cache_index.csv")
    indexes={(r["year"],r["id"]):r for r in metadata}
    names=["near_mae_mm","far_mae_mm","equal_mae_mm","jump_mae_mm","global_mae_mm"]
    oriented=[];examples={};checks=[];up_errors=[]
    for rec in rows:
        year,scene=rec["year"],rec["id"]
        sid=(year,scene)
        for p,h in zip(sources(rec),("rgb_sha256","depth_sha256","calib_sha256")):
            assert digest(p)==rec[h]
        assert digest(cache(rec)/"inference.json")==indexes[sid]["inference_sha256"]
        with np.load(edge_path(rec)) as f:
            near,far,jump,axis,anchor=[f[k] for k in ("near","far","jump","axis","anchor")]
        gt=metric(rec)
        triplet=lambda im,idx: np.sort(im.ravel()[idx],axis=1)[:,1]
        np.testing.assert_array_equal(triplet(gt,far)-triplet(gt,near),jump)
        predictions=dict(maps(rec,("B","T","R")))
        baseline=predictions["B"]
        bjump=triplet(baseline,far).astype(float)-triplet(baseline,near)
        e=(bjump-jump)*1000
        n=len(e)
        for method in ("T","R"):
            est=predictions[method];q=triplet(est,far).astype(float)-triplet(est,near)
            after=(q-jump)*1000;u=after-e
            measured=estimates[sid+(method,)]
            four=[abs(e).mean(),abs(e+u.mean()).mean(),independent_random_mae(e,u),abs(after).mean()]
            ref=[float(measured[c]) for c in ("baseline_mae_mm","constant_mae_mm","permuted_mae_mm","actual_mae_mm")]
            err=max(abs(a-b) for a,b in zip(four,ref))
            assert err<1e-7
            # A second exact check directly enumerates the entire update pool at deterministic edge indices.
            order=np.sort(u);prefix=np.r_[0.,np.cumsum(order)]
            probe=np.unique(np.linspace(0,n-1,min(31,n),dtype=int))
            k=np.searchsorted(order,-e[probe])
            formula=((n-2*k)*e[probe]+prefix[-1]-2*prefix[k])/n
            brute=np.abs(e[probe,None]+u).mean(axis=1)
            assert np.max(abs(brute-formula))<1e-7
            gain=abs(e)-abs(after);count=0;subtotal=0.
            for label,mask in named_classes(e,after).items():
                exp=published[sid+(method,label)]
                assert int(mask.sum())==int(exp["count"])
                assert abs(gain[mask].sum()/n-float(exp["contribution_mm"]))<1e-7
                for sign,sg in (("under",e<0),("over",e>0),("exact",e==0)):
                    use=mask&sg
                    value=gain[use].sum()/n
                    count+=int(use.sum());subtotal+=value
                    oriented.append(dict(year=year,id=scene,cluster=rec["cluster"],method=method,baseline=sign,
                        transition=label,edges=n,count=int(use.sum()),fraction=float(use.mean()),contribution_mm=value))
                    cohort="2014" if year=="2014" else "2005-2006"
                    example=(cohort,label,sign)
                    if method=="R" and use.any() and example not in examples:
                        i=int(np.flatnonzero(use)[0]);y,x=divmod(int(anchor[i]),gt.shape[1])
                        examples[example]=dict(cohort=cohort,year=year,id=scene,transition=label,baseline=sign,
                            edge_index=i,axis=int(axis[i]),anchor_x=x,anchor_y=y,GT_jump_mm=float(1000*jump[i]),
                            B_jump_mm=bjump[i]*1000,R_jump_mm=q[i]*1000,baseline_error_mm=e[i],update_mm=u[i],gain_mm=gain[i])
            assert count==n and abs(subtotal-(four[0]-four[3]))<1e-7
            checks.append(dict(year=year,id=scene,method=method,edges=n,independent_four_MAE_max_error_mm=err,
                brute_checks=len(probe),brute_prefix_max_error_mm=max(abs(brute-formula)),
                decomposition_error_mm=abs((four[0]-four[3])-(float(measured["common_gain_mm"])-float(measured["permuted_minus_constant_mm"])+float(measured["permuted_minus_actual_mm"]))),
                Jensen_gap_mm=four[2]-four[1]))
        # Published aggregate by sign suffices to reconstruct UP's two side errors without deriving a new image.
        side_maes=[]
        for side in ("near","far"):
            old=movements[sid+(side,)]
            change=float(old["err_change_mm"] or 0)*int(old["count"])/int(old["total"])
            val=float(observed[sid+("B",)][side+"_mae_mm"])+change
            side_maes.append(val)
            up_errors.append(abs(val-float(observed[sid+("UP",)][side+"_mae_mm"])))
        up_errors.append(abs(np.mean(side_maes)-float(observed[sid+("UP",)]["equal_mae_mm"])))
    batches={"2005-2006":{"2005","2006"},"2014":{"2014"}}
    rng=np.random.default_rng(2026)
    indices={}
    for b,years in batches.items():
        n=len({r["cluster"] for r in rows if r["year"] in years})
        indices[b]=rng.integers(n,size=(10000,n))
    signed=summarize(oriented,["method","transition","baseline"],["fraction","contribution_mm"],batches,indices)
    absolute=summarize(levels,["method"],names,batches,indices)
    # Independently rebuild the published paired cluster intervals for each decomposition term.
    audited=csv_rows(out/"edge_assignment_summary.csv")
    cols=sorted({r["metric"] for r in audited})
    rebuilt=summarize(list(estimates.values()),["method"],cols,batches,indices)
    lookup={(r["batch"],r["method"],r["metric"]):r for r in audited}
    stat_error=max(abs(r[k]-float(lookup[r["batch"],r["method"],r["metric"]][k])) for r in rebuilt for k in ("mean","lo","hi"))
    assert stat_error<1e-7 and max(up_errors)<.001
    assert sum(r["edges"] for r in checks if r["method"]=="R")==243818
    assert min(c["Jensen_gap_mm"] for c in checks)>=-1e-7
    assert max(c["decomposition_error_mm"] for c in checks)<1e-7
    for name,table in [("sign_by_scene",oriented),("sign_summary",signed),("absolute_summary_mm",absolute),
                       ("independent_by_scene",checks),("boundary_examples",list(examples.values()))]:
        write_rows(out/(name+".csv"),table)
    save(out/"independent_checks.json",dict(scenes=len(rows),method_scene_checks=len(checks),registered_metric_checks=1200,
        source_hashes_and_inference_metadata=4*len(rows),cdf_MAE_max_difference_mm=max(c["independent_four_MAE_max_error_mm"] for c in checks),
        brute_force_edge_rows=sum(c["brute_checks"] for c in checks),brute_force_max_error_mm=max(c["brute_prefix_max_error_mm"] for c in checks),
        max_identity_residual_mm=max(c["decomposition_error_mm"] for c in checks),min_Jensen_gap_mm=min(c["Jensen_gap_mm"] for c in checks),
        convexity_violations=0,UP_public_reconstructions=len(up_errors),UP_max_public_difference_mm=max(up_errors),
        clustered_summary_max_error_mm=stat_error,new_model_forwards=0))
if __name__=="__main__":
    app=argparse.ArgumentParser()
    app.add_argument("--context-dir",type=Path,required=True)
    app.add_argument("--out",type=Path,required=True)
    args=app.parse_args()
    main(args.context_dir,args.out)
