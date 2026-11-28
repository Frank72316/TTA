"""Read the pinned maps on CPU, verify input identity and replay the unconditional experiment."""
import platform
import sys
import time
from datetime import datetime,timezone

from common import *

CTX=ROOT.parent/"depth_occlusion_context"
sys.path.insert(0,str(CTX))
from analysis import maps
from coverage import edge_path
from data import cache,included,legacy_row,old

def predicted_jump(view,near,far):
    p=view.ravel()
    return np.median(p[far],axis=1).astype(np.float64)-np.median(p[near],axis=1).astype(np.float64)

def run():
    started=time.perf_counter()
    start_utc=datetime.now(timezone.utc).isoformat()
    for line in (ROOT/"source_sha256.txt").read_text().splitlines():
        h,p=line.split()
        assert digest(ROOT/p)==h,p
    records=included()
    index={(r["year"],r["id"]):r for r in read(CTX/"cache_index.csv")}
    old_idx={r["id"]:r for r in read(CTX.parent/"depth_occlusion_tta"/"cache_index.csv")}
    previous={(r["year"],r["id"],r["method"]):r for r in read(PREVIOUS/"edge_assignment_by_scene.csv")}
    rows=[];replay=[];max_diff=0.
    for seq,r in enumerate(records):
        ident={k:r[k] for k in ("year","id","cluster")}
        idx=index[r["year"],r["id"]]
        edges=edge_path(r);pred=cache(r)/"predictions.npz"
        assert digest(edges)==idx["edge_sha256"] and digest(pred)==idx["prediction_sha256"]
        assert digest(cache(r)/"inference.json")==idx["inference_sha256"]
        if r["year"]=="2014":
            orig=old.work(legacy_row(r))/"predictions.npz"
            assert digest(orig)==old_idx[r["id"]]["raw_prediction_sha"]
        with np.load(edges) as f:
            near,far,jump,axis=(f[k] for k in ("near","far","jump","axis"))
        assert near.shape==far.shape==(len(axis),3) and np.isin(axis,(0,1)).all()
        arrays=dict(axis=axis)
        for method,view in maps(r,("B","R","T")):
            q=predicted_jump(view,near,far)
            if method=="B":
                base=q
                arrays["abs_B_jump_mm"]=abs(q)*1000.
                arrays["e"]=(q-jump)*1000.
                continue
            e=arrays["e"];u=(q-base)*1000.
            arrays[method]=u
            assert np.isfinite(e).all() and np.isfinite(u).all()
            all_random=random_mae(e,u)
            actual=np.abs(e+u).mean()
            prev=previous[r["year"],r["id"],method]
            assert int(prev["n_edges"])==len(axis)
            diff=max(abs(all_random-float(prev["permuted_mae_mm"])),abs(actual-float(prev["actual_mae_mm"])))
            assert diff<1e-7
            max_diff=max(diff,max_diff)
            replay.append(dict(**ident,method=method,total_mm=all_random-actual))
        dst=array_path(r);dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.with_suffix(".part").open("wb") as f:
            np.savez(f,**arrays)
        dst.with_suffix(".part").replace(dst)
        rows.append(dict(**ident,seq=seq,n_edges=len(axis),edge_sha256=idx["edge_sha256"],
                   prediction_sha256=idx["prediction_sha256"],array_sha256=digest(dst)))
    assert len(rows)==len({(r["year"],r["id"]) for r in rows})==50
    assert sum(r["n_edges"] for r in rows)==243818
    results=cluster_stats(replay,["total_mm"])
    previous_summary=read(PREVIOUS/"edge_assignment_summary.csv")
    for row in results:
        oldrow=next(x for x in previous_summary if x["batch"]==row["batch"] and x["method"]==row["method"] and x["metric"]=="permuted_minus_actual_mm")
        assert max(abs(row[k]-float(oldrow[k])) for k in ("mean","lo","hi"))<1e-7
        if row["method"]=="R":
            assert round(row["mean"],1)==({"2014":17.3,"2005-2006":4.0}[row["batch"]])
    write(OUT/"input_index.csv",rows);write(OUT/"replay.csv",results)
    save_json(OUT/"replay_check.json",dict(start_utc=start_utc,elapsed_s=time.perf_counter()-started,
        host=platform.node(),python=platform.python_version(),numpy=np.__version__,scene_methods=100,legacy_MAE_comparisons=200,
        max_old_MAE_difference_mm=max_diff,bootstrap_intervals_checked=4,protocol_sha256=digest(ROOT/"protocol.md"),
        pinned_commit="78a419dad2f272da083a0693611d57ef81c614a7",new_model_forwards=0))
    for row in results:print(row,flush=True)

if __name__=="__main__":
    run()
