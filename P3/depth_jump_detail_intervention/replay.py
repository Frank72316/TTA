"""Verify native H and the previous conditional assignment before evaluating L."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from edge_data import *

def one(rec):
    b,h,info,near,far,gt,jb,old_u,groups=pinned_fields(rec)
    boxes=[s["box"] for s in info["layouts"]]
    jh=jump(predict_map(b,h,boxes,info["k"]["R"]),near,far)
    np.testing.assert_array_equal((jh-jb)*1000,old_u)
    actual,conditional,P=mae_pair(gt-jb,jh-jb,groups)
    reference=next(r for r in read(PREVIOUS/"results/per_scene.csv") if r["year"]==rec["year"] and r["id"]==rec["id"] and r["method"]=="R")
    error=max(abs(a-float(reference[k])) for a,k in zip((actual,conditional,P),("actual_mm","conditional_mm","residual_mm")))
    assert error<1e-8
    return dict(**ident(rec),scale="native_R",edges=len(gt),bins=len(groups),H_actual_mm=actual,H_cond_mm=conditional,H_P_mm=P,max_reference_error_mm=error)

if __name__=="__main__":
    load_freeze()
    with ProcessPoolExecutor(4) as pool:rows=list(pool.map(one,included()))
    assert len(rows)==50 and sum(r["edges"] for r in rows)==243818 and sum(r["bins"] for r in rows)==495
    stats=intervals(rows,["H_P_mm"]);ref=read(PREVIOUS/"results/summary.csv")
    for r in stats:
        oldrow=next(q for q in ref if q["batch"]==r["batch"] and q["metric"]=="residual_mm" and q["method"]=="R")
        assert max(abs(r[k]-float(oldrow[k])) for k in ("mean","lo","hi"))<1e-8
        print(r,flush=True)
    write(RESULTS/"native_replay.csv",rows)
    write(RESULTS/"native_replay_summary.csv",stats)
