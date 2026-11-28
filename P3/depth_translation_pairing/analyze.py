"""Summarize paired frame errors and cluster bootstrap with equal frame weights."""
import argparse
from pathlib import Path
import numpy as np
from metrics import HERE,VIEWS,frames,read,write,save

def analyze(out):
    records=read(out/"frame_results.csv")
    frame_list=frames()
    assert [r["frame_id"] for r in records]==[r["frame_id"] for r in frame_list]
    assert all(r["scene"]==s["scene"] and r["truth"]==s["truth"] for r,s in zip(records,frame_list))
    data=lambda k:np.array([float(r[k]) for r in records],dtype=np.float64)
    same,opp,o=data("MSE_same"),data("MSE_opposite"),data("MSE_O")
    assert np.array_equal(same,(data("MSE_LL")+data("MSE_RR"))/2)
    assert np.array_equal(opp,(data("MSE_LR")+data("MSE_RL"))/2)
    targets={"G":same-opp,"O_minus_same":o-same,"O_minus_opposite":o-opp}
    labels=np.array([r["scene"] for r in records]);scenes=sorted(set(labels))
    assert len(scenes)==95 and len(records)==128
    groups=[np.flatnonzero(labels==s) for s in scenes]
    sizes=np.array([len(g) for g in groups])
    rng=np.random.default_rng(20261128);draws=rng.integers(len(groups),size=(10000,len(groups)))
    denominators=sizes[draws].sum(axis=1)
    table=[]
    for key,vector in [("MSE_O",o),("MSE_same",same),("MSE_opposite",opp),*targets.items()]:
        assert np.array_equal(vector,data(key))
        r=dict(metric=key,unit="m2",frame_mean=float(vector.mean()),ci_low="",ci_high="",better="",worse="",tie="")
        if key in targets:
            sums=np.array([vector[idx].sum() for idx in groups])
            ci=np.quantile(sums[draws].sum(axis=1)/denominators,[.025,.975])
            r.update(ci_low=ci[0],ci_high=ci[1],better=int((vector>0).sum()),worse=int((vector<0).sum()),tie=int((vector==0).sum()))
        table.append(r)
    for k in ("positive_contribution","negative_contribution","product_mean"):
        table.append(dict(metric=k,unit="m2",frame_mean=float(data(k).mean()),ci_low="",ci_high="",better="",worse="",tie=""))
    assert np.allclose(data("G"),data("product_mean"),atol=1e-12,rtol=0)
    assert np.allclose(data("G"),data("positive_contribution")+data("negative_contribution"),atol=1e-12,rtol=0)
    assert np.array_equal(data("valid_pixels"),data("positive_pixels")+data("negative_pixels")+data("zero_pixels"))
    view_audit=read(out/"view_audit.csv")
    assert len(view_audit)==len({(r["frame_id"],r["view"]) for r in view_audit})==640
    assert {(r["frame_id"],v) for r in records for v in VIEWS}=={(r["frame_id"],r["view"]) for r in view_audit}
    assert all(r["origin"]==("verified_old_O" if r["view"]=="O" else "new_translation") for r in view_audit)
    proof=dict(frames=len(records),scenes=len(scenes),bootstrap=10000,seed=20261128,
               max_identity_abs_error=float(data("identity_abs_error").max()),
               max_signed_sum_abs_error=float(data("signed_sum_abs_error").max()),
               max_mean_prediction_abs_error=float(data("mean_prediction_max_abs_error").max()),
               geometry_pixels=int(data("geometry_pixels").sum()),valid_pixels=int(data("valid_pixels").sum()),
               extreme_frames={k:min(records,key=lambda r:(sign*float(r["G"]),r["frame_id"]))["frame_id"]
                               for k,sign in (("max",-1),("min",1))})
    write(out/"summary.csv",table);save(out/"validation.json",proof)
    print(table);print(proof)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE);analyze(p.parse_args().out)
