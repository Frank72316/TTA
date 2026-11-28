"""Independent CPU replay from window means, then resample actual frames by scene."""
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from order_data import HERE,read,save

def verify(out):
    windows={(r["frame_id"],r["window"]):r for r in read(out/"windows.csv")}
    effects=defaultdict(list);scene={};counter=defaultdict(int);worst=0.;directions=0
    recipe=[("O",),("O","L2","L4"),("O","R2","R4"),("O","L2","R4"),("O","R2","L4")]
    pairs=read(out/"region_pairs.csv")
    for r in pairs:
        fid=r["frame_id"];wa=windows[fid,r["A"]];wb=windows[fid,r["B"]]
        mm=lambda w:Fraction(int(w["mm_sum"]),int(w["valid_pixels"]))
        truth=(mm(wa)-mm(wb))/1000
        bounds=(Fraction(1,10),Fraction(3,10),Fraction(1,1))
        layer=next((name for name,cut in zip(("equal","near","small","medium"),(0,*bounds))
                    if (truth==0 if name=="equal" else abs(truth)<cut)),"large")
        assert r["stratum"]==layer and float(truth)==float(r["d"])
        a,b=[np.array([sum(float(w[v]) for v in vs)/len(vs) for vs in recipe],dtype=np.float64) for w in (wa,wb)]
        q=a-b;d=float(truth);err=(q-d)**2;states=np.sign(q)
        for j,k in enumerate(("O","LL","RR","LR","RL")):
            true_state="undefined" if not d else "tie" if states[j]==0 else "correct" if states[j]==np.sign(d) else "wrong"
            assert true_state==r["state_"+k]
            worst=max(worst,abs(q[j]-float(r["q_"+k])),abs(err[j]-float(r["sqerr_"+k])))
        if layer not in ("small","medium","large"):continue
        ok=(states==np.sign(d)).astype(float)
        same,opp=err[1:3].mean(),err[3:5].mean()
        arr=[same-opp,(ok[3:5].mean()-ok[1:3].mean())*100,err[0]-opp,(ok[3:5].mean()-ok[0])*100]
        effects[fid].append(arr);scene[fid]=r["scene"]
        for j,k in ((1,3),(2,4)):
            counter[int(states[j]),int(states[k])]+=1;directions+=1
    keys=list(effects);frame=np.array([np.mean(effects[k],axis=0) for k in keys]);labels=sorted(set(scene.values()))
    groups=[np.array([i for i,k in enumerate(keys) if scene[k]==s]) for s in labels]
    rng=np.random.default_rng(20261128)
    bs=np.array([frame[np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])].mean(axis=0) for _ in range(10000)])
    ci=np.quantile(bs,[.025,.975],axis=0).T
    summary={r["metric"]:r for r in read(out/"summary.csv") if r["scope"]=="main"}
    max_ci_error=0.
    for j,k in enumerate(("G","A","gain_vs_O","acc_vs_O")):
        s=summary[k];ref=[float(s[f]) for f in ("ci_low","ci_high")]
        max_ci_error=max(max_ci_error,float(np.max(abs(ci[j]-ref))))
        assert abs(frame[:,j].mean()-float(s["estimate"]))<1e-12
    assert worst<1e-12 and max_ci_error<1e-12
    save(out/"replay.json",dict(verified_region_pairs=len(pairs),verified_recipe_states=5*len(pairs),
         max_abs_prediction_or_squared_error_difference=worst,main_directional_comparisons=directions,
         main_strict_sign_crossings=counter[-1,1]+counter[1,-1],independent_scene_interval_comparisons=4,
         max_abs_interval_endpoint_difference=max_ci_error))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE);verify(p.parse_args().out)
