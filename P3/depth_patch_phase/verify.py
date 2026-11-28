"""Independent region reconstruction, bootstrap and experiment-accounting checks."""
import itertools
from collections import Counter,defaultdict
from fractions import Fraction
import numpy as np
from study import HERE,SEED,PAIRS,ARMS,key,read,load,save,sha

def main():
    proof=load(HERE/"freeze.json")
    assert sha(HERE/"protocol.md")==proof["protocol_sha256"] and sha(HERE/"frames.csv")==proof["frame_sha256"]
    frames=read(HERE/"frames.csv")
    ids=[f["frame_id"] for f in frames]
    assert len(set(ids))==len({f["scene"] for f in frames})==32
    by=defaultdict(list)
    for w in read(HERE/"windows.csv"):
        by[w["frame_id"]].append(w)
    main_pairs={(r["frame_id"],int(r["i"]),int(r["j"])) for r in read(HERE/"region_pairs.csv") if int(r["main"])}
    published={(r["frame_id"],r["arm"],int(r["a"]),int(r["b"])):r for r in read(HERE/"frame_results.csv")}
    seen=set()
    v={arm:[] for arm in ARMS}
    maxerr=0.
    for fid in ids:
        ws=by[fid]
        inds=[i for i,w in enumerate(ws) if int(w["valid"])>=922]
        gt=[Fraction(int(w["mm_sum"]),1000*int(w["valid"])) if int(w["valid"]) else 0 for w in ws]
        ij=[(i,j) for i,j in itertools.combinations(inds,2) if abs(gt[i]-gt[j])>=Fraction(1,10)]
        seen.update((fid,int(ws[i]["window"]),int(ws[j]["window"])) for i,j in ij)
        i,j=np.array(ij).T
        d=np.array([float(gt[u]-gt[t]) for u,t in ij])
        def windows(name):
            return np.array([float(w[name]) for w in ws])
        o=windows("O")
        for arm in ARMS:
            gains=[]
            for a,b in PAIRS:
                ms=[]
                for u,t in ((-1,-1),(1,1),(-1,1),(1,-1)):
                    # Aggregate within A and B first, reverse the main estimator's operation order.
                    agg=(o+windows(arm+"_"+key(u*a))+windows(arm+"_"+key(t*b)))/3
                    ms.append(float(np.mean((agg[i]-agg[j]-d)**2)))
                r=published[fid,arm,a,b]
                assert len(i)==int(r["n_regions"])
                direct=(ms[0]+ms[1]-ms[2]-ms[3])/2
                maxerr=max(maxerr,abs(direct-float(r["G"])),*(abs(m-float(r["mse_"+k])) for m,k in zip(ms,["LL","RR","LR","RL"])))
                gains.append(direct)
            v[arm].append([np.mean(gains[:2]),np.mean(gains[2:])])
    assert seen==main_pairs and maxerr<1e-12
    summary={(r["arm"],r["metric"]):r for r in read(HERE/"summary.csv")}
    idx=np.random.default_rng(SEED).integers(0,32,size=(10000,32))
    for arm in ARMS:
        data=np.asarray(v[arm])
        draws=data[idx].mean(axis=1)
        for col,name in enumerate(("G_same_phase","G_cross_phase")):
            vals=[data[:,col].mean(),*np.percentile(draws[:,col],[2.5,97.5])]
            ref=summary[arm,name]
            assert np.allclose(vals,[float(ref[k]) for k in ("estimate","low95","high95")],rtol=0,atol=1e-12)
    audit=read(HERE/"view_audit.csv")
    keys=[(r["frame_id"],r["arm"],r["view"]) for r in audit]
    assert len(set(keys))==len(keys)==544 and {r["frame_id"] for r in audit}==set(ids)
    runs=load(HERE/"runs.json")
    counts=Counter(r["arm"] for r in audit)
    assert counts=={"shared":32,"reflect":256,"real":256}
    for r in audit:
        assert int(r["differing_overlap_pixels"])==0 and r["feature_grid"]=="37x43"
        assert 0<=int(r["differing_pixels"])<=int(r["exposed_pixels"])==abs(int(r["dx"]))*518
    for field,n in (("reflect",288),("real_extra",256),("preprocess",32)):
        assert sum(r[field] for r in runs)>=n
    save(HERE/"replay.json",dict(regions_checked=len(seen),per_recipe_region_checks=len(seen)*4*2*4,
                                 max_frame_mse_or_gain_error=maxerr,intervals_replayed=4,unique_predictions=len(keys),
                                 synchronized_forward_seconds=sum(float(r["forward_ms"]) for r in audit)/1000))

if __name__=="__main__":
    main()
