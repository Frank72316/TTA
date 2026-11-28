"""CPU replay of region contrasts, clustered intervals and the prespecified sign test."""
import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from study import HERE,SEED,ARMS,PAIRS,key,phase,read,write,save,load,sha

def intervals(vec,idx):
    v=np.asarray(vec,dtype=np.float64)
    lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
    return dict(estimate=float(v.mean()),low95=float(lo),high95=float(hi),n=len(v),
                positive=int((v>0).sum()),negative=int((v<0).sum()),ties=int((v==0).sum()))

def main(out):
    windows=read(out/"windows.csv")
    assert sha(out/"windows.csv")==load(out/"regions.json")["windows_sha256"]
    frames=read(HERE/"frames.csv")
    assert len(frames)==len({r["scene"] for r in frames})==32
    byfid=defaultdict(list)
    for r in windows:
        byfid[r["frame_id"]].append(r)
    assert set(byfid)=={f["frame_id"] for f in frames}
    regional=[]
    rows=[]
    worst=0.
    for f in frames:
        fid=f["frame_id"]
        win=byfid[fid]
        assert len(win)==16 and all(w["scene"]==f["scene"] for w in win)
        keep=[]
        for a,b in itertools.combinations([w for w in win if int(w["eligible"])],2):
            an,bn=int(a["valid"]),int(b["valid"])
            den=an*bn
            num=int(a["mm_sum"])*bn-int(b["mm_sum"])*an
            mainpair=abs(num)>=100*den
            d=num/(1000*den)
            regional.append(dict(frame_id=fid,scene=f["scene"],i=a["window"],j=b["window"],d=d,main=int(mainpair)))
            if mainpair:
                keep.append((a,b,d))
        if not keep:
            continue
        delta=lambda k:np.asarray([float(a[k])-float(b[k]) for a,b,_ in keep],dtype=np.float64)
        d=np.asarray([item[2] for item in keep])
        o=delta("O")
        for arm,(a,b) in itertools.product(ARMS,PAIRS):
            la,ra,lb,rb=[delta(arm+"_"+key(v)) for v in (-a,a,-b,b)]
            q=[(o+i+j)/3 for i,j in ((la,lb),(ra,rb),(la,rb),(ra,lb))]
            m_ll,m_rr,m_lr,m_rl=[(r-d)**2 for r in q]
            same=(m_ll+m_rr)/2
            opp=(m_lr+m_rl)/2
            g=same-opp
            prod=(la-ra)*(lb-rb)/9
            err=float(np.max(np.abs(prod-g)))
            worst=max(worst,err)
            assert err<1e-12 and np.allclose(q[0]+q[1],q[2]+q[3],rtol=0,atol=1e-12)
            rows.append(dict(frame_id=fid,scene=f["scene"],arm=arm,a=a,b=b,
                        group="same_phase" if phase(a)==phase(b) else "cross_phase",n_regions=len(keep),
                        mse_O=float(np.mean((o-d)**2)),mse_LL=float(m_ll.mean()),mse_RR=float(m_rr.mean()),
                        mse_LR=float(m_lr.mean()),mse_RL=float(m_rl.mean()),mse_same=float(same.mean()),
                        mse_opposite=float(opp.mean()),G=float(g.mean()),G_product=float(prod.mean()),max_residual=err))
    write(out/"region_pairs.csv",regional)
    write(out/"frame_results.csv",rows)
    fids=[f["frame_id"] for f in frames if any(r["frame_id"]==f["frame_id"] for r in rows)]
    idx=np.random.default_rng(SEED).integers(len(fids),size=(10000,len(fids)))
    grouped=defaultdict(dict)
    for r in rows:
        grouped[r["arm"],r["a"],r["b"]][r["frame_id"]]=r
    summary=[]
    vectors={}
    def add(arm,metric,v):
        vectors[arm,metric]=v
        summary.append(dict(arm=arm,metric=metric,**intervals(v,idx)))
    for arm in ARMS:
        def matrix(field):
            return np.array([[grouped[arm,a,b][f][field] for a,b in PAIRS] for f in fids])
        gs=matrix("G")
        for j,(a,b) in enumerate(PAIRS):
            add(arm,f"G_{a}_{b}",gs[:,j])
        same_phase=gs[:,:2].mean(axis=1)
        cross_phase=gs[:,2:].mean(axis=1)
        add(arm,"G_same_phase",same_phase)
        add(arm,"G_cross_phase",cross_phase)
        add(arm,"phase_contrast",same_phase-cross_phase)
        baseline=matrix("mse_O").mean(axis=1)
        opp=matrix("mse_opposite")
        same=matrix("mse_same")
        rule=np.column_stack((opp[:,:2],same[:,2:])).mean(axis=1)
        for name,v in (("mse_O",baseline),("mse_fixed_opposite",opp.mean(axis=1)),
                       ("mse_fixed_same",same.mean(axis=1)),("mse_phase_rule",rule),
                       ("fixed_opposite_minus_rule",opp.mean(axis=1)-rule),
                       ("O_minus_rule",baseline-rule),("O_minus_fixed_opposite",baseline-opp.mean(axis=1))):
            add(arm,name,v)
        assert np.allclose(opp.mean(axis=1)-rule,-cross_phase/2,rtol=0,atol=1e-12)
    for name in ("G_same_phase","G_cross_phase","phase_contrast"):
        add("real_minus_reflect",name,vectors["real",name]-vectors["reflect",name])
    write(out/"summary.csv",summary)
    sl={(r["arm"],r["metric"]):r for r in summary}
    decision={}
    for arm in ARMS:
        ag=sl[arm,"G_same_phase"]
        cf=sl[arm,"G_cross_phase"]
        decision[arm]=dict(both_signs=bool(ag["estimate"]>0 and cf["estimate"]<0),
                           strict_flip=bool(ag["low95"]>0 and cf["high95"]<0),
                           positive_modulation=bool(sl[arm,"phase_contrast"]["low95"]>0))
    save(out/"validation.json",dict(frames=len(fids),scenes=len(fids),excluded=[r["frame_id"] for r in frames if r["frame_id"] not in fids],
                                    eligible_pairs=len(regional),primary_pairs=sum(r["main"] for r in regional),
                                    identity_checks=sum(r["n_regions"] for r in rows),max_residual=worst,
                                    arms=decision,external_confirmation_gate=all(v["strict_flip"] for v in decision.values())))
    print(decision)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=HERE)
    main(ap.parse_args().out)
