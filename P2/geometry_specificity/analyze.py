"""Finite-pool decisions, paired uncertainty and pre-development strength matching."""
import argparse
import hashlib
import json
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

from geometry import POOLS
from history import HERE, read, write


def load(path, n):
    rows=read(path)
    ids=sorted({r["pair_id"] for r in rows})
    bank={(r["pair_id"],r["side"],r["view"]):r for r in rows}
    assert len(ids)==n and len(rows)==len(bank)==66*n
    margins={}
    strength=[]
    for v in ["O",*(v for pool in POOLS.values() for v in pool)]:
        values=[]
        pixels=[]
        for i in ids:
            a,b=[bank[i,s,v] for s in "AB"]
            assert a["P0"]==b["P0"] and a["human"]==b["human"]
            values.append((float(a["score"])-float(b["score"]))*(1 if a["human"]=="A" else -1))
            pixels.extend(float(r["mean_abs_rgb255"]) for r in (a,b))
        margins[v]=np.array(values,dtype=np.float64)
        if v!="O":
            delta=abs(margins[v]-margins["O"])
            strength.append(dict(view=v,family=v[0],median_abs_margin_shift=float(np.median(delta)),
                                 mean_abs_margin_shift=float(np.mean(delta)),median_input_rgb255=float(np.median(pixels)),
                                 mean_input_rgb255=float(np.mean(pixels))))
    return np.array(ids),margins,strength


def freeze(path,dest):
    _,_,strength=load(path,64)
    write(dest/"calibration_strength.csv",strength)
    s={r["view"]:r["median_abs_margin_shift"] for r in strength}
    t,g=POOLS["T"],POOLS["S"]
    low=max(min(s[v] for v in t),min(s[v] for v in g))
    high=min(max(s[v] for v in t),max(s[v] for v in g))
    ts=[v for v in t if low<=s[v]<=high]
    ss=[v for v in g if low<=s[v]<=high]
    # Among pools of maximum equal size inside the common interval, minimize
    # total absolute distance. Sorting gives the optimal 1D assignment.
    size=min(len(ts),len(ss))
    choices=[]
    for left,right in product(combinations(ts,size),combinations(ss,size)):
        pairs=list(zip(sorted(left,key=lambda v:(s[v],v)),sorted(right,key=lambda v:(s[v],v))))
        choices.append((sum(abs(s[a]-s[b]) for a,b in pairs),pairs))
    _,pairs=min(choices)
    data=dict(calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calibration_n=64,
              common_low=low,common_high=high,pairs=pairs,version_statistics=strength)
    out=dest/"strength_freeze.json"
    if out.exists():
        assert json.loads(out.read_text())==json.loads(json.dumps(data))
    else:
        out.write_text(json.dumps(data,indent=2)+"\n")
    return data


def calculate(ids,m,pools,ks,scope,dest):
    orig=m["O"]
    right,wrong,tied=orig>0,orig<0,orig==0
    rng=np.random.default_rng(20261128)
    bootstrap=rng.integers(len(ids),size=(10000,len(ids)))
    detail,summary,contrasts,covers,eligible_cases=[],[],[],[],[]
    seven={r["pair_id"]:r for r in read(HERE/"historical_seven.csv")}
    dev=len(ids)==256
    for k in ks:
        single=[f for f,p in pools.items() if len(p)>=k-1]
        counts={}
        means={}
        for f in single+(["TS"] if k%2 and {"T","S"}<=set(single) else []):
            if f=="TS":
                specs=[x+y for x,y in product(combinations(pools["T"],(k-1)//2),combinations(pools["S"],(k-1)//2))]
            else:
                specs=list(combinations(pools[f],k-1))
            d=orig[:,None]+np.stack([sum(m[v] for v in spec) for spec in specs],axis=1)
            c,t,w=(a.sum(1) for a in (d>0,d==0,d<0))
            num=len(specs)
            assert np.all(c+t+w==num)
            counts[f]=(c,t,w,num)
            means[f]=d.mean(1)/k
            for j,i in enumerate(ids):
                detail.append(dict(pair_id=i,K=k,strategy=f,combinations=num,m0=orig[j],mean_margin=means[f][j],
                                   correct=c[j],tie=t[j],wrong=w[j],p_correct=c[j]/num,p_tie=t[j]/num,p_wrong=w[j]/num))
                if scope=="full" and dev and i in seven and f in ("T","S"):
                    hits=np.flatnonzero(d[j]>0)
                    support={v:sum(v in specs[a] for a in hits) for v in pools[f]}
                    solo={v:float(m[v][j]) for v in pools[f]}
                    eligible_cases.append(dict(pair_id=i,K=k,strategy=f,correct=c[j],combinations=num,
                        tie=t[j],wrong=w[j],m0_new=orig[j],correct_combo_inclusions=json.dumps(support),
                        single_margins=json.dumps(solo),two_score_correct=" ".join(v for v in pools[f] if orig[j]+m[v][j]>0),
                        moves_toward_human=" ".join(v for v in pools[f] if m[v][j]>orig[j])))
            record=dict(K=k,strategy=f,combinations=num,n=len(ids),base_correct=int(right.sum()),base_wrong=int(wrong.sum()),
                        base_tie=int(tied.sum()),accuracy_pct=c.mean()*100/num,correct_expected=c.sum()/num)
            for pre,a in (("correct",right),("wrong",wrong),("tie",tied)):
                for name,y in (("correct",c),("tie",t),("wrong",w)):
                    record[pre+"_to_"+name]=y[a].sum()/num
            for name,mask,y in (("fix",wrong,c),("harm",right,w)):
                record[name+"_any_ids"]=" ".join(ids[mask & (y>0)])
                record[name+"_stable_ids"]=" ".join(ids[mask & (4*y>=num)])
            summary.append(record)
        error=float(np.max(abs(means["TS"]-(means["T"]+means["S"])/2))) if "TS" in means else None
        assert error is None or error<1e-12
        for x,y in [("T","S"),("TS","T"),("TS","S"),("B","C")]:
            if x not in counts or y not in counts:
                continue
            cx,_,_,nx=counts[x]
            cy,_,_,ny=counts[y]
            diff=100*(cx/nx-cy/ny)
            lo,hi=np.quantile(diff[bootstrap].mean(1),[.025,.975])
            contrasts.append(dict(K=k,left=x,right=y,left_combos=nx,right_combos=ny,diff_pp=diff.mean(),lo_pp=lo,
                                  hi_pp=hi,mix_identity_worst=error if error is not None else "",resamples=10000))
            for threshold in ("any","quarter"):
                lx=wrong & (cx>0 if threshold=="any" else 4*cx>=nx)
                ly=wrong & (cy>0 if threshold=="any" else 4*cy>=ny)
                r=dict(K=k,left=x,right=y,left_combos=nx,right_combos=ny,threshold=threshold)
                for name,mask in (("left_only",lx&~ly),("right_only",ly&~lx),("both",lx&ly)):
                    r[name]=int(mask.sum())
                    r[name+"_ids"]=" ".join(ids[mask])
                covers.append(r)
    for name,rows in [("pair_probabilities",detail),("strategy_summary",summary),("comparison",contrasts),
                      ("coverage",covers),("seven_parameters",eligible_cases)]:
        if rows:
            write(dest/(scope+"_"+name+".csv"),rows)
    if eligible_cases:
        grouped={i:dict(seven[i]) for i in seven}
        for r in eligible_cases:
            grouped[r["pair_id"]][r["strategy"]+"_K"+str(r["K"])]=f'{r["correct"]}/{r["combinations"]}'
        write(dest/"seven_cases.csv",list(grouped.values()))


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=="freeze":
        freeze(args.scores,args.out)
        return
    ids,m,strength=load(args.scores,args.n)
    write(args.out/"version_strength.csv",strength)
    calculate(ids,m,POOLS,(3,5,9),"full",args.out)
    if args.n==256:
        pairs=json.loads(args.freeze.read_text())["pairs"]
        if pairs:
            pp={f:tuple(p[j] for p in pairs) for j,f in enumerate("TS")}
            calculate(ids,m,pp,[k for k in (2,3,5,9) if k-1<=len(pairs)],"matched",args.out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["freeze","analyze"])
    p.add_argument("--scores",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--n",type=int,choices=[64,256])
    p.add_argument("--freeze",type=Path)
    run(p.parse_args())
