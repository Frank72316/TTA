"""Frame-equal estimates, paired scene intervals, ranking transitions, and frozen cases."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from order_data import HERE,SOURCE,RECIPES,CHANGES,read,write,save,sha,frozen

SCOPES=("main","small","medium","large","near","equal")
STATES=("wrong","correct","tie")
METRICS=("mse_O","mse_same","mse_opposite","G","gain_vs_O","acc_O","acc_same","acc_opposite","A","acc_vs_O",
         "fix_rate","harm_rate","tie_net_rate","strict_flip_rate","any_flip_rate","tie_O","tie_same","tie_opposite")

def frame_stats(fid,scene,scope,rr):
    n=len(rr)
    record=dict(frame_id=fid,scene=scene,scope=scope,n=n)
    record.update(dict.fromkeys(METRICS,""))
    if not n:return record,{}
    avg=lambda key:np.mean([float(r[key]) for r in rr],dtype=np.float64)
    o=avg("sqerr_O");same=avg("mse_same");opp=avg("mse_opposite")
    record.update(mse_O=o,mse_same=same,mse_opposite=opp,G=avg("G"),gain_vs_O=o-opp)
    assert abs(record["G"]-(same-opp))<1e-12
    if scope=="equal":return record,{}
    state_perc=lambda k,s:100*sum(r["state_"+k]==s for r in rr)/n
    for state,prefix in (("correct","acc"),("tie","tie")):
        single=state_perc("O",state)
        a,b,c,d=[state_perc(k,state) for k in ("LL","RR","LR","RL")]
        record.update({prefix+"_O":single,prefix+"_same":(a+b)/2,prefix+"_opposite":(c+d)/2})
    record["A"]=record["acc_opposite"]-record["acc_same"]
    record["acc_vs_O"]=record["acc_opposite"]-record["acc_O"]
    counts={(u,v):Counter((r["state_"+u],r["state_"+v]) for r in rr) for u,v in CHANGES}
    total=sum(counts.values(),Counter())
    rate=lambda a,b:100*total[a,b]/(2*n)
    record.update(fix_rate=rate("wrong","correct"),harm_rate=rate("correct","wrong"),
                  tie_net_rate=rate("tie","correct")-rate("correct","tie"))
    record["strict_flip_rate"]=record["fix_rate"]+record["harm_rate"]
    record["any_flip_rate"]=100*sum(any({r["state_"+u],r["state_"+v]}=={"correct","wrong"} for u,v in CHANGES) for r in rr)/n
    assert abs(record["A"]-(record["fix_rate"]-record["harm_rate"]+record["tie_net_rate"]))<1e-12
    return record,counts

def cluster_stats(frame_rows,scope):
    valid=[r for r in frame_rows if r["n"]]
    base=dict(scope=scope,region_pairs=sum(r["n"] for r in valid),frames=len(valid),scenes=len({r["scene"] for r in valid}))
    table=[]
    if valid:
        groups=[np.array([i for i,r in enumerate(valid) if r["scene"]==s]) for s in sorted({r["scene"] for r in valid})]
        sizes=np.array([len(ix) for ix in groups])
        rng=np.random.default_rng(20261128);draws=rng.integers(len(groups),size=(10000,len(groups)))
        denom=sizes[draws].sum(axis=1)
    for key in METRICS:
        unit="m2" if key.startswith("mse_") or key in ("G","gain_vs_O") else "%"
        if key in ("A","acc_vs_O","tie_net_rate"):unit="pp"
        r=dict(**base,metric=key,unit=unit,
               estimate="",ci_low="",ci_high="")
        if valid and valid[0][key]!="":
            vec=np.array([f[key] for f in valid],dtype=np.float64)
            r["estimate"]=float(vec.mean())
            sums=np.array([vec[ix].sum() for ix in groups])
            ci=np.quantile(sums[draws].sum(axis=1)/denom,[.025,.975])
            r.update(ci_low=ci[0],ci_high=ci[1])
        table.append(r)
    return table

def choose_cases(main_rows):
    records=[]
    for label in ("unchanged","corrected"):
        seen=set();first=None;count=0
        for r in main_rows:
            if float(r["G"])<=0:continue
            for u,v in CHANGES:
                before=r["state_"+u];after=r["state_"+v]
                eligible=before==after and before!="tie" if label=="unchanged" else before=="wrong" and after=="correct"
                if eligible and float(r["sqerr_"+v])<float(r["sqerr_"+u]):
                    count+=1;seen.add((r["frame_id"],r["A"],r["B"]))
                    if first is None:first=dict(frame_id=r["frame_id"],A=int(r["A"]),B=int(r["B"]),before=u,after=v)
        records.append(dict(kind=label,eligible_directed_comparisons=count,eligible_distinct_pairs=len(seen),first=first))
    return records

def run(out):
    frozen()
    log=json.loads((out/"run.json").read_text())
    for name,h in log["hashes"].items():assert sha(out/name)==h
    rows=read(out/"region_pairs.csv");wins=read(out/"windows.csv")
    frames=read(SOURCE/"frame_results.csv");index={r["frame_id"]:n for n,r in enumerate(frames)}
    assert len(wins)==2048 and len(frames)==128
    assert [(index[r["frame_id"]],int(r["A"]),int(r["B"])) for r in rows]==sorted({(index[r["frame_id"]],int(r["A"]),int(r["B"])) for r in rows})
    eligible=Counter(w["frame_id"] for w in wins if w["eligible"]=="1")
    strata=defaultdict(list);all_counts=Counter()
    for r in rows:
        fid=r["frame_id"];s=r["stratum"];all_counts[fid]+=1;strata[s,fid].append(r)
        if s in ("small","medium","large"):strata["main",fid].append(r)
    assert all(all_counts[fid]==eligible[fid]*(eligible[fid]-1)//2 for fid in index)
    by_frame=[];summaries=[];trans=[];coverage=[]
    for scope in SCOPES:
        scope_table=[];totals={change:Counter() for change in CHANGES}
        for f in frames:
            fid=f["frame_id"]
            record,c=frame_stats(fid,f["scene"],scope,strata[scope,fid])
            scope_table.append(record)
            for path,counts in c.items():totals[path].update(counts)
        summary=cluster_stats(scope_table,scope)
        for (u,v),c in totals.items():
            if scope=="equal":continue
            n=sum(c.values());assert n==summary[0]["region_pairs"]
            flat={a+"_to_"+b:c[a,b] for a in STATES for b in STATES}
            trans.append(dict(scope=scope,before=u,after=v,n=n,corrected=c["wrong","correct"],harmed=c["correct","wrong"],
                    involving_tie=sum(count for pair,count in c.items() if "tie" in pair),
                    unchanged_strict=c["wrong","wrong"]+c["correct","correct"],**flat))
        by_frame.extend(scope_table);summaries.extend(summary)
        coverage.append({k:summary[0][k] for k in ("scope","region_pairs","frames","scenes")})
    write(out/"frame_results.csv",by_frame);write(out/"summary.csv",summaries);write(out/"transitions.csv",trans);write(out/"coverage.csv",coverage)
    selected=[r for r in rows if r["stratum"] in ("small","medium","large")]
    strict=lambda r,u,v:{r["state_"+u],r["state_"+v]}=={"wrong","correct"}
    detail=dict(total_pairs=len(rows),main_pairs=len(selected),valid_windows=sum(eligible.values()),
           cases=choose_cases(selected),max_G_residual=max(float(r["identity_error"]) for r in rows),
           main_pairs_with_any_flip=sum(any(strict(r,u,v) for u,v in CHANGES) for r in selected),
           main_pairs_with_any_tie=sum(any(r["state_"+k]=="tie" for k in ("LL","LR","RR","RL")) for r in selected),
           bootstrap_seed=20261128,replicates=10000,new_forwards=0)
    save(out/"validation.json",detail)
    print(json.dumps(detail))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE);run(p.parse_args().out)
