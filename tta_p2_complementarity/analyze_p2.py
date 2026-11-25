"""Enumerate decisions before pooling. Bootstrap complete pair-level outcome vectors."""
from collections import Counter
import math
import time
import numpy as np
from study import *

STATES=("correct","wrong","tie")
STRATEGIES=["O",*PURE,*MIX]

def input_scores():
    rows=read_csv(ROOT/"scores.csv")
    bank={}
    ps={p["pair_id"]:p for p in pairs()}
    old=pilot_scores()
    for r in rows:
        i,s,v=r["pair_id"],r["side"],r["view"]; k=(i,s,v)
        assert k not in bank and r["image_id"]==ps[i]["image_"+s] and r["P0"]==ps[i]["P0"]
        assert r["native_dtype"]=="float16" and all(x not in ("",None) for x in r.values())
        assert r["family"]==v[0] and (float(r["input_mean_abs_RGB255"])>0)==(v!="O")
        bank[k]=float(r["score"])
        assert math.isfinite(bank[k]) and np.float16(bank[k])==bank[k]
        if k in old: assert old[k]==bank[k]
    assert bank.keys()==set(itertools.product(ps,("A","B"),VERSIONS))
    assert len({(r["pair_id"],r["token_sha256"]) for r in rows})==64
    return list(ps.values()),bank,rows

def enumerate_decisions(ps,bank):
    combos=[]; per=[]
    for p in ps:
        i=p["pair_id"]; human=p["human_winner_P0"]
        orig=bank[i,"A","O"]-bank[i,"B","O"]; base=outcome(choice(orig),human)
        for st in STRATEGIES:
            versions=[()] if st=="O" else combinations(st)
            records=[]
            for vs in versions:
                views=("O",*vs)
                a,b=[sum(bank[i,side,v] for v in views) for side in ("A","B")]
                # Compare the exact binary sums before division by the common budget.
                d=(a-b)/len(views); winner=choice(d); result=outcome(winner,human)
                rec=dict(pair_id=i,strategy=st,view1=vs[0] if vs else "—",view2=vs[1] if vs else "—",
                         budget=len(views),human_winner=human,d0=orig,base_pick=choice(orig),base_outcome=base,
                         mean_A=a/len(views),mean_B=b/len(views),d_aug=d,pick=winner,result=result)
                records.append(rec)
            combos.extend(records)
            n=len(records);counts=Counter(r["result"] for r in records)
            entry=dict(pair_id=i,strategy=st,human_winner=human,base_pick=choice(orig),base_outcome=base,d0=orig,n=n,
                       d_min=min(r["d_aug"] for r in records),d_max=max(r["d_aug"] for r in records))
            for y in STATES:
                entry[y+"_n"]=counts[y]
                entry[y+"_rate"]=counts[y]/n
            entry["correction_rate"]=counts["correct"]/n if base=="wrong" else ""
            entry["harm_rate"]=counts["wrong"]/n if base=="correct" else ""
            per.append(entry)
    assert len(combos)==64*(1+4*6+6*16) and len(per)==64*11
    return combos,per

def summarize(per):
    ids=[p["pair_id"] for p in pairs()]
    data={(r["pair_id"],r["strategy"]):r for r in per}
    # 48 is a common denominator for both 1/6 and 1/16.
    unit=np.array([[data[i,s]["correct_n"]*(48//data[i,s]["n"]) for s in STRATEGIES] for i in ids])
    draw=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)))
    boot=100*unit[draw].mean(axis=1)/48
    acc=100*unit.mean(axis=0)/48
    def ci(x):return np.quantile(x,[.025,.975],method="linear").tolist()
    stats=[]
    for j,s in enumerate(STRATEGIES):
        group=[data[i,s] for i in ids]
        rec=dict(strategy=s,classification="reference" if s=="O" else "single" if s in PURE else "primary" if s in PRIMARY else "exploratory",
                 n_pairs=len(ids),n_combinations=group[0]["n"])
        for y in STATES:
            rec[y+"_expected"]=sum(r[y+"_n"] for r in group)/rec["n_combinations"]
            rec[y+"_rate"]=rec[y+"_expected"]/len(ids)
        for x,y in itertools.product(STATES,repeat=2):
            rec[x+"_to_"+y]=sum(r[y+"_n"] for r in group if r["base_outcome"]==x)/rec["n_combinations"]
        rec["delta_vs_O_pp"]=acc[j]-acc[0]
        rec["delta_vs_O_lo"],rec["delta_vs_O_hi"]=ci(boot[:,j]-boot[:,0])
        rec["net_correct_pairs"]=rec["correct_expected"]-stats[0]["correct_expected"] if stats else 0.
        assert math.isclose(sum(rec[y+"_rate"] for y in STATES),1)
        stats.append(rec)
    comparisons=[]
    for st in MIX:
        j=STRATEGIES.index(st); sa,sb=st[0]*2,st[1]*2
        a,b=STRATEGIES.index(sa),STRATEGIES.index(sb)
        winner=sa if acc[a]>acc[b] else sb if acc[b]>acc[a] else sa+"="+sb
        for label,av,samples in [(sa,acc[a],boot[:,a]),(sb,acc[b],boot[:,b]),
                                 ("best_of_components",max(acc[a],acc[b]),np.maximum(boot[:,a],boot[:,b]))]:
            lo,hi=ci(boot[:,j]-samples)
            comparisons.append(dict(strategy=st,comparison=label,observed_stronger=winner,
                    diff_pp=acc[j]-av,lo_pp=lo,hi_pp=hi,bootstrap_draws=10000,seed=42))
    return stats,comparisons

def summarize_shifts(bank,raw,ps):
    shifts=[]
    for f,vs in FAMILIES.items():
        for label,pool in [(f,vs),*((v,[v]) for v in vs)]:
            single=[bank[p["pair_id"],s,v]-bank[p["pair_id"],s,"O"] for p,s,v in itertools.product(ps,("A","B"),pool)]
            gap=[(bank[p["pair_id"],"A",v]-bank[p["pair_id"],"A","O"])-(bank[p["pair_id"],"B",v]-bank[p["pair_id"],"B","O"]) for p,v in itertools.product(ps,pool)]
            selected=[r for r in raw if r["view"] in pool]
            a=np.abs(single)
            shifts.append(dict(variant=label,n_scores=len(single),mean_signed=np.mean(single),median_signed=np.median(single),
                     median_abs=np.median(a),mean_abs=np.mean(a),q25_abs=np.quantile(a,.25),q75_abs=np.quantile(a,.75),
                     median_abs_gap_shift=np.median(np.abs(gap)),mean_abs_gap_shift=np.mean(np.abs(gap)),
                     median_input_RGB255=np.median([float(r["input_mean_abs_RGB255"]) for r in selected]),
                     unchanged_model_inputs=sum(float(r["input_mean_abs_RGB255"])==0 for r in selected)))
    return shifts

def mechanism(per):
    out=[]
    sets={}
    for st in STRATEGIES:
        rows=[r for r in per if r["strategy"]==st]
        sets[st]={k:{r["pair_id"] for r in rows if r["base_outcome"]==old and r[new+"_n"]>0}
                  for k,old,new in [("fix","wrong","correct"),("harm","correct","wrong")]}
    for st in MIX:
        a,b=st[0]*2,st[1]*2
        for event in ("fix","harm"):
            la,lb,m=(sets[name][event] for name in (a,b,st))
            photo=set.union(*(sets[s][event] for s in ("BB","CC","BC")))
            pure=set.union(*(sets[s][event] for s in PURE))
            other=set.union(*(sets[s][event] for s in PURE+MIX if s!=st))
            fmt=lambda v:" ".join(sorted(v)) or "—"
            out.append(dict(strategy=st,event=event,component_a=a,component_b=b,
                            support_a=fmt(la),support_b=fmt(lb),support_mix=fmt(m),
                            component_intersection=fmt(la&lb),
                            jaccard_components=len(la&lb)/len(la|lb) if la and lb else "NA",
                            unique_vs_components=fmt(m-(la|lb)),unique_vs_all_pure=fmt(m-pure),
                            unique_vs_photo=fmt(m-photo),unique_vs_all_other_strategies=fmt(m-other)))
    return out

def main():
    start=time.perf_counter()
    ps,bank,raw=input_scores()
    combos,per=enumerate_decisions(ps,bank)
    stats,comp=summarize(per)
    for name,rows in [("combination_results.csv",combos),("pair_strategy_results.csv",per),("strategy_summary.csv",stats),
                      ("mix_comparisons.csv",comp),("score_shifts.csv",summarize_shifts(bank,raw,ps)),("mechanism.csv",mechanism(per))]:
        write_csv(ROOT/name,rows)
    print("64 pairs x 120 three-view version choices + 64 baseline decisions;",time.perf_counter()-start,"seconds")

if __name__=="__main__":
    main()

