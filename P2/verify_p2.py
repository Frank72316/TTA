"""Independent audit using exact integer score units and integer vote counts."""
from collections import Counter,defaultdict
from fractions import Fraction
import numpy as np
from study import *

def main():
    ps={p["pair_id"]:p for p in pairs()}
    raw=read_csv(ROOT/"scores.csv")
    bank={(r["pair_id"],r["side"],r["view"]):Fraction(r["score"]) for r in raw}
    assert len(raw)==len(bank)==2176 and set(bank)==set(itertools.product(ps,("A","B"),VERSIONS))
    envsha=digest(ROOT/"environment.json")
    cached=[]
    for i,p in ps.items():
        for side in ("A","B"):
            rec=file_json(ROOT/"cache"/f"{i}_{side}.json")
            assert rec["protocol"]==envsha
            cached.extend(rec["rows"])
    assert [{k:str(v) for k,v in r.items()} for r in cached]==raw
    assert all(r["image_id"]==ps[r["pair_id"]]["image_"+r["side"]] and r["P0"]==ps[r["pair_id"]]["P0"] for r in raw)
    counts=Counter(r["source"] for r in raw)
    runs=file_json(ROOT/"runs.json")
    assert counts["P2_fresh"]==sum(r["new_inferences"] for r in runs)
    assert len(raw)-counts["P2_fresh"]==sum(r["old_hits"] for r in runs)
    decisions=read_csv(ROOT/"combination_results.csv")
    totals=defaultdict(Counter);seen=set()
    for r in decisions:
        i,s=r["pair_id"],r["strategy"];vs=[r["view1"],r["view2"]] if s!="O" else []
        k=i,s,tuple(vs);assert k not in seen;seen.add(k)
        assert len(vs)==len(set(vs)) and int(r["budget"])==1+len(vs)
        assert vs==[] or tuple(vs) in combinations(s)
        margin=sum(bank[i,"A",v]-bank[i,"B",v] for v in ["O",*vs])/(len(vs)+1)
        assert float(margin)==float(r["d_aug"])
        result=outcome(choice(margin),ps[i]["human_winner_P0"])
        assert r["result"]==result and r["pick"]==choice(margin)
        totals[i,s][result]+=1
    per=read_csv(ROOT/"pair_strategy_results.csv");assert len(per)==64*11 and len(seen)==64*121
    expected=defaultdict(lambda:defaultdict(Fraction))
    for r in per:
        i,s=r["pair_id"],r["strategy"];t=totals[i,s];n=sum(t.values())
        assert n==int(r["n"])==(1 if s=="O" else 6 if s in PURE else 16)
        for state in ("correct","wrong","tie"):
            assert t[state]==int(r[state+"_n"])
            expected[s][state]+=Fraction(t[state],n)
    sums=read_csv(ROOT/"strategy_summary.csv")
    for r in sums:
        st=r["strategy"]
        assert sum(expected[st].values())==64
        for state in ("correct","wrong","tie"):
            assert float(expected[st][state])==float(r[state+"_expected"])
    assert expected["O"]=={"correct":43,"wrong":21,"tie":0}
    # Cross-check paired bootstrap from integer outcomes using multinomial pair weights.
    keys=["O",*PURE,*MIX];ids=list(ps)
    rates=np.array([[totals[i,s]["correct"] / sum(totals[i,s].values()) for s in keys] for i in ids])
    sample=np.random.default_rng(42).integers(0,64,size=(10000,64))
    weights=np.array([np.bincount(draw,minlength=64) for draw in sample])
    boot=weights@rates*100/64
    for r in read_csv(ROOT/"mix_comparisons.csv"):
        st=r["strategy"];a,b=keys.index(st[0]*2),keys.index(st[1]*2)
        control=np.maximum(boot[:,a],boot[:,b]) if r["comparison"]=="best_of_components" else boot[:,keys.index(r["comparison"])]
        assert np.allclose(np.quantile(boot[:,keys.index(st)]-control,[.025,.975]),[float(r["lo_pp"]),float(r["hi_pp"])],atol=1e-12,rtol=0)
    assert len({(r["model_width"],r["model_height"]) for r in raw})==1
    assert all(float(r["input_mean_abs_RGB255"])>0 for r in raw if r["view"]!="O")
    for line in (ROOT/"logs/pilot-before.sha256").read_text().splitlines():
        h,name=line.split("  ",1);assert digest(ROOT/name)==h
    print(f"PASS: 64 frozen P0 pairs, 128 images, 2176 unique scores; baseline 43/21/0; {len(seen)-64} strict 3-view decisions; 640 strategy/pair rows plus 64 baseline; 18 paired CI checks. Input files unchanged.")

if __name__=="__main__":main()

