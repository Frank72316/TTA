"""Independent candidate-first exact aggregation and comparison to the supplied analysis."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from prepare import ROOT, read, write, as_json

METHODS=("original","center","matched","swapped")
QS=("crown","wing")
PART={"crown":"鸟冠","wing":"翅膀"}
NAMES={"original":"原图","center":"中心裁剪","matched":"对应部位","swapped":"交换部位"}
def pick(a,b):
    return "A" if a>b else "B" if a<b else "tie"

def verify(out):
    rows=read(ROOT/"pairs.csv");scores=read(out/"scores.csv")
    assert len(scores)==1536 and len(rows)==192
    images={r["image_id"]:r for r in rows}
    pairs=defaultdict(dict)
    for r in rows:pairs[r["pair_id"]][r["side"]]=r
    by={}
    for s in scores:
        i=s["image_id"];r=images[i]
        assert (s["side"],s["pair_id"])==(r["side"],r["pair_id"])
        assert s["prompt"]==f"The bird has {r['color']} plumage on its {s['query']}."
        key=(i,s["view"],s["query"])
        assert key not in by
        # Each stored decimal must round-trip losslessly to its native binary16 value.
        val=Fraction(s["score"]);assert val==Fraction.from_float(float(np.float16(s["score"])))
        by[key]=val
    assert len({(s["image_id"],s["view"],s["tensor_sha256"]) for s in scores})==768
    bundled={r["pair_id"]:r for r in read(out/"results/pair_results.csv")}
    assert sorted(pairs)==sorted(bundled) and len(pairs)==96
    summary=json.loads((out/"results/summary.json").read_text())
    questions=[];per_pair=[];matched_margins=0
    for pid,ab in sorted(pairs.items()):
        a,b=ab["A"],ab["B"]
        assert a["species"]==b["species"] and a["color"]==b["color"]
        pair=dict(pair_id=pid,species=a["species"],color=a["color"],image_A=a["image_id"],image_B=b["image_id"],
                  definite=int(all(r[q+"_certainty"]=="4" for r in (a,b) for q in QS)))
        flags=Counter()
        for q in QS:
            other=next(v for v in QS if v!=q)
            gold=next(k for k in ab if ab[k][q+"_present"]=="1")
            assert ab[gold][other+"_present"]=="0" and sum(int(r[q+"_present"]) for r in (a,b))==1
            row=dict(**{k:pair[k] for k in ("pair_id","species","color","definite","image_A","image_B")},query=q,gold=gold)
            pair[q+"_gold"]=gold
            for method,vs in zip(METHODS,[("original",),("original","center"),("original",q),("original",other)]):
                one,two=[sum(by[r["image_id"],v,q] for v in vs)/len(vs) for r in (a,b)]
                winner=pick(one,two);status="tie" if winner=="tie" else ("correct" if winner==gold else "wrong")
                signed=(one-two)*(1 if gold=="A" else -1)
                assert signed==Fraction(bundled[pid][method+"_"+q]);matched_margins+=1
                row.update({method+"_A":float(one),method+"_B":float(two),method+"_margin":float(signed),
                            method+"_pick":winner,method+"_status":status})
                pair[method+"_"+q+"_pick"]=winner
                flags[method]+=int(status=="correct")
            questions.append(row)
        assert pair["crown_gold"]!=pair["wing_gold"]
        for m in METHODS:pair[m+"_strict_questions"]=flags[m]
        per_pair.append(pair)
    write(out/"questions.csv",questions);write(out/"pair_choices.csv",per_pair)
    matrix=[];tables=[];trans=[]
    for scope in ("all","definitely"):
        qrows=[r for r in questions if scope=="all" or r["definite"]]
        prows=[r for r in per_pair if scope=="all" or r["definite"]]
        for q in QS:
            group=[r for r in qrows if r["query"]==q]
            for crop in QS:
                m="matched" if q==crop else "swapped"
                c=Counter(r[m+"_status"] for r in group)
                matrix.append(dict(subset=scope,query=q,added_view=crop,total=len(group),correct=c["correct"],wrong=c["wrong"],tie=c["tie"]))
        for m in METHODS:
            c=Counter(r[m+"_status"] for r in qrows);p=Counter(r[m+"_strict_questions"] for r in prows)
            table=dict(subset=scope,method=m,total=len(qrows),correct=c["correct"],wrong=c["wrong"],tie=c["tie"],
                       strict_accuracy=c["correct"]/len(qrows),both=p[2],one=p[1],neither=p[0])
            for q in QS:table[q+"_correct"]=sum(r[m+"_status"]=="correct" for r in qrows if r["query"]==q)
            if scope=="all":
                assert (table["both"],table["tie"],table["strict_accuracy"],table["crown_correct"],table["wing_correct"])==tuple(
                    summary[m][k] for k in ("both_correct","tie_questions","strict_accuracy","crown_correct","wing_correct"))
            tables.append(table)
        for ref,m in [("original","center"),("original","matched"),("original","swapped"),("center","matched"),("swapped","matched")]:
            for q in ("both",)+QS:
                counter=Counter((r[ref+"_status"],r[m+"_status"]) for r in qrows if q=="both" or r["query"]==q)
                for src in ("correct","wrong","tie"):
                    for dst in ("correct","wrong","tie"):
                        trans.append(dict(subset=scope,reference=ref,method=m,query=q,before=src,after=dst,n=counter[src,dst]))
    write(out/"four_cells.csv",matrix);write(out/"strategy_summary.csv",tables);write(out/"transitions.csv",trans)
    # Independent grouped sums. Equal weight per question even if cluster sizes differ.
    cluster=sorted({r["species"] for r in per_pair})
    sums=np.array([[sum(r[m+"_strict_questions"] for r in per_pair if r["species"]==c) for m in METHODS]
                  for c in cluster],dtype=np.int32)
    sizes=np.array([2*sum(r["species"]==c for r in per_pair) for c in cluster])
    rng=np.random.default_rng(20261128);draw=rng.integers(len(cluster),size=(10000,len(cluster)))
    intervals=[]
    for i,ref in enumerate(METHODS):
        if ref=="matched":continue
        d=sums[:,METHODS.index("matched")]-sums[:,i]
        boot=d[draw].sum(1)/sizes[draw].sum(1)
        ci=100*np.quantile(boot,[.025,.975]);pp=100*d.sum()/sizes.sum()
        bundle=summary["matched_minus_"+ref]
        assert np.allclose(ci,bundle["ci_pp"],rtol=0,atol=1e-12) and abs(pp-bundle["delta_pp"])<1e-12
        intervals.append(dict(reference=ref,delta_pp=pp,ci_low_pp=ci[0],ci_high_pp=ci[1]))
    write(out/"contrasts.csv",intervals)
    cases=[]
    for count in (2,0):
        candidates=[r for r in per_pair if r["matched_strict_questions"]==count]
        if count==2:candidates.sort(key=lambda r:(r["original_strict_questions"]==2,r["pair_id"]))
        if count==0 and not candidates:candidates=[r for r in per_pair if r["matched_strict_questions"]==1]
        if candidates:cases.append(candidates[0]["pair_id"])
    record=dict(questions=len(questions),pairs=len(per_pair),verified_oriented_margins=matched_margins,
                clusters=len(cluster),bootstrap_replicates=len(draw),verified_ci=len(intervals),example_pairs=cases,
                full_score_rows=len(scores),unique_inputs=len({r["tensor_sha256"] for r in scores}))
    assert sum(r["definite"] for r in per_pair)==29
    as_json(out/"validation.json",record)
    return cases

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT);a=p.parse_args();print(verify(a.out))
