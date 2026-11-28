"""Audit records and compare float64 decisions against exact scaled-integer sums."""
import argparse
import csv
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from geometry import PARAMS, POOLS
from history import HERE, read, write


def verify(root):
    audit=[]
    for cohort,n in (("old64",64),("dev256",256)):
        folder=root/cohort
        scores=read(folder/"scores.csv")
        grouped={}
        for r in scores:
            i,s,v=r["pair_id"],r["side"],r["view"]
            side=grouped.setdefault(i,{}).setdefault(s,{})
            assert v not in side
            side[v]=r
        ids=sorted(grouped)
        assert len(ids)==n and len(scores)==66*n
        manifest=(HERE.parent.parent/"P1/baseline/pairs.csv" if n==64 else HERE.parent/"translation/pairs_frozen.csv")
        if manifest.is_file():
            originals={r["pair_id"]:r for r in read(manifest)}
            assert set(originals)==set(ids) and len({r["prompt_id"] for r in originals.values()})==n
            for i,p in originals.items():
                a,b=int(p["rank_A"]),int(p["rank_B"])
                assert a!=b and p["human_winner_P0"]==("A" if a<b else "B")
                for side in "AB":
                    r=grouped[i][side]["O"]
                    assert (r["P0"],r["prompt_id"],r["human"],r["image_id"])==(p["P0"],p["prompt_id"],p["human_winner_P0"],p["image_"+side])
        scale=max(float(r["score"]).as_integer_ratio()[1] for r in scores)
        marg={v:[] for v in PARAMS}
        duplicates=0
        for i in ids:
            sides=grouped[i]
            assert all(set(sides[s])==set(PARAMS) for s in "AB")
            for s in "AB":
                rows=sides[s]
                duplicates+=len(rows)-len({r["tensor_sha"] for r in rows.values()})
                for r in rows.values():
                    source=rows[r["scored_view"]]
                    assert r["tensor_sha"]==source["tensor_sha"] and r["score"]==source["score"]
                    assert r["token_sha"]==rows["O"]["token_sha"] and r["image_id"]==rows["O"]["image_id"]
            for v in PARAMS:
                a,b=[sides[s][v] for s in "AB"]
                assert a["P0"]==b["P0"] and a["human"]==b["human"]
                va,vb=float(a["score"])*scale,float(b["score"])*scale
                assert va.is_integer() and vb.is_integer()
                marg[v].append(int(va-vb)*(1 if a["human"]=="A" else -1))
        arr={v:np.asarray(values,dtype=np.int64) for v,values in marg.items()}
        if cohort=="dev256":
            legacy=HERE.parent/"translation/scores.csv"
            if legacy.is_file():
                hist={(r["pair_id"],r["side"]):float(r["score"]) for r in read(legacy) if r["view"]=="O"}
                seven={r["pair_id"]:r for r in read(HERE/"historical_seven.csv")}
                controls=[]
                old_views=[v for v in POOLS["T"] if v.endswith("_3")]
                for j,i in enumerate(ids):
                    m0=arr["O"][j]/scale
                    side=grouped[i]["A"]["O"]["human"]
                    h=(hist[i,"A"]-hist[i,"B"])*(1 if side=="A" else -1)
                    samepool=[arr["O"][j]+arr[a][j]+arr[b][j] for a,b in combinations(old_views,2)]
                    controls.append(dict(pair_id=i,old_m0=h,new_m0=m0,
                                         baseline_changed=int(np.sign(h)!=np.sign(m0)),historical_seven=int(i in seven),
                                         historical_TT_correct=seven[i]["TT"] if i in seven else "",
                                         current_3pct_TT_correct=sum(x>0 for x in samepool),
                                         current_3pct_TT_tie=sum(x==0 for x in samepool)))
                write(root/"baseline_alignment.csv",controls)
        prob=read(folder/"full_pair_probabilities.csv")
        lookup={(r["pair_id"],int(r["K"]),r["strategy"]):r for r in prob}
        assert len(lookup)==len(prob)==15*n
        cells=0
        for k,f in product((3,5,9),("B","C","T","S","TS")):
            specs=(combinations(POOLS[f],k-1) if f!="TS" else
                   (a+b for a,b in product(combinations(POOLS["T"],(k-1)//2),combinations(POOLS["S"],(k-1)//2))))
            pos=np.zeros(n,dtype=int)
            zero=pos.copy()
            neg=pos.copy()
            count=0
            for spec in specs:
                sums=arr["O"].copy()
                for v in spec:
                    sums+=arr[v]
                pos+=sums>0
                zero+=sums==0
                neg+=sums<0
                count+=1
            cells+=count*n
            for ix,i in enumerate(ids):
                r=lookup[i,k,f]
                assert (pos[ix],zero[ix],neg[ix],count)==tuple(int(r[key]) for key in ("correct","tie","wrong","combinations"))
        runs=json.loads((folder/"runs.json").read_text())
        assert len(scores)==sum(r["new"]+r["duplicate_hits"] for r in runs)
        audit.append(dict(cohort=cohort,pairs=n,scores=len(scores),unique_tensors_within_image=len(scores)-duplicates,
                          unique_tensor_hashes=len({r["tensor_sha"] for r in scores}),
                          unique_tensor_token_pairs=len({(r["tensor_sha"],r["token_sha"]) for r in scores}),
                          duplicate_tensors=duplicates,integer_combination_decisions=cells,decision_mismatches=0))
    (root/"verification.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    verify(p.parse_args().root)
