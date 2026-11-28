"""Recompute all decisions and uncertainty from the four native scores per photo."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from common import HERE, SEED, choose, rows, save_rows, sha

STRATEGIES=("original","image_only","synchronized")
NAMES={"original":"原图","image_only":"只翻图","synchronized":"图文同步"}
OTHER={"left":"right","right":"left"}
METRICS=("both","question_accuracy","question_tie","same_choice")


def decisions(a,b):
    """Each entry of a and b is a mapping (view, query) -> score."""
    result={}
    for q,other in OTHER.items():
        o=a["O",q]-b["O",q]
        f=a["F",q]-b["F",q]
        f_other=a["F",other]-b["F",other]
        for st,m,k in zip(STRATEGIES,(o,o+f,o+f_other),(1,2,2)):
            result[st,q]=(choose(m),m/k)
    return result


def analyze(out):
    pairs=rows(out/"pairs.csv"); n=len(pairs)
    assert all(sha((out/f).read_bytes())==v for f,v in json.loads((out/"freeze.json").read_bytes())["hashes"].items())
    score_rows=rows(out/"scores.csv")
    bank={(r["pair_id"],r["side"],r["view"],r["direction"]):r for r in score_rows}
    assert len(bank)==len(score_rows)==8*n and len({r["pair_id"] for r in pairs})==n
    results=[]; score_cache=[]
    for p in pairs:
        pid=p["pair_id"]
        assert {p["winner_"+q] for q in OTHER}=={"A","B"}
        assert p["prompt_left"].replace(" to the left of "," to the right of ")==p["prompt_right"]
        photos=[]
        for side in "AB":
            vals={}
            for v in ("O","F"):
                for q in OTHER:
                    r=bank[pid,side,v,q]
                    assert (r["prompt"],r["image_id"])==(p["prompt_"+q],p["image_"+side])
                    vals[v,q]=float(r["score"])
                    assert np.isfinite(vals[v,q]) and float(np.float16(vals[v,q]))==vals[v,q]
                assert bank[pid,side,v,"left"]["tensor_sha256"]==bank[pid,side,v,"right"]["tensor_sha256"]
            photos.append(vals)
            score_cache.append((pid,side,p["image_"+side],vals))
        outcomes=decisions(*photos)
        row=dict(pair_id=pid,image_A=p["image_A"],image_B=p["image_B"],winner_left=p["winner_left"],winner_right=p["winner_right"])
        for st in STRATEGIES:
            choices=[]
            for q in OTHER:
                decision,margin=outcomes[st,q]
                row[st+"_"+q]=decision
                row[st+"_margin_"+q]=margin
                choices.append(decision)
            correct=sum(choice==p["winner_"+q] for choice,q in zip(choices,OTHER))
            row.update({st+"_both":int(correct==2),st+"_question_accuracy":correct/2,st+"_question_tie":choices.count("=")/2,
                        st+"_same_choice":int(choices[0]==choices[1] and choices[0]!="=")})
        results.append(row)
    save_rows(out/"pair_results.csv",results)
    summary=[]
    for st in STRATEGIES:
        item=dict(strategy=st,n_pairs=n,n_questions=2*n)
        for m in METRICS:
            denominator=2*n if m.startswith("question") else n
            item[m+"_count"]=round(sum(r[st+"_"+m] for r in results)*denominator/n)
            item[m+"_rate"]=item[m+"_count"]/denominator
        item["question_wrong_count"]=2*n-item["question_accuracy_count"]-item["question_tie_count"]
        summary.append(item)
    save_rows(out/"summary.csv",summary)
    draws=np.random.default_rng(SEED).integers(n,size=(10_000,n))
    delta=[]
    for after,before in (("synchronized","image_only"),("synchronized","original"),("image_only","original")):
        for m in METRICS:
            v=np.array([r[after+"_"+m]-r[before+"_"+m] for r in results],dtype=np.float64)
            lo,hi=np.percentile(v[draws].mean(1)*100,(2.5,97.5))
            delta.append(dict(after=after,before=before,metric=m,diff_pp=100*v.mean(),lo95_pp=lo,hi95_pp=hi,
                pairs_improved=int(sum(v>0)),pairs_decreased=int(sum(v<0)),pairs_unchanged=int(sum(v==0))))
    save_rows(out/"comparisons.csv",delta)
    # Eight distinct photographs drawn without reference to model outcomes.
    keys=np.random.default_rng(SEED).choice(len(score_cache),8,replace=False)
    checks=[]
    for idx in keys:
        pid,side,image_id,original=score_cache[idx]
        mirrored={("F" if v=="O" else "O",q):value for (v,q),value in original.items()}
        scores=decisions(original,mirrored)
        for q in OTHER:
            m=scores["image_only",q][1]
            assert m==0 and scores["image_only",q][0]=="="
            checks.append(dict(pair_id=pid,side=side,image_id=image_id,direction=q,
                o_score=original["O",q],f_score=original["F",q],candidate_1=(original["O",q]+original["F",q])/2,
                candidate_2=(mirrored["O",q]+mirrored["F",q])/2,margin=m,choice="="))
    save_rows(out/"mirror_check.csv",checks)
    return results,summary,delta


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=HERE)
    analyze(ap.parse_args().out)
