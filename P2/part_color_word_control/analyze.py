"""Offline paired analysis of the true-color cache and the new neutral scores."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from color_control import ROOT, OLD, QUERIES, VIEWS, check_scores, frozen, read, write, as_json, digest

CONDITIONS=("neutral","color")
ROLES=("matched","swapped")
STAGES=("crop","tta")

def state(margin):
    if margin==0:return "tie"
    return "agree" if margin>0 else "opposite"

def prepare_scores(out,photos):
    tables={k:check_scores(rows,photos,k=="neutral") for k,rows in [
        ("neutral",read(out/"neutral_scores.csv")),
        ("color",[s for s in read(OLD/"scores.csv") if s["view"] in VIEWS])]}
    for key,c in tables["color"].items():
        n=tables["neutral"][key]
        assert n["tensor_sha256"]==c["tensor_sha256"] and n["token_sha256"]!=c["token_sha256"]
    for mode in CONDITIONS:
        for s in tables[mode].values():
            x=float(s["score"])
            assert np.isfinite(x) and float(np.float16(x))==x
    return tables

def question(a,b,q,scores,bundle):
    assert {a[q+"_present"],b[q+"_present"]}=={"0","1"}
    sign=1 if a[q+"_present"]=="1" else -1
    r=dict(pair_id=a["pair_id"],species=a["species"],color=a["color"],query=q,image_A=a["image_id"],
           image_B=b["image_id"],gold="A" if sign==1 else "B")
    for condition in CONDITIONS:
        t=scores[condition]
        def s(photo,v):return float(t[photo["image_id"],v,q]["score"])
        d={v:sign*(s(a,v)-s(b,v)) for v in VIEWS}
        r[condition+"_original_d"]=d["original"]
        for role in ROLES:
            added=q if role=="matched" else next(v for v in QUERIES if v!=q)
            for stage in STAGES:
                stem=f"{condition}_{stage}_{role}"
                if stage=="crop":av,bv=s(a,added),s(b,added);margin=d[added]
                else:
                    av,bv=[(s(p,"original")+s(p,added))/2 for p in (a,b)]
                    margin=(d["original"]+d[added])/2
                assert sign*(av-bv)==margin
                if stage=="tta" and condition=="color":assert margin==float(bundle[role+"_"+q])
                r[stem+"_d"]=margin;r[stem+"_status"]=state(margin)
                r[stem+"_pick"]="A" if av>bv else "B" if av<bv else "tie"
        for stage in STAGES:
            r[f"{condition}_{stage}_adv"]=r[f"{condition}_{stage}_matched_d"]-r[f"{condition}_{stage}_swapped_d"]
    for role in ROLES:
        r[role+"_word_gain"]=r["color_crop_"+role+"_d"]-r["neutral_crop_"+role+"_d"]
    r["I"]=r["matched_word_gain"]-r["swapped_word_gain"]
    r["I_tta"]=r["color_tta_adv"]-r["neutral_tta_adv"]
    assert r["I"]==r["color_crop_adv"]-r["neutral_crop_adv"]==2*r["I_tta"]
    return r

def infer_mean(pair_results):
    by=defaultdict(list)
    for r in pair_results:by[r["species"]].append(r["pair_I"])
    order=sorted(by)
    assert len(order)==76
    sizes=np.array([len(by[key]) for key in order])
    totals=np.array([sum(by[key]) for key in order],dtype=np.float64)
    rng=np.random.default_rng(20261128)
    indices=rng.integers(len(order),size=(10000,len(order)))
    sims=totals[indices].sum(axis=1)/sizes[indices].sum(axis=1)
    return dict(mean=float(totals.sum()/sizes.sum()),ci95=np.quantile(sims,[.025,.975]).tolist(),
                clusters=len(order),replicates=len(sims),seed=20261128)

def run(out):
    frozen()
    photos=read(OLD/"pairs.csv");scores=prepare_scores(out,photos)
    assert len(photos)==len({p["image_id"] for p in photos})==192
    pairs=defaultdict(dict)
    for p in photos:pairs[p["pair_id"]][p["side"]]=p
    bundle={r["pair_id"]:r for r in read(OLD/"results/pair_results.csv")}
    assert len(pairs)==96 and set(pairs)==set(bundle)
    questions=[];pair_results=[]
    for pid,ab in sorted(pairs.items()):
        a,b=ab["A"],ab["B"]
        assert a["species"]==b["species"] and a["color"]==b["color"]
        crown,wing=[question(a,b,q,scores,bundle[pid]) for q in QUERIES]
        assert crown["gold"]!=wing["gold"]
        questions.extend([crown,wing])
        pair_results.append(dict(pair_id=pid,species=a["species"],color=a["color"],image_A=a["image_id"],image_B=b["image_id"],
                                 crown_I=crown["I"],wing_I=wing["I"],pair_I=(crown["I"]+wing["I"])/2))
    write(out/"questions.csv",questions);write(out/"pairs_results.csv",pair_results)
    table=[];moves=[];means=[]
    for group in ("both",)+QUERIES:
        rows=[r for r in questions if group=="both" or r["query"]==group]
        means.append(dict(query=group,n=len(rows),**{k:np.mean([r[k] for r in rows]) for k in
            ("matched_word_gain","swapped_word_gain","neutral_crop_adv","color_crop_adv","neutral_tta_adv","color_tta_adv","I","I_tta")},
            I_positive=sum(r["I"]>0 for r in rows),I_zero=sum(r["I"]==0 for r in rows),I_negative=sum(r["I"]<0 for r in rows)))
        for role in ROLES:
            for stage in STAGES:
                for condition in CONDITIONS:
                    k=f"{condition}_{stage}_{role}_status"
                    counts=Counter(r[k] for r in rows)
                    table.append(dict(query=group,stage=stage,role=role,prompt=condition,n=len(rows),
                                      agree=counts["agree"],tie=counts["tie"],opposite=counts["opposite"]))
                trans=Counter((r[f"neutral_{stage}_{role}_status"],r[f"color_{stage}_{role}_status"]) for r in rows)
                assert sum(trans.values())==len(rows)
                for before in ("agree","opposite","tie"):
                    for after in ("agree","opposite","tie"):
                        moves.append(dict(query=group,stage=stage,role=role,before=before,after=after,n=trans[before,after]))
    reproduction={(r["stage"],r["role"]):r["agree"] for r in table if r["query"]=="both" and r["prompt"]=="color"}
    assert reproduction=={("crop","matched"):94,("crop","swapped"):94,("tta","matched"):99,("tta","swapped"):95}
    write(out/"decision_counts.csv",table);write(out/"transitions.csv",moves);write(out/"means.csv",means)
    low=min(pair_results,key=lambda r:(r["pair_I"],r["pair_id"]))
    high=min(pair_results,key=lambda r:(-r["pair_I"],r["pair_id"]))
    summary=dict(primary=infer_mean(pair_results),question_means={r["query"]:r["I"] for r in means},
                 extrema={"max":high["pair_id"],"min":low["pair_id"]},pairs=len(pairs),questions=len(questions),
                 algebraic_checks=len(questions),max_abs_tta_residual=max(abs(r["I"]/2-r["I_tta"]) for r in questions),
                 true_scores_read=len(scores["color"]),neutral_scores_read=len(scores["neutral"]),
                 distinct_matched_input_hashes=len({s["tensor_sha256"] for s in scores["neutral"].values()}),
                 matched_legacy_tta_margins=2*len(questions),neutral_sha256=digest(out/"neutral_scores.csv"))
    assert summary["primary"]["mean"]==means[0]["I"]
    as_json(out/"summary.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT);run(p.parse_args().out)
