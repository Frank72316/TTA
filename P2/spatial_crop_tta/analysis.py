"""Rebuild the three assignments, paired transitions, strata and noun diagnostics."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import struct
import numpy as np

from design import HERE, SOURCE, ROLES, HALVES, rows, save_json, save_rows, sha

METHODS=("original","global4","local")
OUTCOMES=("correct","reversed","tie")


def label(m):
    return "correct" if m>0 else "reversed" if m<0 else "tie"


def native(row):
    f=float(row["score"])
    assert np.isfinite(f) and struct.unpack("e",struct.pack("e",f))[0]==f
    return f


def transitions(detail,baseline):
    return [dict(reference=baseline,before=before,after=after,n=len(ids),ids=" ".join(ids))
            for before,after in product(OUTCOMES,repeat=2)
            for ids in [[r["pair_id"] for r in detail if (r[baseline],r["local"])==(before,after)]]]


def analyze(root):
    freeze=json.loads((root/"freeze.json").read_text())
    assert all(sha((root/n).read_bytes())==h for n,h in freeze["hashes"].items())
    assert all(sha((SOURCE/n).read_bytes())==h for n,h in freeze["source_hashes"].items())
    pairs=rows(SOURCE/"pairs.csv")
    photos=rows(root/"views.csv")
    views={(r["pair_id"],r["side"]):r for r in photos}
    layers={(r["pair_id"],r["side"]):r for r in rows(root/"strata.csv")}
    assert set(layers)==set(views)==set(product((p["pair_id"] for p in pairs),"AB"))
    old_scores=rows(SOURCE/"scores.csv"); new_scores=rows(root/"scores.csv")
    old={(r["pair_id"],r["side"],r["view"],r["direction"]):native(r) for r in old_scores}
    new={(r["pair_id"],r["side"],r["half"],r["role"]):native(r) for r in new_scores}
    assert len(old_scores)==len(new_scores)==len(old)==len(new)==512
    prior={r["pair_id"]:r for r in rows(HERE.parent/"direction_evidence/pair_results.csv")}
    by_id={p["pair_id"]:p for p in pairs}
    for r in old_scores:
        p=by_id[r["pair_id"]]
        assert (r["image_id"],r["prompt"])==(p["image_"+r["side"]],p["prompt_"+r["direction"]])
    token_sets={}; view_sets={}
    for r in new_scores:
        pid,side,role=r["pair_id"],r["side"],r["role"]
        v=views[pid,side]
        assert (r["image_id"],r["noun"],r["prompt"])==(v["image_id"],v[role],v["prompt_"+role])
        assert token_sets.setdefault(r["prompt"],r["token_sha256"])==r["token_sha256"]
        assert view_sets.setdefault((pid,side,r["half"]),r["tensor_sha256"])==r["tensor_sha256"]
    details=[]; location=[]
    for p in pairs:
        i=p["pair_id"]
        evidence={}
        for side in "AB":
            s=layers[i,side]
            o=old[i,side,"O","left"]-old[i,side,"O","right"]
            f=old[i,side,"F","left"]-old[i,side,"F","right"]
            lx,ly,rx,ry=(new[i,side,h,k] for h,k in product(HALVES,ROLES))
            cx,cy=lx-rx,ry-ly
            e=cx+cy
            evidence[side]=dict(original=o,global4=(o-f)/2,local=e)
            sign=1 if s["true_relation"]=="left" else -1
            assert (sign==1)==(p["winner_left"]==side)
            clear=int(s["clear"])
            ax,ay=sign*cx,sign*cy
            loc=""
            if clear:
                assert (s["x_expected_half"],s["y_expected_half"])==(("left","right") if sign==1 else ("right","left"))
                loc="has_tie" if ax==0 or ay==0 else ("both_wrong","one_correct","both_correct")[int(ax>0)+int(ay>0)]
            location.append(dict(**s,x=p["x"],y=p["y"],left_x=lx,left_y=ly,right_x=rx,right_y=ry,
                x_localization=label(ax) if clear else "",y_localization=label(ay) if clear else "",
                localization=loc,x_evidence=cx,y_evidence=cy,E=e,E_truth=sign*e,image_decision=label(sign*e),
                original_d=o,mirror_d=f,global4_evidence=(o-f)/2))
        left,right=p["winner_left"],p["winner_right"]
        detail=dict(pair_id=i,x=p["x"],y=p["y"],image_A=p["image_A"],image_B=p["image_B"],true_left=left,true_right=right,
                    n_clear=sum(int(layers[i,s]["clear"]) for s in "AB"))
        for method in METHODS:
            margin=evidence[left][method]-evidence[right][method]
            detail.update({method+"_A":evidence["A"][method],method+"_B":evidence["B"][method],
                           method+"_margin":margin,method:label(margin)})
        assert detail["original_margin"]==float(prior[i]["M"]) and (left,right)==(prior[i]["true_left"],prior[i]["true_right"])
        details.append(detail)
    save_rows(root/"image_results.csv",location)
    save_rows(root/"pair_results.csv",details)
    summary=[]
    for scope in ("all","2","1","0"):
        group=[r for r in details if scope=="all" or str(r["n_clear"])==scope]
        for st in METHODS:
            counts=Counter(r[st] for r in group)
            summary.append(dict(clear_images_in_pair=scope,method=st,n=len(group),
                **{k:counts[k] for k in OUTCOMES},accuracy=counts["correct"]/len(group) if group else ""))
    assert [r["original"] for r in details].count("correct")==30 and [r["original"] for r in details].count("reversed")==34
    save_rows(root/"summary.csv",summary)
    moves=transitions(details,"global4")+transitions(details,"original")
    save_rows(root/"transitions.csv",moves)
    delta=np.array([int(r["local"]=="correct")-int(r["global4"]=="correct") for r in details])
    b=freeze["bootstrap"]
    sample=np.random.default_rng(b["seed"]).choice(delta,size=(b["replicates"],len(delta)),replace=True)
    lo,hi=np.percentile(sample.mean(axis=1)*100,[2.5,97.5])
    strata_counts=Counter(r["n_clear"] for r in details)
    assert {str(k):v for k,v in strata_counts.items()}==freeze["pairs_by_clear_count"]
    clear=[r for r in location if int(r["clear"])]
    assert len(clear)==freeze["clear_images"]
    groups=[]
    for level in ("both_correct","one_correct","both_wrong","has_tie"):
        group=[r for r in clear if r["localization"]==level]
        groups.append(dict(localization=level,n=len(group),
            **{"E_"+name:sum(r["image_decision"]==name for r in group) for name in OUTCOMES}))
    save_rows(root/"localization_summary.csv",groups)
    stats=dict(pairs=len(details),photos=len(location),old_scores=len(old),new_scores=len(new),unique_new_tensors=len(set(view_sets.values())),
        unique_noun_tokens=len(set(token_sets.values())),clear_photos=len(clear),other_photos=len(location)-len(clear),
        x_localized=sum(r["x_localization"]=="correct" for r in clear),
        y_localized=sum(r["y_localization"]=="correct" for r in clear),
        net_pairs=int(delta.sum()),diff_pp=float(delta.mean()*100),lo95_pp=float(lo),hi95_pp=float(hi),
        bootstrap=b,old_scores_sha=sha((SOURCE/"scores.csv").read_bytes()),new_scores_sha=sha((root/"scores.csv").read_bytes()))
    save_json(root/"analysis.json",stats)
    return stats


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=HERE)
    print(json.dumps(analyze(ap.parse_args().out),indent=2))
