"""Compute signed single-photo evidence and rerun the old paired assignment."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import numpy as np
from design import HERE,SOURCE,NOUN,LINES,ROLES,KEPT,rows,save_rows,save_json,digest,noun_prompt

def outcome(value):
    return "correct" if value>0 else "reversed" if value<0 else "tie"

def tally(values):
    counts=Counter(outcome(v) for v in values)
    return {k:counts[k] for k in ("correct","reversed","tie")}

def transitions(before,after):
    counts=Counter((outcome(a),outcome(b)) for a,b in zip(before,after))
    return {f"{a}_to_{b}":counts[a,b] for a,b in product(("correct","reversed","tie"),repeat=2)}

def clustered(rows_,fields):
    pairs=sorted({r["pair_id"] for r in rows_})
    by=[[r for r in rows_ if r["pair_id"]==p] for p in pairs]
    sizes=np.array(list(map(len,by)))
    sums=np.array([[sum(r[k] for r in group) for k in fields] for group in by])
    counts=np.random.default_rng(42).multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=10000)
    draws=(counts@sums)/(counts@sizes)[:,None]
    return {k:dict(mean=sums[:,j].sum()/len(rows_),ci95=np.quantile(draws[:,j],[.025,.975]).tolist()) for j,k in enumerate(fields)}

def native(row):
    v=float(row["score"])
    assert np.isfinite(v) and float(np.float16(v))==v
    return v

def evidence(score,role_x,role_y):
    left_x,right_x=score("left",role_x),score("right",role_x)
    left_y,right_y=score("left",role_y),score("right",role_y)
    e=(left_x-right_x)+(right_y-left_y)
    assert e==(left_x+right_y)-(right_x+left_y)
    return e

def run(out):
    frozen=json.loads((HERE/"freeze.json").read_text())
    assert all(digest(HERE/f)==h for f,h in frozen["hashes"].items())
    photos=rows(HERE/"photos.csv"); groups={(r["image_id"],int(r["line"])):r["group"] for r in rows(HERE/"groups.csv")}
    pbank={r["image_id"]:r for r in photos}
    old=HERE.parent/"spatial_crop_tta"
    source=rows(out/"scores.csv");bank={};tensor={};token={}
    for r in source:
        fid=r["image_id"]; line=int(r["line"]);half=r["kept"];role=r["role"];p=pbank[fid]
        assert line in LINES and half in KEPT and role in ROLES
        assert (p["pair_id"],p["side"],p[role],noun_prompt(p[role]))==(r["pair_id"],r["side"],r["noun"],r["prompt"])
        assert tensor.setdefault((fid,line,half),r["tensor_sha"])==r["tensor_sha"]
        assert token.setdefault(r["prompt"],r["token_sha"])==r["token_sha"]
        bank[fid,line,half,role]=native(r)
    assert len(bank)==len(source)==3072 and len(tensor)==768
    assert len(groups)==384
    # Check the smoke values and previous tokenizer signatures.
    smoke=rows(out/"smoke_scores.csv")
    assert len(smoke)==48 and all(bank[r["image_id"],int(r["line"]),r["kept"],r["role"]]==native(r) for r in smoke)
    prior={}
    for r in rows(old/"scores.csv"):
        assert token[r["prompt"]]==r["token_sha256"]
        prior[r["pair_id"],r["side"],r["half"],r["role"]]=native(r)
    for r in rows(NOUN/"scores.csv"):
        if r["seed"]=="42":
            assert token[noun_prompt(r["noun"])]==r["token_sha"]
            prior[r["pair_id"],r["side"],r["half"],"alt_"+r["role"]]=native(r)
    assert len(prior)==1024
    detail=[];primary=[];central={}
    for p in photos:
        fid=p["image_id"];pid=p["pair_id"];side=p["side"];sign=int(p["sign"]);ev={}
        for line in LINES:
            score=lambda h,k:bank[fid,line,h,k]
            t=evidence(score,"x","y");a=evidence(score,"alt_x","alt_y")
            ev[line]=(sign*t,sign*a)
            detail.append(dict(pair_id=pid,side=side,image_id=fid,line=line,group=groups[fid,line],main=p["main"],
                sign=sign,true_E=t,alt_E=a,signed_true=sign*t,signed_alt=sign*a,true_outcome=outcome(sign*t),alt_outcome=outcome(sign*a)))
        if p["main"]=="1":
            split,same=int(p["split_line"]),int(p["same_line"])
            assert groups[fid,split]=="split" and groups[fid,same]=="same" and {split,same}=={25,75}
            dt,da=np.subtract(ev[split],ev[same])
            primary.append(dict(pair_id=pid,side=side,image_id=fid,x=p["x"],y=p["y"],alt_x=p["alt_x"],alt_y=p["alt_y"],
                split_line=split,same_line=same,sign=sign,same_true=ev[same][0],split_true=ev[split][0],
                same_alt=ev[same][1],split_alt=ev[split][1],delta_true=dt,delta_alt=da,interaction=dt-da))
        get_old=lambda h,k:prior[pid,side,h,k]
        central[pid,side]=dict(mask_true=ev[50][0],mask_alt=ev[50][1],
             crop_true=sign*evidence(get_old,"x","y"),crop_alt=sign*evidence(get_old,"alt_x","alt_y"))
    assert len(primary)==64 and len({r["pair_id"] for r in primary})==48
    save_rows(out/"image_conditions.csv",detail);save_rows(out/"primary_photos.csv",primary)
    summaries={}
    for name,group in (("all",primary),("split25",[r for r in primary if r["split_line"]==25]),("split75",[r for r in primary if r["split_line"]==75])):
        stats=dict(photos=len(group),pairs=len({r["pair_id"] for r in group}),effects=clustered(group,("delta_true","delta_alt","interaction")))
        for noun in ("true","alt"):
            before=[r["same_"+noun] for r in group];after=[r["split_"+noun] for r in group]
            stats[noun]=dict(same=tally(before),split=tally(after),transitions=transitions(before,after),
                same_mean=sum(before)/len(group),split_mean=sum(after)/len(group))
        summaries[name]=stats
    middle=[]
    legacy={r["pair_id"]:r for r in rows(NOUN/"pair_results.csv") if r["seed"]=="42"}
    for pair in rows(SOURCE/"pairs.csv"):
        pid=pair["pair_id"]; r=dict(pair_id=pid,true_left=pair["winner_left"])
        for k in ("crop_true","crop_alt","mask_true","mask_alt"):
            r[k]=central[pid,"A"][k]+central[pid,"B"][k]
            r[k+"_outcome"]=outcome(r[k])
        assert r["crop_true"]==float(legacy[pid]["true"]) and r["crop_alt"]==float(legacy[pid]["both_alt"])
        for method in ("crop","mask"):
            r[method+"_advantage_pp"]=100*(int(r[method+"_true"]>0)-int(r[method+"_alt"]>0))
        r["change_advantage_pp"]=r["mask_advantage_pp"]-r["crop_advantage_pp"]
        r["change_true_pp"]=100*(int(r["mask_true"]>0)-int(r["crop_true"]>0))
        middle.append(r)
    save_rows(out/"midline_pairs.csv",middle)
    middle_summary={k:tally([r[k] for r in middle]) for k in central[pid,"A"]}
    middle_summary["contrasts"]=clustered(middle,("mask_advantage_pp","crop_advantage_pp","change_advantage_pp","change_true_pp"))
    middle_summary["true_transitions"]=transitions([r["crop_true"] for r in middle],[r["mask_true"] for r in middle])
    examples=[]
    for label,s in (("improved",1),("worsened",-1)):
        chosen=next((r for r in primary if pbank[r["image_id"]]["license_id"]=="4" and s*r["delta_true"]>0),None)
        if chosen:examples.append(dict(example=label,**chosen))
    if examples:save_rows(out/"cases.csv",examples)
    save_json(out/"summary.json",dict(groups=summaries,midline=middle_summary,examples=examples,
        scored_photos=len(pbank),score_count=len(bank),visual_inputs=len(tensor),
        old_crop_scores_used=len(prior),old_source_hashes={str(p.parent.name)+"/"+p.name:digest(p) for p in [old/"scores.csv",NOUN/"scores.csv",NOUN/"pair_results.csv"]},
        freeze_sha=digest(HERE/"freeze.json"),scores_sha=digest(out/"scores.csv")))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=HERE);run(ap.parse_args().out)
