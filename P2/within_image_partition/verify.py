"""Independent integer replay of FP16 dot products and all primary contrasts."""
import argparse,json
from fractions import Fraction
from pathlib import Path
from design import HERE,KEPT,ROLES,LINES,rows,save_json,digest

def audit(out):
    raw=rows(out/"scores.csv");scale=1<<24
    bank={}
    for r in raw:
        val=Fraction(r["score"])*scale
        assert val.denominator==1
        key=(r["image_id"],int(r["line"]),r["kept"],r["role"])
        assert key not in bank
        bank[key]=int(val)
    photos=rows(HERE/"photos.csv");comparisons=rows(out/"primary_photos.csv")
    primary={r["image_id"]:r for r in comparisons}
    truth={r["pair_id"]:r["winner_left"] for r in rows(HERE.parent/"prompt_aware_flip/pairs.csv")}
    results={(r["image_id"],int(r["line"])):r for r in rows(out/"image_conditions.csv")}
    mids={r["pair_id"]:r for r in rows(out/"midline_pairs.csv")}
    classifies={(r["image_id"],int(r["line"])):r for r in rows(HERE/"groups.csv")}
    groupcounts={};local={};interactions=[]
    for p in photos:
        fid=p["image_id"];pid=p["pair_id"];sgn=1 if p["side"]==truth[pid] else -1
        for line in LINES:
            group=classifies[fid,line]
            x,y=map(float,(group["x_left"],group["y_left"]))
            label="partial"
            if min(x,1-y)>=.75 or min(1-x,y)>=.75:label="split"
            elif min(x,y)>=.75 or min(1-x,1-y)>=.75:label="same"
            assert label==group["group"]
            groupcounts[fid,line]=label
            for name,(x,y) in (("true",ROLES[:2]),("alt",ROLES[2:])):
                # Two object assignments of nouns to visible sides on the same photo.
                e=(bank[fid,line,"left",x]+bank[fid,line,"right",y])-(bank[fid,line,"right",x]+bank[fid,line,"left",y])
                local[fid,line,name]=sgn*e
                assert sgn*e==float(results[fid,line]["signed_"+name])*scale
        l,r=groupcounts[fid,25],groupcounts[fid,75]
        if {l,r}=={"split","same"}:
            c=primary[fid];a,b=(25,75) if l=="split" else (75,25)
            d=[local[fid,a,name]-local[fid,b,name] for name in ("true","alt")]
            assert d[0]-d[1]==float(c["interaction"])*scale
            interactions.append(d[0]-d[1])
        else:assert fid not in primary
    for pid,r in mids.items():
        selected=[p["image_id"] for p in photos if p["pair_id"]==pid]
        assert len(selected)==2
        for name in ("true","alt"):assert sum(local[i,50,name] for i in selected)==float(r["mask_"+name])*scale
    summary=json.loads((out/"summary.json").read_text())
    exact=Fraction(sum(interactions),len(interactions)*scale)
    assert float(exact)==summary["groups"]["all"]["effects"]["interaction"]["mean"]
    env=json.loads((out/"environment.json").read_text())
    runs=json.loads((out/"runs.json").read_text())
    assert sum(x["visual_forwards"] for x in runs)==768 and sum(x["new_scores"] for x in runs)==3072
    assert env["freeze_sha"]==digest(HERE/"freeze.json") and env["runner_sha"]==digest(HERE/"score_views.py")
    assert len(bank)==3072 and len(results)==384 and len(primary)==64 and len(mids)==64
    save_json(out/"verification.json",dict(unique_scores=len(bank),image_line_conditions=len(results),checked_signed_evidence=len(local),
        checked_photo_interactions=len(interactions),primary_exact_mean=str(exact),midline_assignments_checked=2*len(mids),
        gpu=env["gpu"],protocol_precedes_inference=True))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=HERE);audit(ap.parse_args().out)
