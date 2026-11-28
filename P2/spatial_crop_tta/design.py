"""Freeze noun prompts, coordinate maps and box-only strata before opening scores."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "prompt_aware_flip"
sys.path.insert(0, str(SOURCE))
from common import rows, save_json, save_rows, sha

TEMPLATE = "A photo of the {}."
HALVES = ("left","right")
ROLES = ("x","y")


def geometry(w,h):
    # Double both axes by pixel replication. The exact midline is now an integer
    # edge even for odd widths. Each half has size w by 2*h in this grid.
    scale = 224 / max(w,2*h)
    dw,dh = round(w*scale),round(2*h*scale)
    return dict(mid_x=w/2, resized_width=dw,resized_height=dh,
                pad_left=(224-dw)//2,pad_top=(224-dh)//2,
                sx=dw/(w/2),sy=dh/h)


def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    prev=json.loads((SOURCE/"freeze.json").read_text())
    for name in ("pairs.csv","images.csv"):
        assert sha((SOURCE/name).read_bytes())==prev["hashes"][name]
    pairs=rows(SOURCE/"pairs.csv")
    images=rows(SOURCE/"images.csv")
    bank={(r["pair_id"],r["side"]):r for r in images}
    assert len(pairs)==len({p["pair_id"] for p in pairs})==64
    assert len(bank)==len(images)==len({r["image_id"] for r in images})==128
    names=sorted({p[role] for p in pairs for role in ROLES})
    noun_prompts={name:TEMPLATE.format("TV" if name=="tv" else name) for name in names}
    score_inputs=[]; strata=[]; smoke=[]
    for p in pairs:
        pid=p["pair_id"]
        if len(smoke)<2 and all(bank[pid,s]["license_id"] not in ("3","6") for s in "AB"):
            smoke.append(pid)
        for side in "AB":
            img=bank[pid,side]
            w,h=int(img["width"]),int(img["height"])
            assert p["image_"+side]==img["image_id"]
            ox,oy,fw,fh=json.loads(img["valid_xywh"])
            mid=Fraction(w,2)
            g=geometry(w,h)
            score_inputs.append(dict(pair_id=pid,side=side,image_id=img["image_id"],file=img["file"],
                jpeg_sha256=img["jpeg_sha256"],x=p["x"],y=p["y"],prompt_x=noun_prompts[p["x"]],
                prompt_y=noun_prompts[p["y"]],width=w,height=h,**g,old_content_xywh=img["valid_xywh"],
                old_input_mid=ox+fw/2,left_x0=0,right_x0=float(mid)))
            fractions={}; edges={}
            for role in ROLES:
                x,_,bw,_=json.loads(img[role+"_original_xywh"],parse_float=Fraction)
                l,r=max(0,x),min(w,x+bw)
                assert l<r
                fractions[role]=Fraction(max(0,min(mid,r)-l),r-l)
                edges[role]=(l,r)
                a,_,b,_=json.loads(img[role+"_input_xyxy"])
                assert abs(float(fractions[role])-max(0,min(ox+fw/2,b)-a)/(b-a)) < 1e-10
            assert edges["x"][1]<=edges["y"][0] or edges["y"][1]<=edges["x"][0]
            x_left=edges["x"][1]<=edges["y"][0]
            assert x_left==(p["winner_left"]==side) and p["winner_right"]!=p["winner_left"]
            fx,fy=fractions["x"],fractions["y"]
            clear_l=fx>=Fraction(3,4) and 1-fy>=Fraction(3,4)
            clear_r=1-fx>=Fraction(3,4) and fy>=Fraction(3,4)
            assert not clear_l or x_left
            assert not clear_r or not x_left
            strata.append(dict(pair_id=pid,side=side,image_id=img["image_id"],true_relation="left" if x_left else "right",
                x_left_fraction=float(fx),x_right_fraction=float(1-fx),y_left_fraction=float(fy),
                y_right_fraction=float(1-fy),clear=int(clear_l or clear_r),
                x_expected_half="left" if clear_l else "right" if clear_r else "",
                y_expected_half="right" if clear_l else "left" if clear_r else ""))
    save_rows(out/"views.csv",score_inputs)
    save_rows(out/"strata.csv",strata)
    groups=Counter(sum(r["clear"] for r in strata if r["pair_id"]==p["pair_id"]) for p in pairs)
    meta=dict(noun_template=TEMPLATE,nouns=noun_prompts,smoke_pairs=smoke,
        source_hashes={n:prev["hashes"][n] for n in ("pairs.csv","images.csv")},
        hashes={n:sha((out/n).read_bytes()) for n in ("strata.csv","views.csv")},
        cutoff=0.75,pairs_by_clear_count=dict(sorted(groups.items())),clear_images=sum(r["clear"] for r in strata),
        bootstrap=dict(unit="natural pair",replicates=10000,seed=42,interval="percentile 95%"),
        case_rule="Smallest ID with both pictures permitting crop redistribution in each category; prefer a new pair for the second category.")
    save_json(out/"freeze.json",meta)
    return meta


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=HERE)
    a=p.parse_args()
    m=freeze(a.out)
    print("Clear photos:",m["clear_images"],"strata:",m["pairs_by_clear_count"],"smoke:",m["smoke_pairs"])
