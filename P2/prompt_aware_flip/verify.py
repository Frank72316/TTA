"""Audit the frozen boxes against COCO and the decisions against exact integer sums."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from common import ANNOTATION_SHA, HERE, choose, rows, save_json, sha, verified_freeze


def audit(out,ann_file):
    assert sha(ann_file.read_bytes())==ANNOTATION_SHA
    coco=json.loads(ann_file.read_bytes())
    ann={a["id"]:a for a in coco["annotations"]}
    by_class=defaultdict(list)
    for a in ann.values():
        by_class[a["image_id"],a["category_id"]].append(a)
    photos={i["id"]:i for i in coco["images"]}
    pairs={r["pair_id"]:r for r in rows(out/"pairs.csv")}
    images=rows(out/"images.csv")
    results={r["pair_id"]:r for r in rows(out/"pair_results.csv")}
    scores=rows(out/"scores.csv")
    frozen=verified_freeze(out)
    assert len(pairs)==frozen["selected"] and pairs.keys()==results.keys()
    assert len({i["image_id"] for i in images})==len(images)==2*len(pairs)
    for s in images:
        p=pairs[s["pair_id"]]; side=s["side"]; iid=int(s["image_id"]); photo=photos[iid]
        w,h=photo["width"],photo["height"]
        assert str(iid)==p["image_"+side] and (w,h)==(int(s["width"]),int(s["height"]))
        ox,oy,nw,nh=json.loads(s["valid_xywh"])
        intervals=[]
        for name in ("x","y"):
            a=ann[int(s[name+"_ann"])]
            assert by_class[iid,int(p[name+"_cat"])]==[a] and a["iscrowd"]==0
            x,y,bw,bh=a["bbox"]
            assert [x,y,bw,bh]==json.loads(s[name+"_original_xywh"])
            x1,y1,x2,y2=max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
            assert (x2-x1)*(y2-y1)>=.8*bw*bh
            scaled=(x1/w*nw+ox,y1/h*nh+oy,x2/w*nw+ox,y2/h*nh+oy)
            assert max(abs(u-v) for u,v in zip(scaled,json.loads(s[name+"_input_xyxy"])))<1e-9
            intervals.append((x1/w,x2/w))
        (l,r),(l2,r2)=intervals
        assert r<=l2 or r2<=l
        assert (r<=l2)==(p["winner_left"]==side)
        assert p["winner_left"]!=p["winner_right"] and abs((l+r-l2-r2)/2)>=.2
    denom=max(float(r["score"]).as_integer_ratio()[1] for r in scores)
    ints={}
    for r in scores:
        v=float(r["score"])*denom
        assert v.is_integer()
        k=(r["pair_id"],r["side"],r["view"],r["direction"])
        assert k not in ints
        ints[k]=int(v)
    decisions=0
    for pid,r in results.items():
        for q,other in (("left","right"),("right","left")):
            for st,addon in (("original",None),("image_only",q),("synchronized",other)):
                def value(side):
                    return ints[pid,side,"O",q]+(ints[pid,side,"F",addon] if addon else 0)
                diff=value("A")-value("B")
                assert choose(diff)==r[st+"_"+q]
                assert diff/(denom*(2 if addon else 1))==float(r[st+"_margin_"+q])
                decisions+=1
    checks=rows(out/"mirror_check.csv")
    assert len(checks)==16 and len({r["image_id"] for r in checks})==8
    assert all(float(r["candidate_1"])==float(r["candidate_2"]) and r["choice"]=="=" for r in checks)
    assert len(ints)==8*len(pairs)
    save_json(out/"verification.json",dict(pairs=len(pairs),distinct_originals=len(images),scores=len(ints),
        independently_recomputed_decisions=decisions,decision_mismatches=0,checked_boxes=len(images)*2,
        mirror_check_questions=len(checks),mirror_ties=len(checks)))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=HERE)
    p.add_argument("--annotations",type=Path,required=True)
    a=p.parse_args(); audit(a.out,a.annotations)
