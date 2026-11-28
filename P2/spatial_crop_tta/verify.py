"""Independent checks of the score algebra and optional replay of all image tensors on CPU."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

from design import HERE, SOURCE, rows, save_json, sha


def verify(out):
    new={(r["pair_id"],r["side"],r["half"],r["role"]):Fraction(r["score"]) for r in rows(out/"scores.csv")}
    old={(r["pair_id"],r["side"],r["view"],r["direction"]):Fraction(r["score"]) for r in rows(SOURCE/"scores.csv")}
    assert len(old)==len(new)==512
    details=rows(out/"pair_results.csv")
    layers=rows(out/"image_results.csv")
    assert len({r["pair_id"] for r in details})==len(details)==64
    assert len({r["image_id"] for r in layers})==len(layers)==128
    for r in details:
        i=r["pair_id"]
        L,R=r["true_left"],r["true_right"]
        def whole(v):
            return old[i,L,v,"left"]+old[i,R,v,"right"]-old[i,R,v,"left"]-old[i,L,v,"right"]
        def local(s,xhalf,yhalf):
            return new[i,s,xhalf,"x"]+new[i,s,yhalf,"y"]
        preferred=local(L,"left","right")+local(R,"right","left")
        swapped=local(L,"right","left")+local(R,"left","right")
        for st,num in (("original",whole("O")),("global4",(whole("O")-whole("F"))/2),("local",preferred-swapped)):
            assert num==Fraction(r[st+"_margin"])
            assert r[st]==("correct" if num>0 else "reversed" if num<0 else "tie")
    cats=Counter()
    for r in layers:
        if r["clear"]=="0":
            assert r["localization"]==""
            continue
        matches=ties=0
        for role in ("x","y"):
            good=r[role+"_expected_half"]
            bad="right" if good=="left" else "left"
            better=new[r["pair_id"],r["side"],good,role]-new[r["pair_id"],r["side"],bad,role]
            matches+=better>0
            ties+=better==0
        cat="has_tie" if ties else ("both_wrong","one_correct","both_correct")[matches]
        assert cat==r["localization"]
        cats[cat]+=1
    for r in rows(out/"localization_summary.csv"):
        assert cats[r["localization"]]==int(r["n"])
    smoke=rows(out/"smoke_scores.csv"); final=rows(out/"scores.csv")
    assert len(smoke)==16 and smoke==final[:16]
    runs=json.loads((out/"runs.json").read_text())
    assert sum(r["new_scores"] for r in runs)==512 and sum(r["new_visual_forwards"] for r in runs)==256
    return dict(natural_pairs=64,independent_exact_margins=192,verified_scores=len(new),
                full_reference_scores=len(old),smoke_scores_reused=len(smoke),noun_outcomes_checked=sum(cats.values()))


def replay_images(out,photos):
    import numpy as np
    import torch
    from PIL import Image
    from hpsv2.src.open_clip.transform import image_transform
    tf=image_transform(224,is_train=False,resize_longest_max=True)
    norm=tf.transforms[-1].normalize
    mean=torch.tensor(norm.mean)[:,None,None]; std=torch.tensor(norm.std)[:,None,None]
    scores=rows(out/"scores.csv")
    hashes={(r["pair_id"],r["side"],r["half"]):r["tensor_sha256"] for r in scores}
    smoke_ids=json.loads((out/"freeze.json").read_text())["smoke_pairs"]
    count=pngs=0
    for r in rows(out/"views.csv"):
        path=photos/r["file"]
        assert sha(path.read_bytes())==r["jpeg_sha256"]
        with Image.open(path) as im:
            pixels=np.asarray(im.convert("RGB"))
        h,w=pixels.shape[:2]
        # Independent integer index mapping: original x = floor((start+j)/2).
        for k,half in enumerate(("left","right")):
            indices=(k*w+np.arange(w))//2
            local=pixels[(np.arange(2*h)//2)[:,None],indices]
            tensor=tf(Image.fromarray(local)).contiguous()
            assert sha(tensor.numpy().tobytes())==hashes[r["pair_id"],r["side"],half]
            if r["pair_id"] in smoke_ids:
                a=((tensor*std+mean).permute(1,2,0).numpy()*255).round().clip(0,255).astype("uint8")
                with Image.open(out/"smoke"/f'{r["pair_id"]}_{r["side"]}_{half}.png') as im:
                    assert np.array_equal(a,np.asarray(im))
                pngs+=1
            count+=1
    assert count==256 and pngs==8
    return dict(cpu_replayed_tensor_hashes=count,smoke_pngs_pixel_exact=pngs)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=HERE)
    p.add_argument("--photos",type=Path,help="Replay the official transform on CPU in the HPS environment")
    args=p.parse_args()
    stats=verify(args.out)
    if args.photos:
        stats.update(replay_images(args.out,args.photos))
    if args.photos or not (args.out/"verification.json").exists():
        save_json(args.out/"verification.json",stats)
    print(json.dumps(stats))
