"""Score the two fixed half-images. The scorer receives no boxes or relation labels."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from design import HERE, SOURCE, HALVES, ROLES, geometry, rows, save_json, save_rows, sha


def run(data,pilot,out,smoke_only):
    start=perf_counter()
    freeze=json.loads((out/"freeze.json").read_text())
    assert all(sha((out/n).read_bytes())==v for n,v in freeze["hashes"].items())
    views=rows(out/"views.csv")
    order=freeze["smoke_pairs"]
    grouped=defaultdict(list)
    for r in views:
        grouped[r["pair_id"]].append(r)
    ids=order if smoke_only else order+[i for i in grouped if i not in order]
    sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
    from score import init
    import torch
    model,tf,tok,env=init(pilot,"cuda")
    previous=json.loads((SOURCE/"environment.json").read_text())
    for field in ("hps_version","precision","weights","torch","pillow","preprocess","scorer_sha"):
        assert env[field]==previous[field],field
    env.update(gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],image_batch=1,text_batch=1,
               freeze_sha=sha((out/"freeze.json").read_bytes()),crop_script_sha=sha(Path(__file__).read_bytes()),
               design_script_sha=sha((HERE/"design.py").read_bytes()))
    if (out/"environment.json").exists():
        assert json.loads((out/"environment.json").read_text())==env
    else:
        save_json(out/"environment.json",env)
    key=sha((out/"environment.json").read_bytes())
    cache=out/"cache"; cache.mkdir(exist_ok=True)
    smoke=out/"smoke"; smoke.mkdir(exist_ok=True)
    original_hash={(r["pair_id"],r["side"]):r["tensor_sha256"] for r in rows(SOURCE/"scores.csv")
                   if (r["view"],r["direction"])==("O","left")}
    resize=tf.transforms[2]
    assert type(resize).__name__=="ResizeMaxSize" and resize.max_size==224
    mean=torch.tensor(tf.transforms[-1].normalize.mean)[:,None,None]
    std=torch.tensor(tf.transforms[-1].normalize.std)[:,None,None]

    def display(tensor):
        pixels=((tensor*std+mean).permute(1,2,0).numpy()*255).round().clip(0,255).astype("uint8")
        return Image.fromarray(pixels)

    features={}; token_sha={}; total=[]; checks=[]; hit=0; visual=0; score_seconds=0.0
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for pid in ids:
            file=cache/(pid+".json")
            if file.exists():
                entry=json.loads(file.read_text())
                assert entry["key"]==key and len(entry["scores"])==8
                total.extend(entry["scores"]); checks.extend(entry["checks"]); hit+=8
                continue
            records=[]; evidence=[]
            for src in grouped[pid]:
                side=src["side"]
                photo=data/src["file"]
                assert sha(photo.read_bytes())==src["jpeg_sha256"]
                with Image.open(photo) as im:
                    rgb=im.convert("RGB")
                w,h=rgb.size
                assert (w,h)==(int(src["width"]),int(src["height"]))
                orig=tf(rgb)
                assert sha(orig.numpy().tobytes())==original_hash[pid,side]
                g=geometry(w,h)
                assert all(float(src[k])==v for k,v in g.items())
                doubled=rgb.resize((2*w,2*h),Image.Resampling.NEAREST)
                halves=[doubled.crop((j*w,0,(j+1)*w,2*h)) for j in range(2)]
                if pid in order:
                    assert np.array_equal(np.concatenate([np.asarray(v) for v in halves],axis=1),np.asarray(doubled))
                    (smoke/f"{pid}_{side}_original.jpg").write_bytes(photo.read_bytes())
                    preview=display(orig)
                    x=float(src["old_input_mid"]); _,top,_,fh=json.loads(src["old_content_xywh"])
                    ImageDraw.Draw(preview).line((x,top,x,top+fh-1),fill="#ffbd24",width=2)
                    preview.save(smoke/f"{pid}_{side}_mid.png")
                dw,dh,px,py=(int(g[k]) for k in ("resized_width","resized_height","pad_left","pad_top"))
                support=resize(torch.ones((1,2*h,w)))[0]>.5
                expected=torch.zeros((224,224),dtype=torch.bool); expected[py:py+dh,px:px+dw]=True
                assert torch.equal(support,expected)
                for half,img in zip(HALVES,halves):
                    tensor=tf(img).contiguous()
                    assert tensor.shape==(3,224,224)
                    tensor_sha=sha(tensor.numpy().tobytes())
                    if pid in order:
                        display(tensor).save(smoke/f"{pid}_{side}_{half}.png")
                    for role in ROLES:
                        prompt=src["prompt_"+role]
                        if prompt not in features:
                            tokens=tok([prompt])
                            assert torch.count_nonzero(tokens)<tokens.numel()
                            token_sha[prompt]=sha(tokens.numpy().tobytes())
                            features[prompt]=model.encode_text(tokens.cuda(),normalize=True)
                    tick=perf_counter()
                    image_features=model.encode_image(tensor.unsqueeze(0).cuda(),normalize=True)
                    visual+=1
                    for role in ROLES:
                        prompt=src["prompt_"+role]
                        value=(image_features@features[prompt].T).squeeze()
                        assert value.dtype==torch.float16
                        native=float(value.item())
                        assert np.isfinite(native)
                        records.append(dict(pair_id=pid,side=side,image_id=src["image_id"],half=half,
                            role=role,noun=src[role],prompt=prompt,score=native,
                            tensor_sha256=tensor_sha,token_sha256=token_sha[prompt]))
                    score_seconds+=perf_counter()-tick
                evidence.append(dict(pair_id=pid,side=side,full_tensor_sha256=original_hash[pid,side],
                                     support_xywh=[px,py,dw,dh]))
            assert len(records)==8
            save_json(file,dict(key=key,scores=records,checks=evidence))
            total.extend(records); checks.extend(evidence)
    assert len(total)==len({(r["pair_id"],r["side"],r["half"],r["role"]) for r in total})==8*len(ids)
    save_rows(out/("smoke_scores.csv" if smoke_only else "scores.csv"),total)
    if smoke_only:
        save_json(out/"smoke.json",checks)
    log=out/"runs.json"
    run_stats=dict(end_utc=datetime.now(timezone.utc).isoformat(),smoke_only=smoke_only,records=len(total),
        new_visual_forwards=visual,new_scores=2*visual,checkpoint_score_hits=hit,new_text_forwards=len(features),
        image_scoring_seconds=score_seconds,wall_seconds=perf_counter()-start)
    save_json(log,(json.loads(log.read_text()) if log.exists() else [])+[run_stats])
    print(run_stats,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True,help="Directory of the original val2017 JPEGs")
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE)
    p.add_argument("--smoke-only",action="store_true")
    a=p.parse_args(); run(a.data,a.pilot,a.out,a.smoke_only)
