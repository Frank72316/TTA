"""Official HPS v2.1 scoring. Cache one complete pair at a time."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from common import HERE, rows, save_json, save_rows, sha, verified_freeze, legacy_environment


def run(data, out, pilot):
    started=perf_counter()
    sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
    from score import init
    import torch
    freeze=verified_freeze(out)
    pairs=rows(out/"pairs.csv")
    image_rows=rows(out/"images.csv")
    source={(s["pair_id"],s["side"]):s for s in image_rows}
    assert len(pairs)==freeze["selected"] and len(source)==len({s["image_id"] for s in image_rows})==2*len(pairs)
    model,transform,tok,meta=init(pilot,"cuda")
    resize=transform.transforms[2]
    assert type(resize).__name__=="ResizeMaxSize" and resize.max_size==224
    meta.update(gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),freeze_sha=sha((out/"freeze.json").read_bytes()),
        flip_inference_sha=sha(Path(__file__).read_bytes()),torch_image_batch=1,torch_text_batch=1)
    env=out/"environment.json"
    legacy=legacy_environment(meta)
    if env.exists():
        assert json.loads(env.read_text()) in (meta,legacy)
    else:
        save_json(env,meta)
    signature=sha(env.read_bytes())
    compatible={signature}
    if legacy is not None:
        compatible.add(sha((json.dumps(legacy,indent=2,ensure_ascii=False)+"\n").encode()))
    cache=data/"flip_scores"; cache.mkdir(exist_ok=True)
    view_dir=data/"model_views"; view_dir.mkdir(exist_ok=True)
    norm=transform.transforms[-1].normalize
    mean=torch.tensor(norm.mean)[:,None,None]; std=torch.tensor(norm.std)[:,None,None]
    token_ids={k:int(tok([k])[0,1]) for k in ("left","right")}
    assert token_ids["left"]!=token_ids["right"]
    run_info=dict(new_image_forwards=0,new_image_text_scores=0,checkpoint_scores=0,gpu_image_seconds=0.)
    total=[]; checks=[]
    for ix,p in enumerate(pairs):
        pid=p["pair_id"]
        key=cache/(pid+".json")
        if key.exists():
            prev=json.loads(key.read_text())
            assert prev["signature"] in compatible and len(prev["rows"])==8
            total.extend(prev["rows"]); checks.extend(prev["checks"])
            run_info["checkpoint_scores"]+=8
        else:
            prompts={q:p["prompt_"+q] for q in ("left","right")}
            assert prompts["left"].replace(" to the left of "," to the right of ")==prompts["right"]
            token={q:tok([v]) for q,v in prompts.items()}
            changed=torch.where(token["left"][0]!=token["right"][0])[0].tolist()
            assert len(changed)==1 and all(int(token[q][0,changed[0]])==token_ids[q] for q in prompts)
            assert all(torch.count_nonzero(t)<t.numel() for t in token.values())
            local=[]; check=[]
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                text={q:model.encode_text(t.cuda(),normalize=True) for q,t in token.items()}
                for side in "AB":
                    s=source[pid,side]
                    path=data/"val2017"/s["file"]
                    assert sha(path.read_bytes())==s["jpeg_sha256"] and p["image_"+side]==s["image_id"]
                    with Image.open(path) as im:
                        rgb=im.convert("RGB")
                    mirrored=ImageOps.mirror(rgb)
                    assert np.array_equal(np.asarray(mirrored),np.asarray(rgb)[:,::-1])
                    inputs={"O":transform(rgb),"F":transform(mirrored)}
                    assert all(tuple(t.shape)==(3,224,224) for t in inputs.values())
                    ox,oy,w,h=json.loads(s["valid_xywh"])
                    # Measure the actual padded support on a constant positive canvas.
                    mask=resize(torch.ones((3,rgb.height,rgb.width)))[0]>.5
                    expected=torch.zeros((224,224),dtype=torch.bool)
                    expected[oy:oy+h,ox:ox+w]=True
                    assert torch.equal(mask,expected)
                    orig=inputs["O"][:,oy:oy+h,ox:ox+w]
                    flip=inputs["F"][:,oy:oy+h,ox:ox+w]
                    error=float((orig.flip([-1])-flip).abs().max())
                    assert error<1e-4,error
                    a,b=(json.loads(s[col+"_input_xyxy"]) for col in ("x","y"))
                    is_left=a[2]<=b[0]
                    assert (is_left or b[2]<=a[0]) and is_left==(p["winner_left"]==side)
                    # Reflection around the centre of the unpadded content.
                    mirror_left=(2*ox+w-a[0]) <= (2*ox+w-b[2])
                    assert mirror_left==(p["winner_right"]==side)
                    check.append(dict(pair_id=pid,side=side,token_index=changed[0],left_token=token_ids["left"],
                          right_token=token_ids["right"],original_x_left=is_left,mirror_x_left=mirror_left,
                          mirror_input_max_abs_error=error,padding_matches_annotation=True))
                    for name,tensor in inputs.items():
                        arr=((tensor*std+mean).permute(1,2,0).numpy()*255).round().clip(0,255).astype("uint8")
                        Image.fromarray(arr).save(view_dir/f"{pid}_{side}_{name}.png")
                        t0=perf_counter()
                        feature=model.encode_image(tensor.unsqueeze(0).cuda(),normalize=True)
                        values={q:(feature@emb.T).squeeze() for q,emb in text.items()}
                        assert all(v.dtype==torch.float16 for v in values.values())
                        vals={q:float(v.item()) for q,v in values.items()}
                        run_info["gpu_image_seconds"]+=perf_counter()-t0
                        run_info["new_image_forwards"]+=1
                        for q,value in vals.items():
                            assert np.isfinite(value)
                            local.append(dict(pair_id=pid,side=side,image_id=s["image_id"],view=name,direction=q,
                                prompt=prompts[q],token_sha256=sha(token[q].numpy().tobytes()),
                                tensor_sha256=sha(tensor.contiguous().numpy().tobytes()),score=value))
                            run_info["new_image_text_scores"]+=1
            save_json(key,dict(signature=signature,rows=local,checks=check))
            total.extend(local); checks.extend(check)
        if ix==1:
            assert len(total)==len({(r["pair_id"],r["side"],r["view"],r["direction"]) for r in total})==16
            assert len(checks)==4
            save_json(out/"smoke.json",dict(pairs=[x["pair_id"] for x in pairs[:2]],scores=16,checks=checks.copy()))
            print("Smoke passed (2 pairs, 16 scores)",flush=True)
    assert len(total)==len({(r["pair_id"],r["side"],r["view"],r["direction"]) for r in total})==8*len(pairs)
    assert len(checks)==2*len(pairs)
    save_rows(out/"scores.csv",total)
    run_info.update(wall_seconds=perf_counter()-started,scored_pairs=len(pairs),
        largest_mirror_input_error=max(v["mirror_input_max_abs_error"] for v in checks))
    log=out/"runs.json"
    save_json(log,(json.loads(log.read_text()) if log.exists() else [])+[run_info])
    print(run_info,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--pilot",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE)
    a=ap.parse_args(); run(a.data,a.out,a.pilot)
