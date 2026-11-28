"""Encode each verified view once; score only the two color-free texts."""
import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
import torch
from color_control import ROOT,OLD,QUERIES,VIEWS,SMOKE,frozen,check_scores,wording,digest,as_json,read,write
from infer import init,make_views,hash_tensor,displayed

def effective(t):
    row=t[0].tolist()
    return row[:row.index(0)] if 0 in row else row

def check_direction(rows,photos,old):
    pairs=read(OLD/"pairs.csv")
    legacy={r["pair_id"]:r for r in read(OLD/"results/pair_results.csv")}
    values=check_scores(rows,photos,True);audit=[]
    for pid in SMOKE:
        a,b=sorted([r for r in pairs if r["pair_id"]==pid],key=lambda r:r["side"])
        assert (a["side"],b["side"])==("A","B")
        for q in QUERIES:
            assert sorted([a[q+"_present"],b[q+"_present"]])==["0","1"]
            assert a["crown_present"]!=a["wing_present"] and b["crown_present"]!=b["wing_present"]
            gold=a if a[q+"_present"]=="1" else b
            loser=b if gold is a else a
            item=dict(pair_id=pid,query=q,gold=gold["side"],image_A=a["image_id"],image_B=b["image_id"])
            for condition,scores in (("color",old),("neutral",values)):
                for v in VIEWS:
                    def s(r):return float(scores[r["image_id"],v,q]["score"])
                    margin=s(gold)-s(loser)
                    assert margin==(s(a)-s(b))*(1 if gold is a else -1)
                    item[condition+"_"+v]=margin
            assert item["color_original"]==float(legacy[pid]["original_"+q])
            assert (item["color_original"]+item["color_"+q])/2==float(legacy[pid]["matched_"+q])
            audit.append(item)
    return audit

def run(args):
    started=perf_counter();frozen();out=args.out
    cache=out/"checkpoints";display=out/"display_cache"
    for folder in (out,cache,display):folder.mkdir(parents=True,exist_ok=True)
    photos=read(OLD/"inputs.csv")
    old=check_scores([s for s in read(OLD/"scores.csv") if s["view"] in VIEWS],photos,False)
    selected=[p for p in photos if not args.smoke or p["pair_id"] in SMOKE]
    device=os.environ["CUDA_VISIBLE_DEVICES"]
    assert device.startswith("GPU-") and torch.cuda.device_count()==1
    gpu=subprocess.check_output(["nvidia-smi","-i",device,"--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    assert gpu.split(", ")[1]==device
    model,tf,tokenizer,env=init(args.pilot,"cuda")
    baseline=json.loads((OLD/"environment.json").read_text())
    assert all(baseline[k]==v for k,v in env.items()) and not model.training
    env.update(gpu=gpu,hostname=socket.gethostname(),score_dtype="float16",image_batch=1,text_batch=1,
               source_sha256=digest(ROOT/"source.json"),runner_sha256=digest(Path(__file__)),
               control_sha256=digest(ROOT/"color_control.py"),protocol_sha256=digest(ROOT/"protocol.md"))
    envfile=out/"environment.json"
    if envfile.exists():assert json.loads(envfile.read_text())==env
    else:as_json(envfile,env)
    signature=digest(envfile)
    if not args.smoke:
        assert json.loads((out/"smoke_check.json").read_text())["signature"]==signature
        assert all((cache/(p["image_id"]+".json")).is_file() for p in photos if p["pair_id"] in SMOKE)
    all_rows=[];hits=0;visual=0;texts={};tokens_log={}
    start_loop=perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for q in QUERIES:
            tokens=tokenizer([wording(q)])
            seq=effective(tokens)
            assert seq[-1]==49407 and len(seq)<tokens.shape[-1]
            texts[q]=(model.encode_text(tokens.cuda(),normalize=True),hash_tensor(tokens))
            tokens_log[wording(q)]=seq
        assert texts["wing"][1]!=texts["crown"][1]
        for p in selected:
            i=p["image_id"];file=cache/(i+".json");photo_path=args.data/"images"/p["path"]
            assert digest(photo_path)==p["jpeg_sha256"]
            for q in QUERIES:
                prompt=old[i,"original",q]["prompt"]
                old_tok=tokenizer([prompt])
                assert all(hash_tensor(old_tok)==old[i,v,q]["token_sha256"] for v in VIEWS)
                tokens_log[prompt]=effective(old_tok)
                # The full effective token sequence differs only by the color word.
                prefix=effective(tokenizer(["The bird has"]))[:-1]
                color=effective(tokenizer([p["color"]]))[1:-1]
                no_color=tokens_log[wording(q)]
                assert no_color[:len(prefix)]==prefix and color
                assert tokens_log[prompt]==prefix+color+no_color[len(prefix):]
            if file.exists():
                entry=json.loads(file.read_text())
                assert entry["signature"]==signature
                check_scores(entry["scores"],[p],True)
                all_rows.extend(entry["scores"]);hits+=6;continue
            with Image.open(photo_path) as im:photo=im.convert("RGB")
            assert photo.size==(int(p["width"]),int(p["height"]))
            views,_=make_views(photo,p);records=[]
            for v in VIEWS:
                t=tf(views[v]).contiguous();h=hash_tensor(t)
                assert t.shape==(3,224,224) and all(h==old[i,v,q]["tensor_sha256"] for q in QUERIES)
                displayed(t,tf).save(display/f"{i}_{v}.png")
                feat=model.encode_image(t[None].cuda(),normalize=True);visual+=1
                for q in QUERIES:
                    tx,tokhash=texts[q];native=(feat@tx.T).flatten()
                    assert native.numel()==1 and native.dtype==torch.float16
                    value=float(native.item());assert np.isfinite(value)
                    records.append(dict(pair_id=p["pair_id"],side=p["side"],image_id=i,view=v,query=q,prompt=wording(q),
                                        score=value,tensor_sha256=h,token_sha256=tokhash))
            tmp=file.with_suffix(".tmp");as_json(tmp,dict(signature=signature,scores=records));tmp.replace(file)
            all_rows.extend(records)
    torch.cuda.synchronize()
    check_scores(all_rows,selected,True)
    prefix="smoke_" if args.smoke else ""
    write(out/(prefix+"neutral_scores.csv"),all_rows)
    as_json(out/(prefix+"tokens.json"),tokens_log)
    if args.smoke:
        audit=check_direction(all_rows,selected,old)
        as_json(out/"smoke_check.json",dict(signature=signature,verified_tensors=12,directions=audit))
    stats=dict(photos=len(selected),visual_encodes=visual,new_text_encodes=len(texts),new_scores=2*visual,
               checkpoint_score_hits=hits,loop_seconds=perf_counter()-start_loop,wall_seconds=perf_counter()-started)
    as_json(out/(prefix+"run.json"),stats);print(json.dumps(stats),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True);ap.add_argument("--pilot",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=ROOT);ap.add_argument("--smoke",action="store_true")
    run(ap.parse_args())
