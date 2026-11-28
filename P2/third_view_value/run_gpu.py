"""Frozen test scores and independent wall-clock measurements on one UUID-bound GPU."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageEnhance

from common_io import ARCHIVES, FACTORS, HERE, SEED, VIEWS, digest, read, save, text_hash, write

def image_view(rgb, view):
    if view == "O":
        return rgb
    enhancer = {"B": ImageEnhance.Brightness, "C": ImageEnhance.Contrast}[view[0]]
    return enhancer(rgb).enhance(int(view[1:])/100)

def decode(raw):
    with Image.open(BytesIO(raw)) as im:
        return im.convert("RGB")

def run(args):
    import torch
    import numpy as np
    start = time.perf_counter()
    frozen = json.loads((HERE/"freeze.json").read_text())
    assert text_hash(HERE/"pairs_frozen.csv") == frozen["pairs_sha256_lf"]
    assert text_hash(HERE/"protocol.md") == frozen["protocol_sha256_lf"]
    pairs = read(HERE/"pairs_frozen.csv")
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    assert uuid.startswith("GPU-") and "," not in uuid and torch.cuda.device_count() == 1
    sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
    from score import init
    model,tf,tok,env = init(args.pilot,"cuda")
    assert not model.training
    env.update(gpu_uuid=uuid, host=platform.node(), driver=subprocess.check_output([
        "nvidia-smi","-i",uuid,"--query-gpu=driver_version","--format=csv,noheader"], text=True).strip(),
        script_sha256_lf=text_hash(__file__),common_sha256_lf=text_hash(HERE/"common_io.py"),
        pairs_sha256=frozen["pairs_sha256_lf"], protocol_sha256=frozen["protocol_sha256_lf"],
        image_batch=1,text_batch=1,native_dtype="float16",archives=ARCHIVES)
    for file,h in ARCHIVES.items():
        assert digest(args.data/file) == h
    envpath = HERE/"environment.json"
    if envpath.exists():
        assert env == json.loads(envpath.read_text())
    else:
        save(envpath,env)
    signature = digest(envpath)
    archives = {Path(f).stem: ZipFile(args.data/f) for f in ARCHIVES}
    def raw_pair(p):
        raw = []
        for side in "AB":
            path = Path(p["image_"+side])
            raw.append(archives[path.parent.name].read(path.name))
        return raw
    def similarity(x,text):
        val = model.encode_image(x,normalize=True) @ text.T
        assert val.dtype == torch.float16
        return val.item()
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        if args.mode=="time":
            rng = random.Random(SEED)
            latencies = []
            for i,p in enumerate(pairs[:2] + pairs[:64]):
                raw = raw_pair(p)
                c = f"C{rng.choice(FACTORS):03d}"
                b = f"B{rng.choice(FACTORS):03d}"
                # Six permutations counterbalance order and carryover.
                for k in list(itertools.permutations((1,2,3)))[i%6]:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    text = model.encode_text(tok([p["P0"]]).cuda(),normalize=True)
                    sums = [sum(similarity(tf(image_view(rgb,v)).unsqueeze(0).cuda(),text)
                                for v in ("O",c,b)[:k]) for rgb in map(decode,raw)]
                    winner = "A" if sums[0]>sums[1] else "B" if sums[1]>sums[0] else "tie"
                    torch.cuda.synchronize()
                    ms = (time.perf_counter_ns()-t0)/1e6
                    if i >= 2:
                        latencies.append(dict(pair_id=p["pair_id"],k=k,c1=c,b1=b,ms=ms,winner=winner))
            write(HERE/"timings.csv",latencies)
            save(HERE/"timing_run.json",dict(warmup_pairs=2,measured_pairs=64,extra_image_encodings=66*12,
                                             extra_text_encodings=66*3,wall_seconds=time.perf_counter()-start))
            return
        cache = HERE/"cache"
        cache.mkdir(exist_ok=True)
        rows = []
        hits = qa = fresh = 0
        for index,p in enumerate(pairs[:args.limit]):
            dest = cache/(p["pair_id"]+".json")
            if dest.exists():
                ck = json.loads(dest.read_text())
                assert ck["signature"] == signature
                group = ck["rows"]
                hits += len(group)
            else:
                assert int(p["rank_A"]) != int(p["rank_B"])
                assert ("A" if int(p["rank_A"]) < int(p["rank_B"]) else "B") == p["human_winner_P0"]
                token = tok([p["P0"]]).cuda()
                th = hashlib.sha256(token.cpu().numpy().astype("<i8").tobytes()).hexdigest()
                text = model.encode_text(token,normalize=True)
                group = []
                for side,raw in zip("AB",raw_pair(p)):
                    rgb = decode(raw)
                    image_hash = hashlib.sha256(raw).hexdigest()
                    for v in VIEWS:
                        x = tf(image_view(rgb,v)).contiguous()
                        assert x.shape==(3,224,224) and torch.isfinite(x).all()
                        key = hashlib.sha256(x.numpy().tobytes()).hexdigest()
                        x = x.unsqueeze(0).cuda()
                        val = similarity(x,text)
                        if index < 2 and v=="O":
                            ref = model(x,token)
                            assert (ref["image_features"] @ ref["text_features"].T).item() == val
                            qa += 1
                        group.append(dict(pair_id=p["pair_id"],prompt_id=p["prompt_id"],side=side,image_id=p["image_"+side],
                                          view=v,param="original" if v=="O" else int(v[1:])/100,
                                          source_sha256=image_hash,input_sha256=key,token_sha256=th,native_dtype="float16",score=val))
                if index<2:
                    for side in "AB":
                        assert len({r["input_sha256"] for r in group if r["side"]==side}) == 9
                fresh += len(group)
                save(dest,dict(signature=signature,rows=group))
            assert [(r["side"],r["view"]) for r in group] == list(itertools.product("AB",VIEWS))
            assert all(np.isfinite(r["score"]) and r["image_id"]==p["image_"+r["side"]] and r["prompt_id"]==p["prompt_id"] for r in group)
            rows.extend(group)
            if (index+1)%25==0 or index<2:
                print(index+1,"pairs",len(rows),"scores",flush=True)
        name = "scores.csv" if len(rows)==18*len(pairs) else "smoke_scores.csv"
        write(HERE/name, rows)
        log = HERE/"runs.json"
        runs = json.loads(log.read_text()) if log.exists() else []
        runs.append(dict(pairs=len(rows)//18,new_scores=fresh,cache_hits=hits,qa_image_encodings=qa,
                         wall_seconds=time.perf_counter()-start))
        save(log,runs)
        if args.limit==2:
            for p in pairs[:2]:
                group = [r for r in rows if r["pair_id"]==p["pair_id"] and r["view"]=="O"]
                a,b = [r["score"] for r in group]
                m = a-b if p["human_winner_P0"]=="A" else b-a
                assert m == group["AB".index(p["human_winner_P0"])]["score"] - group[1-"AB".index(p["human_winner_P0"])]["score"]
                print(p["pair_id"],p["rank_A"],p["rank_B"],p["human_winner_P0"],a,b,m)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["score","time"])
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--pilot",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=400,choices=[2,400])
    with (HERE/".gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(ap.parse_args())
