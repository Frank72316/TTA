"""Populate per-image checkpoints; every HPS invocation uses only the original prompt."""
import argparse
import fcntl
import hashlib
import math
import os
import platform
import re
import time
from datetime import datetime,timezone
from importlib.metadata import version
from study import *
from score import WEIGHTS,select_gpu
from views import original,make_view,final_rgb,preview

def check_rows(rows,p,s,token):
    assert [r["view"] for r in rows]==VERSIONS
    assert all(r["pair_id"]==p["pair_id"] and r["side"]==s and r["image_id"]==p["image_"+s]
               and r["P0"]==p["P0"] and r["token_sha256"]==token and r["native_dtype"]=="float16"
               and math.isfinite(float(r["score"])) for r in rows)

def run(limit):
    start=time.perf_counter()
    stamp=datetime.now(timezone.utc).isoformat()
    bank=pilot_scores()
    ps=pairs()
    old=file_json(PILOT/"environment.json")
    assert digest(PILOT/"score.py")==old["score_script_sha256"]
    assert digest(PILOT/"scores.csv")=="545438e8911ee8a1006c3278c826cfefe079be142cc376961fe71ad7b51a2d1a"
    gpu=select_gpu()
    os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_CACHE=str(PILOT/"data/model-cache"),HPS_ROOT=str(ROOT/"runtime/hps"),CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import numpy as np
    import torch
    import hpsv2
    from hpsv2 import img_score
    from hpsv2.src.open_clip import get_tokenizer
    from PIL import features
    from huggingface_hub import hf_hub_download
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert {k:version(k) for k in old["software"]}==old["software"]
    assert gpu["name"]==old["gpu_model"] and gpu["driver"]==old["driver"] and torch.version.cuda==old["cuda"]
    for repo,rev,name,sha in WEIGHTS:
        f=Path(hf_hub_download(repo,name,revision=rev,local_files_only=True))
        assert digest(f)==sha and f.samefile(hf_hub_download(repo,name,local_files_only=True))
    for name,sha in old["validation_archives_sha256"].items():
        assert digest(PILOT/"data"/name)==sha
    vendor=PILOT/"vendor"/("HPSv2-"+old["code_revision"])/"hpsv2"
    installed=Path(img_score.__file__).parent
    for src in [vendor/"img_score.py",*(vendor/"src/open_clip").glob("*.py"),vendor/"src/open_clip/bpe_simple_vocab_16e6.txt.gz",vendor/"src/open_clip/model_configs/ViT-H-14.json"]:
        assert digest(src)==digest(installed/src.relative_to(vendor))
    img_score.initialize_model()
    model=img_score.model_dict["model"]
    tf=img_score.model_dict["preprocess_val"]
    tokenizer=get_tokenizer("ViT-H-14")
    env=dict(pilot_environment=old,host=platform.node(),python=platform.python_version(),torch_build=torch.__version__,
             gpu_name=gpu["name"],jpeg_lib=features.version_codec("jpg"),libjpeg_turbo=features.version_feature("libjpeg_turbo"),
             preprocess=re.sub(r" at 0x[0-9a-fA-F]+","",repr(tf)),score_dtype="float16",versions=VERSIONS,
             pilot_scores_sha=digest(PILOT/"scores.csv"),source_hashes={n:digest(ROOT/n) for n in ("study.py","views.py","score_p2.py")})
    ef=ROOT/"environment.json"
    if ef.exists(): assert env==file_json(ef),"Protocol changed"
    else: write_json(ef,env)
    signature=digest(ef)
    events=[]
    def pre(mod,args):
        e=torch.cuda.Event(enable_timing=True);e.record();events.append(e)
    def post(mod,args,out):
        e=torch.cuda.Event(enable_timing=True);e.record();events.append(e)
    model.register_forward_pre_hook(pre)
    model.register_forward_hook(post)
    cache=ROOT/"cache";cache.mkdir(exist_ok=True)
    allrows=[]
    fresh=old_hits=p2_hits=0
    api_seconds=gpu_seconds=0.
    for p in ps[:limit]:
        i=p["pair_id"]
        tok=tokenizer([p["P0"]])[0].tolist()
        token=hashlib.sha256(json.dumps(tok,separators=(",",":")).encode()).hexdigest()
        for side in ("A","B"):
            dest=cache/f"{i}_{side}.json"
            if dest.exists():
                obj=file_json(dest);assert obj["protocol"]==signature
                rows=obj["rows"];p2_hits+=len(rows)
            else:
                rgb=original(p,side)
                inputs=[];keys=[];rows=[];pix={}
                for v in VERSIONS:
                    im,param=make_view(rgb,v)
                    assert im.size==rgb.size and im.mode=="RGB"
                    final=final_rgb(tf,im)
                    if v=="O": base=final
                    dif=np.abs(final-base)
                    if i=="001": pix[v]=final
                    compute=i in ("001","002") or v not in ("O","B110")
                    record=dict(pair_id=i,side=side,image_id=p["image_"+side],P0=p["P0"],token_sha256=token,
                                view=v,family=v[0],parameter=param,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                                model_width=final.shape[1],model_height=final.shape[0],input_mean_abs_RGB255=float(dif.mean(dtype=np.float64)*255),
                                input_changed_fraction=float(np.any(dif>0,axis=2).mean()),native_dtype="float16",
                                source="P2_fresh" if compute else "pilot_P0_"+("L0" if v=="O" else "L1"),
                                score=None if compute else bank[i,side,v])
                    rows.append(record)
                    if compute: inputs.append(im);keys.append(record)
                if i=="001": preview(pix,side)
                events.clear()
                torch.cuda.synchronize()
                tick=time.perf_counter()
                with torch.inference_mode():
                    vals=hpsv2.score(inputs,p["P0"],hps_version="v2.1")
                torch.cuda.synchronize()
                api_seconds+=time.perf_counter()-tick
                assert len(vals)==len(keys) and len(events)==2*len(vals) and not model.training
                gpu_seconds+=sum(a.elapsed_time(b) for a,b in zip(events[::2],events[1::2]))/1000
                for row,v in zip(keys,vals):
                    assert str(v.dtype)=="float16"
                    row["score"]=float(v)
                for row in rows:
                    if row["view"] in ("O","B110"):
                        assert row["score"]==bank[i,side,row["view"]],(i,side,row)
                check_rows(rows,p,side,token)
                fresh+=len(vals);old_hits+=len(rows)-len(vals)
                write_json(dest,dict(protocol=signature,rows=rows))
            check_rows(rows,p,side,token)
            allrows.extend(rows)
        print(i,len(allrows),flush=True)
    assert len(allrows)==limit*34
    out=ROOT/"scores.csv"
    if not out.exists() or len(read_csv(out))<=len(allrows):write_csv(out,allrows)
    runs=ROOT/"runs.json"
    history=file_json(runs) if runs.exists() else []
    history.append(dict(limit=limit,started_utc=stamp,gpu=gpu,new_inferences=fresh,old_hits=old_hits,p2_hits=p2_hits,
                        records=len(allrows),wall_seconds=time.perf_counter()-start,api_seconds=api_seconds,gpu_forward_seconds=gpu_seconds))
    write_json(runs,history)

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("--limit",type=int,choices=(2,64),required=True)
    with open(ROOT/".score.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(cli.parse_args().limit)

