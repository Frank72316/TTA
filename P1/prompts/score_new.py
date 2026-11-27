import argparse
import fcntl
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

from common import *
from score import WEIGHTS, select_gpu

from layout_provenance import matches_environment

def run(limit):
    prompts=frozen()
    oldenv=jsonfile(OLD/"environment.json")
    stamp=datetime.now(timezone.utc).isoformat()
    start=time.perf_counter()
    gpu=select_gpu()
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(OLD/"data/model-cache"),
                      HPS_ROOT=str(ROOT/"runtime/hps"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import hpsv2
    import torch
    from hpsv2 import img_score
    from huggingface_hub import hf_hub_download
    from PIL import Image
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==1 and torch.cuda.is_available()
    sw={k:version(k) for k in oldenv["software"]}
    assert sw==oldenv["software"]
    for repo,rev,name,sha in WEIGHTS:
        path=Path(hf_hub_download(repo,name,revision=rev,local_files_only=True))
        assert path.samefile(hf_hub_download(repo,name,local_files_only=True)) and digest(path)==sha
    for name,sha in oldenv["validation_archives_sha256"].items():
        assert digest(OLD/"data"/name)==sha
    src_sha=digest(Path(img_score.__file__))
    assert src_sha==digest(OLD/"vendor"/("HPSv2-"+oldenv["code_revision"])/"hpsv2/img_score.py")
    env=dict(software=sw,python=platform.python_version(),host=platform.node(),torch_build=torch.__version__,
             cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=gpu["name"],driver=gpu["driver"],
             code_rev=oldenv["code_revision"],installed_scorer_sha256=src_sha,weights=WEIGHTS,
             archives=oldenv["validation_archives_sha256"],data_revision=oldenv["dataset_revision"],
             brightness=1.1,precision="official CUDA AMP fp16 output; FP64 analysis",hps_version="v2.1",
             prompts_sha=digest(ROOT/"prompts.csv"),core_sha=digest(ROOT/"common.py"),script_sha=digest(Path(__file__)))
    envpath=ROOT/"environment.json"
    if envpath.exists():
        assert matches_environment(jsonfile(envpath), json.loads(json.dumps(env)), ('core_sha', 'script_sha'))
    else:
        write_json(envpath,env)
    signature=digest(envpath)
    cache=ROOT/"cache"
    cache.mkdir(exist_ok=True)
    current=[]
    hits=0
    new=0
    forward_ms=0.
    call_wall=0.
    smoke=[]
    model=None
    for p in prompts[:limit]:
        i=p["pair_id"]
        dest=cache/(i+".json")
        if dest.exists():
            obj=jsonfile(dest)
            assert obj["protocol"]==signature
            rows=obj["rows"]
            hits+=len(rows)
        else:
            if model is None:
                img_score.initialize_model()
                model=img_score.model_dict["model"]
                events=[]
                def pre(module,args):
                    e=torch.cuda.Event(enable_timing=True)
                    e.record()
                    events.append(e)
                def post(module,args,output):
                    e=torch.cuda.Event(enable_timing=True)
                    e.record()
                    events.append(e)
                model.register_forward_pre_hook(pre)
                model.register_forward_hook(post)
            pics=images(i)
            if i in list(PAIRS)[:2]:
                transform=img_score.model_dict["preprocess_val"]
                for idx in (0,2):
                    orig,bright=pics[idx:idx+2]
                    check=Image.blend(Image.new("RGB",orig.size),orig,1.1)
                    assert bright.tobytes()==check.tobytes() and orig.tobytes()!=bright.tobytes()
                    tensors=[transform(im) for im in (orig,bright,check)]
                    assert torch.equal(tensors[1],tensors[2]) and torch.isfinite(tensors[0]).all()
                    smoke.append(dict(pair_id=i,side=CELLS[idx][0],shape=list(tensors[0].shape),
                                      preprocess=str(transform),pixel_sha=[hashlib.sha256(im.tobytes()).hexdigest() for im in (orig,bright)]))
            rows=[]
            pair_ms=0.
            pair_wall=0.
            for cond in ("Q0","Q1"):
                full,sent,_=token_record(p[cond])
                assert full==sent and token_hash(sent)==p[cond+"_token_sha"]
                events.clear()
                torch.cuda.synchronize()
                tick=time.perf_counter()
                with torch.inference_mode():
                    vals=hpsv2.score(pics,p[cond],hps_version="v2.1")
                torch.cuda.synchronize()
                pair_wall+=time.perf_counter()-tick
                assert not model.training and len(vals)==4 and len(events)==8
                pair_ms+=sum(a.elapsed_time(b) for a,b in zip(events[::2],events[1::2]))
                for (side,light),v in zip(CELLS,vals):
                    rows.append(dict(pair_id=i,side=side,prompt_condition=cond,light=light,
                                     image_id=p["image_"+side],score=float(v),native_dtype=str(v.dtype),
                                     text_token_sha=token_hash(sent),factor=1.0 if light=="L0" else 1.1,source="round2_fresh"))
            index_scores(rows,[i],("Q0","Q1"))
            write_json(dest,dict(protocol=signature,rows=rows,forward_ms=pair_ms,api_seconds=pair_wall))
            forward_ms+=pair_ms
            call_wall+=pair_wall
            new+=len(rows)
        index_scores(rows,[i],("Q0","Q1"))
        assert all(r["text_token_sha"]==p[r["prompt_condition"]+"_token_sha"] for r in rows)
        current.extend(rows)
        print(i, len(current),flush=True)
    # The 2-pair smoke command must never shrink a completed export.
    out=ROOT/"scores.csv"
    if not out.exists() or len(read_csv(out))<=len(current):
        write_csv(out,current)
    if smoke:
        write_json(ROOT/"smoke_checks.json",smoke)
    runs=ROOT/"runs.json"
    history=jsonfile(runs) if runs.exists() else []
    history.append(dict(start_utc=stamp,limit=limit,wall_seconds=time.perf_counter()-start,gpu=gpu,
                        forward_seconds=forward_ms/1000,api_seconds=call_wall,new_scores=new,
                        round2_cache_hits=hits,old_score_hits=0))
    write_json(runs,history)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,choices=(2,64),required=True)
    with open(ROOT/".score.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(ap.parse_args().limit)

