"""Pinned official model and singleton image inference, caching text features within one prompt."""
import argparse
import fcntl
import os
import platform
import re
import time
from datetime import datetime,timezone
from importlib.metadata import version
import numpy as np
from common import *
from score import select_gpu,WEIGHTS

def run(limit):
    start=time.perf_counter();stamp=datetime.now(timezone.utc).isoformat()
    ps=frozen();limit=min(limit,len(ps));gpu=select_gpu()
    os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_CACHE=str(PILOT/"data/model-cache"),HPS_ROOT=str(ROOT/"runtime/hps"),CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    from hpsv2 import img_score
    from hpsv2.src.open_clip import get_tokenizer
    from huggingface_hub import hf_hub_download
    torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==1
    p2=file_json(P2/"environment.json");old=p2["pilot_environment"]
    assert {k:version(k) for k in old["software"]}==old["software"]
    assert gpu["name"]==p2["gpu_name"] and torch.__version__==p2["torch_build"]
    for n in ("views.py","study.py"):assert digest(P2/n)==p2["source_hashes"][n]
    for name,sha in old["validation_archives_sha256"].items():assert digest(PILOT/"data"/name)==sha
    for repo,rev,name,sha in WEIGHTS:
        f=Path(hf_hub_download(repo,name,revision=rev,local_files_only=True))
        assert digest(f)==sha and f.samefile(hf_hub_download(repo,name,local_files_only=True))
        if repo=="xswu/HPSv2":checkpoint=f
    vendor=PILOT/"vendor"/("HPSv2-"+old["code_revision"])/"hpsv2";installed=Path(img_score.__file__).parent
    for f in [vendor/"img_score.py",*(vendor/"src/open_clip").glob("*.py"),vendor/"src/open_clip/bpe_simple_vocab_16e6.txt.gz",vendor/"src/open_clip/model_configs/ViT-H-14.json"]:
        assert digest(f)==digest(installed/f.relative_to(vendor))
    img_score.initialize_model()
    model=img_score.model_dict["model"];tf=img_score.model_dict["preprocess_val"]
    assert re.sub(r" at 0x[0-9a-fA-F]+","",repr(tf))==p2["preprocess"]
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["state_dict"],strict=True)
    model.eval()
    tokenizer=get_tokenizer("ViT-H-14")
    env=dict(hps_version="v2.1",code_rev=old["code_revision"],weights=old["weights"],software=old["software"],
             cuda=torch.version.cuda,torch_build=torch.__version__,device=gpu["name"],driver=gpu["driver"],host=platform.node(),
             preprocess=p2["preprocess"],return_dtype="float16",method="Official normalized singleton image/text features and CUDA AMP dot product",
             p2_views_sha=digest(P2/"views.py"),sources={n:digest(ROOT/n) for n in ("common.py","score_p3.py")},
             pairs_sha=digest(ROOT/"pairs_frozen.csv"),protocol_sha=digest(ROOT/"protocol.md"))
    envfile=ROOT/"environment.json"
    if envfile.exists():assert file_json(envfile)==env
    else:write_json(envfile,env)
    signature=digest(envfile);cache=ROOT/"cache";cache.mkdir(exist_ok=True)
    fresh=hits=qa=0;gpu_secs=0.;out=[]
    for p in ps[:limit]:
        i=p["pair_id"];dest=cache/(i+".json")
        if dest.exists():
            obj=file_json(dest);assert obj["protocol"]==signature;rows=obj["rows"];hits+=len(rows)
        else:
            tok=tokenizer([p["P0"]]);token_sha=hashlib.sha256(tok.numpy().astype("<i8").tobytes()).hexdigest()
            rows=[];pil=[];tensors=[]
            for s in ("A","B"):
                rgb=original(p,s)
                for v in VIEWS:
                    view,param=make_view(rgb,v);t=tf(view)
                    assert t.shape==(3,224,224) and view.size==rgb.size
                    if v=="O":orig_tensor=t
                    if i in ("N001","N002") and v!="O":assert not torch.equal(t,orig_tensor)
                    tensors.append(t);pil.append(view)
                    rows.append(dict(pair_id=i,prompt_id=p["prompt_id"],P0=p["P0"],side=s,image_id=p["image_"+s],view=v,param=param,
                                     token_sha256=token_sha,input_pixels_sha=hashlib.sha256(view.tobytes()).hexdigest(),native_dtype="float16"))
            batch=torch.stack(tensors).to("cuda");text=tok.to("cuda")
            t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                t0.record()
                feat=model.encode_text(text,normalize=True)
                vals=torch.cat([(model.encode_image(x.unsqueeze(0),normalize=True) @ feat.T).reshape(1) for x in batch])
                t1.record()
            assert vals.dtype==torch.float16 and not model.training
            vals=vals.cpu().numpy();gpu_secs+=t0.elapsed_time(t1)/1000
            if i in ("N001","N002"):
                with torch.inference_mode():
                    ref=img_score.score(pil,p["P0"],cp=str(checkpoint),hps_version="v2.1")
                assert np.array_equal(vals,np.asarray(ref)),(i,vals,ref)
                qa+=len(ref)
            for row,v in zip(rows,vals):row["score"]=float(v)
            fresh+=len(rows)
            write_json(dest,dict(protocol=signature,rows=rows))
        assert [(r["side"],r["view"]) for r in rows]==list(itertools.product(("A","B"),VIEWS))
        assert all(np.isfinite(float(r["score"])) and r["P0"]==p["P0"] and r["image_id"]==p["image_"+r["side"]] for r in rows)
        out.extend(rows)
        print(i,len(out),flush=True)
    file=ROOT/"scores.csv"
    if not file.exists() or len(read_csv(file))<=len(out):write_csv(file,out)
    file=ROOT/"runs.json";history=file_json(file) if file.exists() else []
    history.append(dict(limit=limit,start_utc=stamp,wall_seconds=time.perf_counter()-start,unique_new_scores=fresh,checkpoint_hits=hits,
                        old_score_hits=0,reference_check_extra_forwards=qa,gpu_score_seconds=gpu_secs,records=len(out),gpu=gpu))
    write_json(file,history)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--limit",type=int,choices=[2,256],required=True)
    with open(ROOT/".score.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(ap.parse_args().limit)

