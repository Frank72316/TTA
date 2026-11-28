"""Score a single cohort under one frozen official HPS configuration, offline."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from geometry import PARAMS, check_geometry, original, view
from history import HERE, read, write


def digest(path):
    h=sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf8")
    tmp.replace(path)


def init(pilot,device):
    os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_CACHE=str(pilot/"data/model-cache"),
                      HPS_ROOT=str(pilot/"data/hps"),CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if device=="cpu":
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    import numpy as np
    import PIL
    from hpsv2 import img_score
    from hpsv2.src.open_clip import get_tokenizer
    from hpsv2.utils import hps_version_map
    from huggingface_hub import hf_hub_download
    import re
    env=json.loads((pilot/"environment.json").read_text())
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS","4")))
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    ckpt=None
    for repo,rev,file,h in env["weights"]:
        path=Path(hf_hub_download(repo,file,revision=rev,local_files_only=True))
        assert path.samefile(hf_hub_download(repo,file,local_files_only=True)) and digest(path)==h
        if file==hps_version_map["v2.1"]:
            ckpt=path
    assert ckpt is not None and img_score.device==device
    img_score.initialize_model()
    model=img_score.model_dict["model"]
    tf=img_score.model_dict["preprocess_val"]
    state=torch.load(ckpt,map_location="cpu",weights_only=True)["state_dict"]
    model.load_state_dict(state,strict=True)
    del state
    model.eval()
    meta=dict(hps_version="v2.1",weights=env["weights"],torch=torch.__version__,pillow=PIL.__version__,numpy=np.__version__,
              cuda=torch.version.cuda,device=device,hardware=torch.cuda.get_device_name() if device=="cuda" else "CPU",
              precision="CUDA autocast FP16" if device=="cuda" else "CPU float32",
              preprocess=re.sub(r" at 0x[0-9a-fA-F]+","",repr(tf)),scorer_sha=digest(Path(img_score.__file__)),
              geometry_sha=digest(HERE/"geometry.py"),implementation_sha=digest(Path(__file__)))
    return model,tf,get_tokenizer("ViT-H-14"),meta


def run(args):
    start=perf_counter()
    check_geometry()
    out=args.out
    out.mkdir(exist_ok=True,parents=True)
    cache=out/"cache"
    cache.mkdir(exist_ok=True)
    pairs=read(args.pairs)
    assert len(pairs) in (64,256) and len({p["pair_id"] for p in pairs})==len(pairs)
    if len(pairs)==256:
        assert args.frozen_strength and args.frozen_strength.is_file()
        assert (HERE/"historical_seven.csv").is_file()
    model,tf,tok,meta=init(args.pilot,args.device)
    import torch
    import numpy as np
    from PIL import Image
    meta["pairs_sha"]=digest(args.pairs)
    meta["image_batch_size"]=args.batch
    meta["validation_archives"]={p.name:digest(p) for p in sorted((args.pilot/"data").glob("validation_*.zip"))}
    assert meta["validation_archives"]==json.loads((args.pilot/"environment.json").read_text())["validation_archives_sha256"]
    meta["strength_freeze"]=digest(args.frozen_strength) if args.frozen_strength else None
    metafile=out/"environment.json"
    if metafile.exists():
        assert json.loads(metafile.read_text())==meta
    else:
        save_json(metafile,meta)
    signature=digest(metafile)
    total=[]
    forwarded=reused=same=0
    inference_seconds=0.
    norm=tf.transforms[-1].normalize
    std=torch.tensor(norm.std)[:,None,None]
    mean=torch.tensor(norm.mean)[:,None,None]
    for p in pairs[args.start:args.limit]:
        ix=p["pair_id"]
        dest=cache/(ix+".json")
        if dest.exists():
            record=json.loads(dest.read_text())
            assert record["signature"]==signature and len(record["rows"])==66
            total.extend(record["rows"])
            reused+=66
            continue
        token=tok([p["P0"]])
        token_sha=sha256(token.numpy().astype("<i8").tobytes()).hexdigest()
        ctx=torch.autocast("cuda",dtype=torch.float16) if args.device=="cuda" else nullcontext()
        rows=[]
        with torch.inference_mode(),ctx:
            text=model.encode_text(token.to(args.device),normalize=True)
            for side in "AB":
                rgb=original(args.pilot,p,side)
                scored={}
                inputs=[]
                side_rows=[]
                for v,(family,param) in PARAMS.items():
                    image=view(rgb,v)
                    tensor=tf(image).contiguous()
                    assert tensor.shape==(3,224,224) and image.size==rgb.size
                    key=sha256(tensor.numpy().tobytes()).hexdigest()
                    pixels=sha256(image.tobytes()).hexdigest()
                    if v=="O":
                        base=tensor
                        original_sha=pixels
                    rgbdiff=((tensor-base)*std).abs().mean(dtype=torch.float64).item()*255
                    duplicate=key in scored
                    if duplicate:
                        source=scored[key]
                        same+=1
                    else:
                        inputs.append((v,tensor))
                        scored[key]=v
                        source=v
                        forwarded+=1
                    if ix==pairs[0]["pair_id"] or ix==pairs[1]["pair_id"]:
                        path=out/"smoke"/ix/side
                        path.mkdir(exist_ok=True,parents=True)
                        arr=((tensor*std+mean).permute(1,2,0).numpy()*255).round().clip(0,255).astype("uint8")
                        Image.fromarray(arr).save(path/(v+".png"))
                    side_rows.append(dict(pair_id=ix,prompt_id=p["prompt_id"],P0=p["P0"],human=p["human_winner_P0"],
                                     side=side,image_id=p["image_"+side],view=v,family=family,parameter=json.dumps(param),
                                     original_rgb_sha=original_sha,view_rgb_sha=pixels,token_sha=token_sha,
                                     tensor_sha=key,mean_abs_rgb255=rgbdiff,scored_view=source))
                values={}
                for j in range(0,len(inputs),args.batch):
                    chunk=inputs[j:j+args.batch]
                    tick=perf_counter()
                    features=model.encode_image(torch.stack([t for _,t in chunk]).to(args.device),normalize=True)
                    raw=(features@text.T).flatten()
                    assert raw.dtype==(torch.float16 if args.device=="cuda" else torch.float32)
                    vals=raw.cpu().tolist()
                    assert np.isfinite(vals).all()
                    values.update(zip((v for v,_ in chunk),vals))
                    inference_seconds+=perf_counter()-tick
                for row in side_rows:
                    row["score"]=values[row["scored_view"]]
                rows.extend(side_rows)
        assert len(rows)==66
        save_json(dest,dict(signature=signature,rows=rows))
        total.extend(rows)
        print(ix,forwarded,round(perf_counter()-start,1),flush=True)
    outcsv=out/"scores.csv"
    if not outcsv.exists() or len(read(outcsv))<=len(total):
        write(outcsv,total)
    runs=out/"runs.json"
    history=json.loads(runs.read_text()) if runs.exists() else []
    history.append(dict(end_utc=datetime.now(timezone.utc).isoformat(),limit=args.limit,records=len(total),new=forwarded,
                        checkpoint_hits=reused,duplicate_hits=same,external_hits=0,wall_s=perf_counter()-start,
                        image_inference_s=inference_seconds))
    save_json(runs,history)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--pilot",type=Path,required=True)
    cli.add_argument("--pairs",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--limit",type=int,required=True)
    cli.add_argument("--start",type=int,default=0)
    cli.add_argument("--batch",type=int,default=1)
    cli.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    cli.add_argument("--frozen-strength",type=Path)
    run(cli.parse_args())
