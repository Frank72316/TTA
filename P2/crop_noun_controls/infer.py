import argparse,os,json,sys,subprocess
from pathlib import Path
from time import perf_counter
import torch
from PIL import Image
from freeze import HERE,read,put,sha
sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
from score import init
import hashlib
def hashed(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def run(root,data):
    start=perf_counter(); old=HERE.parent/"spatial_crop_tta"
    fr=json.loads((HERE/"freeze.json").read_text());assert sha(HERE/"mappings.csv")==fr["mappings"]
    mapped=read(HERE/"mappings.csv"); views=read(old/"views.csv")
    src={(r["pair_id"],r["side"],r["half"],r["role"]):r for r in read(old/"scores.csv")}
    model,tf,tok,env=init(root/"inputs/tta_pilot","cuda")
    hist=json.loads((old/"environment.json").read_text());assert all(hist[k]==v for k,v in env.items())
    gpu=subprocess.check_output(["nvidia-smi","-i","3","--query-gpu=uuid,name","--format=csv,noheader"],text=True).strip()
    assert gpu.split(",")[0]==os.environ["CUDA_VISIBLE_DEVICES"]
    meta=dict(**env,gpu_uuid=gpu,freeze=sha(HERE/"freeze.json"),old_scores=sha(old/"scores.csv"),views_sha=sha(old/"views.csv"),script=sha(Path(__file__)))
    text={};records=[];visual=0;checks=0;main=0;sensitivity=0
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for index,r in enumerate(views):
            pid,side=r["pair_id"],r["side"];mp=[m for m in mapped if m["pair_id"]==pid and m["donor"]]
            if not mp:continue
            f=data/r["file"];assert sha(f)==r["jpeg_sha256"]
            with Image.open(f) as im: rgb=im.convert("RGB")
            w,h=rgb.size;double=rgb.resize((2*w,2*h),Image.Resampling.NEAREST)
            for j,half in enumerate(("left","right")):
                tensor=tf(double.crop((j*w,0,(j+1)*w,2*h))).contiguous()
                key=hashed(tensor); assert key==src[pid,side,half,"x"]["tensor_sha256"]
                vi=model.encode_image(tensor[None].cuda(),normalize=True);visual+=1
                bank={}
                def score(noun):
                    prompt="A photo of the "+("TV" if noun=="tv" else noun)+"."
                    if prompt not in text:
                        tokens=tok([prompt]);text[prompt]=(model.encode_text(tokens.cuda(),normalize=True),hashed(tokens))
                    feat,token=text[prompt];v=(vi@feat.T).flatten();assert v.dtype==torch.float16
                    return float(v.item()),token
                if index<4:
                    for role in ("x","y"):
                        val,token=score(r[role]);histrow=src[pid,side,half,role]
                        assert val==float(histrow["score"]) and token==histrow["token_sha256"];checks+=1
                for m in mp:
                    for role in ("x","y"):
                        noun=m["alt_"+role]
                        if noun not in bank:
                            if m["seed"]=="42": main+=1
                            else:sensitivity+=1
                            bank[noun]=score(noun)
                        val,token=bank[noun]
                        records.append(dict(seed=m["seed"],pair_id=pid,side=side,half=half,role=role,noun=noun,
                            score=val,tensor_sha=key,token_sha=token))
            if index==3: print("two pairs passed",flush=True)
    put(HERE/"scores.csv",records);(HERE/"environment.json").write_text(json.dumps(meta,indent=2))
    (HERE/"run.json").write_text(json.dumps(dict(visual_forwards=visual,smoke_old_score_checks=checks,
        main_new_similarities=main,additional_unique_sensitivity_similarities=sensitivity,
        old_score_rows=len(src),wall_seconds=perf_counter()-start),indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--data",type=Path,required=True)
    a=ap.parse_args();run(a.root,a.data)
