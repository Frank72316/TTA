"""Each retained half is shown in its original position on the original-size canvas."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter
import numpy as np
from PIL import Image
import torch
from design import HERE,SOURCE,LINES,ROLES,KEPT,GRAY,rows,save_json,save_rows,sha,digest,noun_prompt
sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
from score import init

def mask_photo(photo,k,side):
    a=np.array(photo,copy=True)
    if side=="left":a[:,k:]=GRAY
    else:a[:,:k]=GRAY
    return Image.fromarray(a)

def hash_tensor(t):return sha(t.contiguous().numpy().tobytes())

def displayed(t,tf):
    norm=tf.transforms[-1].normalize
    rgb=(t*torch.tensor(norm.std)[:,None,None]+torch.tensor(norm.mean)[:,None,None]).permute(1,2,0).numpy()
    return Image.fromarray(np.uint8(np.clip(np.rint(rgb*255),0,255)))

def verify_smoke(photo,im,row,cut,views,original,tensors,tf,folder):
    """Compare original and masked tensors at the exact same pixel coordinates."""
    k=int(cut["x_pixel"]);a=np.asarray(photo);w,h=photo.size
    x,y,rw,rh=json.loads(im["valid_xywh"]);line=int(cut["line"])
    assert np.array_equal(np.concatenate([np.asarray(views[0])[:,:k],np.asarray(views[1])[:,k:]],axis=1),a)
    folder.mkdir(exist_ok=True,parents=True)
    fid=row["image_id"]
    if line==25:displayed(original,tf).save(folder/(fid+"_O.png"))
    result=[]
    mask_line=x+k*rw/w
    for half,masked,t in zip(KEPT,views,tensors):
        arr=np.asarray(masked)
        assert np.all((arr[:,k:] if half=="left" else arr[:,:k])==GRAY)
        # A five-output-pixel guard excludes bicubic interpolation across the seam.
        xx=np.arange(x,x+rw)+.5
        safe=xx<mask_line-5 if half=="left" else xx>mask_line+5
        assert safe.any() and torch.equal(t[:,y:y+rh,x:x+rw][:,:,safe],original[:,y:y+rh,x:x+rw][:,:,safe])
        displayed(t,tf).save(folder/f"{fid}_{line}_{half}.png")
        for role in ("x","y"):
            sx,sy,bw,bh=json.loads(im[role+"_original_xywh"])
            sx0,sx1=max(0,sx),min(w,sx+bw);sy0,sy1=max(0,sy),min(h,sy+bh)
            if (half=="left" and sx1>k) or (half=="right" and sx0<k):continue
            box=[x+sx0*rw/w,y+sy0*rh/h,x+sx1*rw/w,y+sy1*rh/h]
            assert np.allclose(box,json.loads(im[role+"_input_xyxy"]),rtol=0,atol=1e-9)
            xxroi=safe&(xx>=box[0])&(xx<box[2]); yy=np.arange(y,y+rh)+.5
            yyroi=(yy>=box[1])&(yy<box[3])
            if xxroi.any() and yyroi.any():
                tv=t[:,y:y+rh,x:x+rw][:,yyroi][:,:,xxroi]
                ov=original[:,y:y+rh,x:x+rw][:,yyroi][:,:,xxroi]
                err=float((tv-ov).abs().max().item());assert err==0
                result.append(dict(image_id=fid,line=line,kept=half,role=role,input_box=box,roi_pixels=int(xxroi.sum()*yyroi.sum()),max_tensor_error=err))
    return result

def run(a):
    start=perf_counter();f=json.loads((HERE/"freeze.json").read_text())
    assert all(digest(HERE/n)==v for n,v in f["hashes"].items())
    assert f["mapping_sha"]==digest(HERE.parent/"crop_noun_controls/mappings.csv")
    ids=f["smoke_ids"];photos=rows(HERE/"photos.csv")
    target=[p for p in photos if p["image_id"] in ids] if a.smoke else photos
    cuts={(r["image_id"],int(r["line"])):r for r in rows(HERE/"groups.csv")}
    images={r["image_id"]:r for r in rows(SOURCE/"images.csv")}
    orig_hash={r["image_id"]:r["tensor_sha256"] for r in rows(SOURCE/"scores.csv") if r["view"]=="O" and r["direction"]=="left"}
    gpu=subprocess.check_output(["nvidia-smi","-i",os.environ["CUDA_VISIBLE_DEVICES"],"--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    assert gpu.split(", ")[1]==os.environ["CUDA_VISIBLE_DEVICES"] and torch.cuda.device_count()==1
    model,tf,tokenizer,env=init(a.pilot,"cuda")
    old_env=json.loads((HERE.parent/"crop_noun_controls/environment.json").read_text())
    assert all(old_env.get(k)==v for k,v in env.items())
    env.update(gpu=gpu,hostname=socket.gethostname(),freeze_sha=digest(HERE/"freeze.json"),runner_sha=digest(Path(__file__)),
                batch_image=1,batch_text=1,gray=GRAY)
    a.out.mkdir(exist_ok=True,parents=True); cache=a.out/"checkpoints";cache.mkdir(exist_ok=True)
    ep=a.out/"environment.json"
    if ep.exists():assert json.loads(ep.read_text())==env
    else:save_json(ep,env)
    signature=digest(ep);texts={};out=[];checked=[];visual=hits=0
    if not a.smoke:assert (a.out/"smoke.json").exists() and all((cache/(i+".json")).exists() for i in ids)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for n,p in enumerate(target,1):
            fid=p["image_id"];file=cache/(fid+".json")
            if file.exists():
                entry=json.loads(file.read_text());assert entry["signature"]==signature and len(entry["scores"])==24
                out.extend(entry["scores"]);checked.extend(entry["smoke"]);hits+=24;continue
            jpg=a.data/p["file"];assert digest(jpg)==p["jpeg_sha256"]
            with Image.open(jpg) as im:rgb=im.convert("RGB")
            original=tf(rgb).contiguous()
            assert hash_tensor(original)==orig_hash[fid] and original.shape==(3,224,224)
            assert rgb.size==(int(p["width"]),int(p["height"]))
            records=[];audit=[]
            for pct in LINES:
                cut=cuts[fid,pct]
                views=[mask_photo(rgb,int(cut["x_pixel"]),s) for s in KEPT]
                tensors=[tf(im).contiguous() for im in views]
                if fid in ids:
                    audit.extend(verify_smoke(rgb,images[fid],p,cut,views,original,tensors,tf,a.out/"smoke"))
                for half,t in zip(KEPT,tensors):
                    th=hash_tensor(t); feat=model.encode_image(t[None].cuda(),normalize=True);visual+=1
                    for role in ROLES:
                        prompt=noun_prompt(p[role])
                        if prompt not in texts:
                            tokens=tokenizer([prompt])
                            assert 0<torch.count_nonzero(tokens)<tokens.numel()
                            texts[prompt]=(model.encode_text(tokens.cuda(),normalize=True),hash_tensor(tokens))
                        tx,toksha=texts[prompt]; native=(feat@tx.T).flatten()
                        assert native.numel()==1 and native.dtype==torch.float16
                        s=float(native.item());assert np.isfinite(s)
                        records.append(dict(pair_id=p["pair_id"],side=p["side"],image_id=fid,line=pct,kept=half,
                                        role=role,noun=p[role],prompt=prompt,score=s,tensor_sha=th,token_sha=toksha))
            if fid in ids:assert {r["role"] for r in audit}=={"x","y"}
            save_json(file,dict(signature=signature,scores=records,smoke=audit))
            out.extend(records);checked.extend(audit)
            if n%32==0:print(n,flush=True)
    assert len(out)==24*len(target)==len({(r["image_id"],r["line"],r["kept"],r["role"]) for r in out})
    save_rows(a.out/("smoke_scores.csv" if a.smoke else "scores.csv"),out)
    if a.smoke:save_json(a.out/"smoke.json",dict(photo_ids=ids,unchanged_rois=checked,actual_input_pixels=224))
    runs=a.out/"runs.json";log=json.loads(runs.read_text()) if runs.exists() else []
    log.append(dict(stage="smoke" if a.smoke else "full",photos=len(target),visual_forwards=visual,new_scores=4*visual,
                   text_forwards=len(texts),checkpoint_score_hits=hits,wall_seconds=perf_counter()-start))
    save_json(runs,log);print(log[-1],flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--pilot",type=Path,required=True);ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE);ap.add_argument("--smoke",action="store_true");run(ap.parse_args())
