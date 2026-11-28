import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from pixels import PHASE,padded,jpeg,blur,energy,visible
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"translation_padding"))
from engine import Engine,hashed,original,digest,save_json,write
def run(root,device):
    start=perf_counter(); strength=json.loads((HERE/"strength.json").read_text())
    assert strength["calibration_sha"]==digest(HERE/"calibration.csv")
    engine=Engine(root,device)
    env=dict(**engine.meta,strength_sha=digest(HERE/"strength.json"),protocol_sha=digest(HERE/"protocol.md"),
        pixels_sha=digest(HERE/"pixels.py"),script=digest(Path(__file__)))
    envpath=HERE/"environment.json"
    if envpath.exists():assert json.loads(envpath.read_text())==env
    else:save_json(envpath,env)
    ckdir=HERE/"cache";ckdir.mkdir(exist_ok=True)
    key=digest(envpath); rows=[];resume=0; dedup=0
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for i,p in enumerate(engine.pairs):
            pid=p["pair_id"];ck=ckdir/(pid+".json")
            if ck.exists():
                v=json.loads(ck.read_text());assert v["key"]==key;rows+=v["rows"];resume+=1;continue
            records=[];engine.prompt(p)
            for side in "AB":
                rgb=original(engine.pilot,p,side);ot=engine.tf(rgb).contiguous();norm=engine.tf.transforms[-1].normalize
                o=visible(ot,norm,rgb);e0=energy(o);old=engine.historical(p,side,"O",rgb,rgb,ot,i<2);bank={hashed(ot.numpy()):old}
                def row(name,t,val,arr):
                    records.append(dict(pair_id=pid,side=side,human=p["human_winner_P0"],image_id=p["image_"+side],view=name,
                        score=val,tensor_sha=hashed(t.numpy()),token_sha=engine.token,energy_ratio=energy(arr)/e0-1,rgb_mse=np.mean((arr-o)**2)))
                row("O",ot,old,o); base_gauss=None
                for x,y in PHASE:
                    im,box=padded(rgb,(x,y))
                    for name,aug in (("J",jpeg(im,box)),("G",blur(im,box,strength["radius"]))):
                        if name=="G":
                            a=np.asarray(aug)
                            if base_gauss is None:base_gauss=a
                            else:assert np.array_equal(base_gauss,a)
                        t=engine.tf(aug).contiguous();h=hashed(t.numpy())
                        if h not in bank:bank[h]=engine.score(t)
                        else:dedup+=1
                        row(f"{name}{x}{y}",t,bank[h],visible(t,norm,rgb))
            save_json(ck,dict(key=key,rows=records));rows+=records
            if i==1 or (i+1)%64==0:print(i+1,flush=True)
    assert len(rows)==4608
    write(HERE/"scores.csv",rows)
    save_json(HERE/"run.json",dict(wall_seconds=perf_counter()-start,visual_forwards=engine.visual,
        historical_O_hits=engine.hits,duplicate_tensors=dedup,resumed_pairs=resume,smoke_original_replays=4))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--device",type=int,required=True)
    a=ap.parse_args();run(a.root,a.device)
