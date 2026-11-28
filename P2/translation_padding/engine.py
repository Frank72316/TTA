"""Common historical HPS loader and cache validation for the two pixel interventions."""
import hashlib,json,os,sys,subprocess
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
from score import init,digest,save_json
from geometry import original,view,AXES
from history import read,write

def hashed(array):return hashlib.sha256(array.tobytes()).hexdigest()

class Engine:
    def __init__(self,root,device):
        self.pilot=root/"inputs/tta_pilot"
        self.pairs=read(root/"inputs/pairs_frozen.csv")
        old=root/"production/dev256"
        self.model,self.tf,self.tokenizer,env=init(self.pilot,"cuda")
        prev=json.loads((old/"environment.json").read_text())
        assert all(v==prev[k] for k,v in env.items()) and prev["image_batch_size"]==1
        assert prev["pairs_sha"]==digest(root/"inputs/pairs_frozen.csv") and len(self.pairs)==256
        assert prev["validation_archives"]=={n:digest(self.pilot/"data"/n) for n in prev["validation_archives"]}
        query=subprocess.check_output(["nvidia-smi","-i",str(device),"--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
        assert query.split(",")[0]==os.environ["CUDA_VISIBLE_DEVICES"]
        self.meta=dict(**env,device_binding=query,source_scores_sha=digest(old/"scores.csv"),pairs_sha=prev["pairs_sha"],
            archives=prev["validation_archives"],image_batch=1,text_batch=1)
        self.bank={(r["pair_id"],r["side"],r["view"]):r for r in read(old/"scores.csv") if r["view"]=="O" or r["view"].startswith("T_")}
        self.visual=0;self.hits=0
    def prompt(self,p):
        tokens=self.tokenizer([p["P0"]]); self.token=hashed(tokens.numpy().astype("<i8"))
        self.text=self.model.encode_text(tokens.cuda(),normalize=True)
    def score(self,tensor):
        self.visual+=1
        value=(self.model.encode_image(tensor[None].cuda(),normalize=True)@self.text.T).flatten()
        assert value.numel()==1 and value.dtype==torch.float16
        return float(value.item())
    def historical(self,p,side,v,rgb,image,tensor,smoke):
        r=self.bank[p["pair_id"],side,v]
        assert (r["original_rgb_sha"],r["view_rgb_sha"],r["tensor_sha"],r["token_sha"])==(hashed(np.asarray(rgb)),hashed(np.asarray(image)),hashed(tensor.numpy()),self.token)
        assert (r["P0"],r["human"],r["image_id"],r["prompt_id"])==(p["P0"],p["human_winner_P0"],p["image_"+side],p["prompt_id"])
        val=float(r["score"])
        if smoke: assert self.score(tensor)==val
        self.hits+=1
        return val
