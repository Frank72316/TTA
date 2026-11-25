"""Rebuild only display inputs on the existing dataset host. No model initialization or forward."""
from pathlib import Path
import os,sys,csv,hashlib,json,re,shutil
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
P2=ROOT.parent/"tta_p2_complementarity"
P3=ROOT.parent/"tta_p3_translation"
pilot=ROOT.parent/"tta_pilot"
sys.dont_write_bytecode=True
sys.path.insert(0,str(P2))
from views import original,make_view,final_rgb
from hpsv2.src.open_clip.transform import image_transform
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
tf=image_transform(224,False,resize_longest_max=True)
env=json.loads((P2/"environment.json").read_text())
assert re.sub(r" at 0x[0-9a-fA-F]+","",repr(tf))==env["preprocess"]
assert hashlib.sha256((P2/"views.py").read_bytes()).hexdigest()==env["source_hashes"]["views.py"]
plan={"003":["O","B090","Tdown"],"008":["O","B090","Tleft"],"N032":["O","B090","Tup","Tdown"],"N141":["O","B090","Tleft","Tright"]}
out=ROOT/"assets";out.mkdir(exist_ok=True)
audit=[]
for group,pairs,cache in [("old",pilot/"pairs.csv",P2/"scores.csv"),("new",P3/"pairs_frozen.csv",P3/"scores.csv")]:
    bank={(r["pair_id"],r["side"],r["view"]):r for r in rows(cache)}
    for p in rows(pairs):
        i=p["pair_id"]
        if i not in plan:continue
        for side in "AB":
            orig=original(p,side)
            for v in plan[i]:
                r=bank[i,side,v]
                assert r["P0"]==p["P0"] and r["image_id"]==p["image_"+side]
                view,params=make_view(orig,v)
                sha=hashlib.sha256(view.tobytes()).hexdigest()
                assert sha==r.get("pixel_sha256",r.get("input_pixels_sha"))
                arr=np.uint8(np.rint(255*np.clip(final_rgb(tf,view),0,1)))
                assert arr.shape==(224,224,3)
                name=f"{i}-{side}-{v}.png";dest=out/name;prior=P3/"old_cases/assets"/name
                if group=="old" and prior.exists():
                    with Image.open(prior) as old:assert np.array_equal(arr,np.asarray(old.convert("RGB")))
                    shutil.copyfile(prior,dest);origin="verified P3 case PNG"
                else:
                    Image.fromarray(arr).save(dest);origin="reconstructed from validation ZIP"
                audit.append(dict(pair_id=i,side=side,view=v,image_id=r["image_id"],params=params,source=origin,pixels_sha256=sha,filename=name,png_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
assert len(audit)==28
with (ROOT/"case_inputs.csv").open("w",encoding="utf8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=audit[0]);w.writeheader();w.writerows(audit)
print("Verified",len(audit),"display views.")

