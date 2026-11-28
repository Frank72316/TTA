import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from PIL import features
from hpsv2.src.open_clip.transform import image_transform
from pixels import PHASE,RADII,padded,jpeg,blur,visible,energy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"geometry_specificity"))
from geometry import original
from history import read,write
from score import digest,save_json
def run(root):
    start=perf_counter();pilot=root/"inputs/tta_pilot"; tf=image_transform(224,is_train=False,resize_longest_max=True)
    pairs=read(pilot/"pairs.csv")
    assert len(pairs)==64
    torch.set_num_threads(2); rows=[];dist={r:[] for r in RADII}; targets=[]
    for p in pairs:
        for side in "AB":
            rgb=original(pilot,p,side); norm=tf.transforms[-1].normalize
            obs=lambda im:visible(tf(im),norm,rgb)
            o=obs(rgb); e0=energy(o); jpg=[]
            for phase in PHASE:
                im,box=padded(rgb,phase);a=obs(jpeg(im,box)); change=energy(a)/e0-1;jpg.append(change)
                rows.append(dict(pair_id=p["pair_id"],side=side,family="J",parameter=str(phase),relative_energy=change,rgb_mse=np.mean((a-o)**2)))
            goal=np.mean(jpg);targets.append(abs(goal));im,box=padded(rgb,PHASE[0])
            for radius in RADII:
                a=obs(blur(im,box,radius));change=energy(a)/e0-1
                dist[radius].append(abs(change-goal))
                rows.append(dict(pair_id=p["pair_id"],side=side,family="G",parameter=radius,relative_energy=change,rgb_mse=np.mean((a-o)**2)))
    best=min(RADII,key=lambda r:(np.mean(dist[r]),r))
    write(HERE/"calibration.csv",rows)
    save_json(HERE/"strength.json",dict(radius=best,mean_abs_error=np.mean(dist[best]),mean_abs_target=np.mean(targets),
        matched=bool(np.mean(dist[best])<=.25*np.mean(targets)),calibration_sha=digest(HERE/"calibration.csv"),
        old_pairs_sha=digest(pilot/"pairs.csv"),protocol_sha=digest(HERE/"protocol.md"),pillow_jpeg=features.version_codec("jpg"),
        libjpeg_turbo=features.version_feature("libjpeg_turbo"),runtime_seconds=perf_counter()-start))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);run(ap.parse_args().root)
