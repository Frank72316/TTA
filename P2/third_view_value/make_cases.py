"""Render the prospectively selected cases on CPU and verify every displayed tensor."""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from hpsv2.src.open_clip.transform import image_transform
from common_io import HERE, read

def main(data):
    pairs={r["pair_id"]:r for r in read(HERE/"pairs_frozen.csv")}
    scores={(r["pair_id"],r["side"],r["view"]):r for r in read(HERE/"scores.csv")}
    cases=json.loads((HERE/"cases.json").read_text())
    if not cases:
        return
    tf=image_transform(224,is_train=False,resize_longest_max=True)
    norm=tf.transforms[-1].normalize
    mean=np.asarray(norm.mean)[:,None,None]
    std=np.asarray(norm.std)[:,None,None]
    fig=plt.figure(figsize=(5.9*len(cases),6.4),layout="constrained")
    subfigs=np.atleast_1d(fig.subfigures(1,len(cases)))
    for case,sf in zip(cases,subfigs):
        pid=case["pair_id"]
        p=pairs[pid]
        views=["O",case["c1"],case["third"]]
        axes=sf.subplots(2,3)
        for row,side in enumerate("AB"):
            path=Path(p["image_"+side])
            with ZipFile(data/(path.parent.name+".zip")) as z, Image.open(BytesIO(z.read(path.name))) as im:
                rgb=im.convert("RGB")
            for ax,v in zip(axes[row],views):
                view=rgb if v=="O" else (ImageEnhance.Brightness if v[0]=="B" else ImageEnhance.Contrast)(rgb).enhance(int(v[1:])/100)
                x=tf(view).contiguous().numpy()
                assert hashlib.sha256(x.tobytes()).hexdigest()==scores[pid,side,v]["input_sha256"]
                ax.imshow(np.moveaxis(np.clip(x*std+mean,0,1),0,-1))
                ax.set(xticks=[],yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[row,0].set_ylabel("Candidate "+side,fontsize=10)
        for ax,v,m in zip(axes[0],views,[case["m_O"],case["m_C1"],case["m_third"]]):
            ax.set_title(("Original" if v=="O" else ("Contrast" if v[0]=="C" else "Brightness")+f" {int(v[1:])/100:.2f}")+f"\nm = {m:+.7f}",fontsize=10)
        winner=p["human_winner_P0"]
        other="B" if winner=="A" else "A"
        start=winner if case["stop"]=="correct" else other
        end=winner if case["final"]=="correct" else other
        desc="Brightness-only rescue" if case["case"]=="brightness_only_rescue" else "New error from brightness"
        sf.suptitle(f"{pid}   {desc}\nHuman winner {winner}   |   Stop {start}   →   Add brightness {end}",fontsize=13)
        sf.supxlabel(f"Two-view margin {case['stop_sum']/2:+.7f}    →    Three-view margin {case['three_sum']/3:+.7f}",fontsize=10)
    fig.savefig(HERE/"cases.png",dpi=155,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    main(ap.parse_args().data)
