"""Render the actual tensors of the first strict correction and strict loss."""
import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from hpsv2.src.open_clip.transform import image_transform
from study import HERE, read, pixels
from geometry import original


def render(pilot,pairs):
    transform=image_transform(224,is_train=False,resize_longest_max=True)
    norm=transform.transforms[-1].normalize
    mean=torch.tensor(norm.mean)[:,None,None]; std=torch.tensor(norm.std)[:,None,None]
    source={p["pair_id"]:p for p in read(pairs)}
    records={(r["pair_id"],r["side"],r["view"]):r for r in read(HERE/"scores.csv")}
    folder=HERE/"figures"; folder.mkdir(exist_ok=True)
    font=ImageFont.truetype("DejaVuSans.ttf",17)
    for case in read(HERE/"cases.csv"):
        pid=case["pair_id"]; b=case["B"]; t=case["T"]; bt=b+"+"+t
        canvas=Image.new("RGB",(960,524),"white"); draw=ImageDraw.Draw(canvas)
        draw.text((10,5),f'{pid}   Human winner: {case["human"]}   |   {case["event"].replace("_"," ")}',font=font,fill="black")
        for i,(v,label) in enumerate(zip(("O",b,t,bt),("Original",f"Brightness {float(b[1:]):.2f}",
                         f"Shift {t.split('_')[1]} 3%","Brightness then shift"))):
            x=32+232*i
            draw.text((x,35),label,font=font,fill="black")
            for j,side in enumerate("AB"):
                y=61+j*229
                raw=original(pilot,source[pid],side)
                tensor=transform(pixels(raw,v)).contiguous()
                assert sha256(tensor.numpy().tobytes()).hexdigest()==records[pid,side,v]["tensor_sha"]
                rgb=(tensor*std+mean).permute(1,2,0).numpy()
                canvas.paste(Image.fromarray(np.rint(255*np.clip(rgb,0,1)).astype("uint8")),(x,y))
                if i==0:draw.text((9,y+96),side,font=font,fill="black")
        canvas.save(folder/(pid+".jpg"),quality=94,subsampling=0)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--pilot",type=Path,required=True); ap.add_argument("--pairs",type=Path,required=True)
    a=ap.parse_args(); render(a.pilot,a.pairs)
