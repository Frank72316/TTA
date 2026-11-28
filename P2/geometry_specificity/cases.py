"""Export verified model inputs for pre-specified case categories."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import POOLS, original, view
from history import read, write


def build(args):
    import torch
    from hpsv2.src.open_clip.transform import image_transform
    model_input=image_transform(224,is_train=False,resize_longest_max=True)
    norm=model_input.transforms[-1].normalize
    mu=torch.tensor(norm.mean)[:,None,None]
    std=torch.tensor(norm.std)[:,None,None]
    pairs={r["pair_id"]:r for r in read(args.pairs)}
    data={(r["pair_id"],r["side"],r["view"]):r for r in read(args.root/"dev256/scores.csv")}
    detail={(r["pair_id"],r["strategy"]):r for r in read(args.root/"dev256/full_pair_probabilities.csv") if r["K"]=="9"}
    def margin(i,v):
        return (float(data[i,"A",v]["score"])-float(data[i,"B",v]["score"]))*(1 if pairs[i]["human_winner_P0"]=="A" else -1)
    categories=[[],[],[]]
    for i in sorted(pairs):
        orig=margin(i,"O")
        t,s=(float(detail[i,f]["mean_margin"]) for f in "TS")
        if orig<0 and t>0 and s<=0:
            categories[0].append((i,"T"))
        if orig<0 and min(t,s)>0:
            categories[1].append((i,"TS"))
        if orig>0 and min(t,s)<0:
            categories[2].append((i,"T" if t<=s else "S"))
    records=[]
    picked=set()
    images=[]
    for kind,candidates in zip(("T_only","both","harm"),categories):
        if not candidates:
            records.append(dict(kind=kind,count=0,pair_id="",human="",m0="",T9="",S9="",shown="",shown_margins=""))
            continue
        # Smallest ID, prefer a not-yet-shown pair if the categories permit it.
        i,families=next((x for x in candidates if x[0] not in picked),candidates[0])
        picked.add(i)
        views=["O"]
        for f in "TS":
            rank=min if kind=="harm" else max
            views.append(rank(POOLS[f],key=lambda v:margin(i,v)))
        records.append(dict(kind=kind,count=len(candidates),pair_id=i,human=pairs[i]["human_winner_P0"],
                            m0=margin(i,"O"),T9=detail[i,"T"]["mean_margin"],S9=detail[i,"S"]["mean_margin"],
                            shown=" ".join(views),shown_margins=" ".join(str(margin(i,v)) for v in views)))
        sheet=Image.new("RGB",(6*231,300),"white")
        draw=ImageDraw.Draw(sheet)
        font_file=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font=ImageFont.truetype(str(font_file),15) if font_file.is_file() else ImageFont.load_default()
        draw.rectangle((0,0,sheet.width,32),fill="#e6eff0")
        for j,v in enumerate(views):
            for si,side in enumerate("AB"):
                im=view(original(args.pilot,pairs[i],side),v)
                tensor=model_input(im).contiguous()
                assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest()==data[i,side,v]["tensor_sha"]
                array=((tensor*std+mu).permute(1,2,0).numpy()*255).round().clip(0,255).astype("uint8")
                x=(j*2+si)*231
                sheet.paste(Image.fromarray(array),(x+3,68))
                draw.text((x+3,40),f"{side}   {v}   m={margin(i,v):+.6f}",fill="#193346",font=font)
        label={"T_only":"Translation-only correction","both":"Both correct","harm":"Geometry introduces an error"}[kind]
        draw.text((6,6),f"{i}    {label}     Human winner: {pairs[i]['human_winner_P0']}     O: {margin(i,'O'):+.6f}     T (K=9): {float(detail[i,'T']['mean_margin']):+.6f}     S (K=9): {float(detail[i,'S']['mean_margin']):+.6f}",fill="#183744",font=font)
        images.append(sheet)
    dest=args.root/"figures"
    dest.mkdir(exist_ok=True)
    write(args.root/"case_figure_data.csv",records)
    if images:
        sheet=Image.new("RGB",(images[0].width,sum(im.height for im in images)),"white")
        for j,im in enumerate(images):
            sheet.paste(im,(0,j*im.height))
        sheet.save(dest/"cases.jpg",quality=95,subsampling=0)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("pilot","pairs","root"):
        p.add_argument("--"+field,type=Path,required=True)
    build(p.parse_args())
