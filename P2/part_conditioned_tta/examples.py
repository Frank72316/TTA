"""Render the prespecified cases using exact, denormalized GPU-run inputs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from prepare import ROOT, read

def font(size,bold=False):
    face="DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(face,size)

def draw_cases(out):
    ids=json.loads((out/"validation.json").read_text())["example_pairs"]
    questions=defaultdict(dict)
    for r in read(out/"questions.csv"):questions[r["pair_id"]][r["query"]]=r
    w,h=774,622
    canvas=Image.new("RGB",(w,h*len(ids)), "#ffffff"); d=ImageDraw.Draw(canvas)
    for n,pid in enumerate(ids):
        top=n*h;qs=questions[pid];r=qs["crown"]
        good=sum(q["gold"]==q["matched_pick"] for q in qs.values())
        d.rectangle((0,top,w-1,top+35),fill="#17475a")
        d.text((12,top+5),f"{pid}    |    Matched-view TTA: {good} of 2 correct",font=font(19,True),fill="white")
        for j,title in enumerate(["Original input","Crown window","Wing window"]):
            d.text((63+233*j,top+42),title,font=font(16,True),fill="#23323c")
        for i,side in enumerate("AB"):
            y=top+68+i*208
            d.text((18,y+78),side,font=font(23,True),fill="#183f55")
            for j,v in enumerate(("original","crown","wing")):
                path=out/"display_cache"/(r["image_"+side]+"_"+v+".png")
                with Image.open(path) as im:canvas.paste(im.resize((200,200),Image.Resampling.LANCZOS),(63+j*233,y))
        for i,q in enumerate(("crown","wing")):
            t=qs[q];y=top+486+i*63
            d.text((16,y),f"The bird has {t['color']} plumage on its {q}.",font=font(17,True),fill="#1c3748")
            choices="     ".join(f"{name}  {t[key+'_pick']}" for name,key in
                    [("Original","original"),("Center","center"),("Matched","matched"),("Swapped","swapped")])
            d.text((17,y+25),f"CUB  {t['gold']}     |     {choices}",font=font(15),fill="#222222")
    canvas.save(out/"cases.jpg",quality=93,subsampling=0)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT);draw_cases(p.parse_args().out)
