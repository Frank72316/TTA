"""Display both candidates and the four local margins for each question at both extrema."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from color_control import ROOT, read

def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",size)

def render(out):
    summary=json.loads((out/"summary.json").read_text())
    questions={(r["pair_id"],r["query"]):r for r in read(out/"questions.csv")}
    pairs={r["pair_id"]:r for r in read(out/"pairs_results.csv")}
    width,height=816,733
    sheet=Image.new("RGB",(width,2*height),"white")
    d=ImageDraw.Draw(sheet)
    for j,which in enumerate(("max","min")):
        top=j*height;pid=summary["extrema"][which];p=pairs[pid]
        d.rectangle((0,top,width,top+46),fill="#203e50")
        title=f"{pid}     {which.upper()} pair mean I  {float(p['pair_I']):+.6f}     |     Color: {p['color']}"
        d.text((14,top+10),title,font=font(20,True),fill="white")
        columns=(65,315,565)
        for col,name in zip(columns,("Original input","Crown window","Wing window")):
            d.text((col,top+53),name,font=font(17,True),fill="#264053")
        for i,side in enumerate("AB"):
            y=top+81+i*201
            d.text((18,y+80),side,font=font(24,True),fill="#203e50")
            for col,v in zip(columns,("original","crown","wing")):
                path=out/"display_cache"/(p["image_"+side]+"_"+v+".png")
                with Image.open(path) as im:
                    sheet.paste(im.resize((195,195),Image.Resampling.LANCZOS),(col,y))
        y=top+486
        d.text((14,y),"Local margins = CUB label winner minus the other candidate",font=font(17,True),fill="#203e50")
        headers=("Query / gold","Matched window","Swapped window","Interaction I")
        for x,title in zip((14,175,414,671),headers):
            d.text((x,y+31),title,font=font(15,True),fill="#23343c")
        for i,q in enumerate(("crown","wing")):
            r=questions[pid,q];line_y=y+60+i*33
            d.text((14,line_y),f"{q} / {r['gold']}",font=font(16),fill="#111111")
            for x,role in ((175,"matched"),(414,"swapped")):
                vals=[float(r[f"{cond}_crop_{role}_d"]) for cond in ("neutral","color")]
                d.text((x,line_y),f"{vals[0]:+.5f}  →  {vals[1]:+.5f}",font=font(15),fill="#111111")
            d.text((671,line_y),f"{float(r['I']):+.6f}",font=font(16,True),fill="#203e50")
        d.text((14,y+134),"Arrows show the change from no color word to the recorded color word.",font=font(16),fill="#344c5c")
        for i,q in enumerate(("crown","wing")):
            r=questions[pid,q]
            choices="     ".join(f"{role}: {r['neutral_tta_'+role+'_pick']} → {r['color_tta_'+role+'_pick']}" for role in ("matched","swapped"))
            d.text((14,y+166+i*26),f"TTA picks, {q}:    {choices}",font=font(16),fill="#23343c")
    sheet.save(out/"cases.jpg",quality=93,subsampling=0)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT);render(p.parse_args().out)
