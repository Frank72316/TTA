from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from study import PILOT, ROOT, FAMILIES, VERSIONS

def original(pair,side):
    path=Path(pair["image_"+side])
    with ZipFile(PILOT/"data"/(path.parent.name+".zip")) as z, Image.open(BytesIO(z.read(path.name))) as im:
        return im.convert("RGB")

def translate(rgb,dx,dy):
    w,h=rgb.size
    px,py=abs(dx),abs(dy)
    padded=np.pad(np.asarray(rgb),((py,py),(px,px),(0,0)),mode="reflect")
    return Image.fromarray(padded[py-dy:py-dy+h,px-dx:px-dx+w])

def make_view(rgb,v):
    assert rgb.mode=="RGB"
    if v=="O": return rgb, "original"
    if v[0] in ("B","C"):
        f=int(v[1:])/100
        op=ImageEnhance.Brightness if v[0]=="B" else ImageEnhance.Contrast
        return op(rgb).enhance(f),str(f)
    if v[0]=="J":
        buf=BytesIO()
        rgb.save(buf,"JPEG",quality=int(v[1:]),subsampling=2,optimize=False,progressive=False)
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("RGB"),f"q={v[1:]};4:2:0"
    # Half-up rounding of 3% of the corresponding width/height, in integers.
    sx,sy=((3*n+50)//100 for n in rgb.size)
    dx,dy={"Tleft":(-sx,0),"Tright":(sx,0),"Tup":(0,-sy),"Tdown":(0,sy)}[v]
    return translate(rgb,dx,dy),f"dx={dx};dy={dy}"

def final_rgb(transform,view):
    import torch
    normalizer=transform.transforms[-1].normalize
    t=transform(view)
    mean=torch.tensor(normalizer.mean)[:,None,None]
    std=torch.tensor(normalizer.std)[:,None,None]
    arr=(t*std+mean).permute(1,2,0).numpy()
    return arr  # retain bicubic overshoot; only the displayed PNG is clipped

def preview(images,side):
    h,w=images["O"].shape[:2]
    cell_h=h+24
    sheet=Image.new("RGB",(5*(w+4),5*cell_h),(248,248,248))
    draw=ImageDraw.Draw(sheet)
    for y,vs in enumerate([["O",*pool] for pool in FAMILIES.values()]+[["O",*FAMILIES["J"]]]):
        for x,v in enumerate(vs):
            arr=images[v] if y<4 else 8*np.abs(images[v]-images["O"])
            im=Image.fromarray(np.uint8(np.rint(255*np.clip(arr,0,1))))
            at=(x*(w+4),y*cell_h)
            draw.text(at,(v if y<4 else v+" |diff| x8"),fill="black")
            sheet.paste(im,(at[0],at[1]+20))
    (ROOT/"figures").mkdir(exist_ok=True)
    sheet.save(ROOT/"figures"/f"input_001_{side}.png")

