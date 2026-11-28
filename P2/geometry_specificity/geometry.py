"""All versions start from the same full-resolution RGB image."""
from io import BytesIO
from math import ceil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageEnhance


PHOTO = (.95, .9625, .975, .9875, 1.0125, 1.025, 1.0375, 1.05)
SCALE = (.97, .9775, .985, .9925, 1.0075, 1.015, 1.0225, 1.03)
AXES = dict(left=(-1,0),right=(1,0),up=(0,-1),down=(0,1))
PARAMS = {"O": ("O", 1)}
for family in ("B","C","S"):
    for factor in (SCALE if family=="S" else PHOTO):
        PARAMS[f"{family}{factor:.4f}"] = (family, factor)
for axis in AXES:
    for pct in (1,3):
        PARAMS[f"T_{axis}_{pct}"] = ("T",(axis,pct))
POOLS = {f:tuple(v for v,(family,_) in PARAMS.items() if family==f) for f in "BCTS"}


def original(pilot,pair,side):
    path = Path(pair["image_"+side])
    assert path.parts[:2]==("images","validation")
    with ZipFile(pilot/"data"/(path.parent.name+".zip")) as z, Image.open(BytesIO(z.read(path.name))) as im:
        return im.convert("RGB")


def view(rgb, key):
    family,f = PARAMS[key]
    w,h = rgb.size
    if family=="O":
        return rgb
    if family=="B":
        return ImageEnhance.Brightness(rgb).enhance(f)
    if family=="C":
        return ImageEnhance.Contrast(rgb).enhance(f)
    if family=="T":
        axis,pct=f
        ax,ay=AXES[axis]
        dx=ax*((w*pct+50)//100)
        dy=ay*((h*pct+50)//100)
        px,py=abs(dx),abs(dy)
        arr=np.pad(np.asarray(rgb),((py,py),(px,px),(0,0)),mode="reflect")
        return Image.fromarray(arr[py-dy:py-dy+h,px-dx:px-dx+w])
    # PIL uses edge coordinates, so (w/2,h/2) is the image centre for even and odd sizes.
    px=3+ceil(max(0,(1/f-1)*w/2))
    py=3+ceil(max(0,(1/f-1)*h/2))
    arr=np.pad(np.asarray(rgb),((py,py),(px,px),(0,0)),mode="reflect")
    return Image.fromarray(arr).transform((w,h),Image.Transform.AFFINE,
        (1/f,0,px+w/2*(1-1/f),0,1/f,py+h/2*(1-1/f)),Image.Resampling.BICUBIC)


def check_geometry():
    for w,h in ((201,151),(200,150)):
        yy,xx=np.indices((h,w))
        ramp=Image.fromarray(np.stack([xx,yy,xx*0+100],axis=-1).astype("uint8"))
        def reflect(v,n):
            r=v%(2*n-2)
            return np.minimum(r,2*n-2-r)
        for name in POOLS["T"]:
            axis,pct=PARAMS[name][1]
            ax,ay=AXES[axis]
            dx=ax*((w*pct+50)//100)
            dy=ay*((h*pct+50)//100)
            expected=np.asarray(ramp)[reflect(yy-dy,h),reflect(xx-dx,w)]
            assert np.array_equal(np.asarray(view(ramp,name)),expected)
        for name in POOLS["S"]:
            scale=PARAMS[name][1]
            arr=np.asarray(view(ramp,name)).astype(float)
            sx=reflect((xx-(w-1)/2)/scale+(w-1)/2,w)
            sy=reflect((yy-(h-1)/2)/scale+(h-1)/2,h)
            assert abs(arr[:,:,0]-sx).max()<1.5 and abs(arr[:,:,1]-sy).max()<1.5
            assert np.all(arr[:,:,2]==100)
