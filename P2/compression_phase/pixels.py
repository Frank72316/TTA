from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
PHASE=((0,0),(4,0),(0,4),(4,4))
RADII=tuple(i/10 for i in range(1,21))
def padded(rgb,phase):
    dx,dy=phase; x=16+dx;y=16+dy; w,h=rgb.size
    px=16+(-(w+x+16))%8; py=16+(-(h+y+16))%8
    im=Image.fromarray(np.pad(np.asarray(rgb),((y,py),(x,px),(0,0)),mode="reflect"))
    box=(x,y,x+w,y+h)
    assert np.array_equal(np.asarray(im.crop(box)),np.asarray(rgb))
    return im,box
def jpeg(im,box):
    buf=BytesIO();im.save(buf,"JPEG",quality=85,subsampling=0,optimize=False,progressive=False)
    with Image.open(buf) as enc:return enc.convert("RGB").crop(box)
def blur(im,box,radius):return im.filter(ImageFilter.GaussianBlur(radius)).crop(box)
def visible(tensor,norm,rgb):
    a=tensor.numpy().astype(float)*np.array(norm.std)[:,None,None]+np.array(norm.mean)[:,None,None]
    w,h=rgb.size; scale=224/max(w,h); dw,dh=round(w*scale),round(h*scale)
    py=(a.shape[1]-dh)//2;px=(a.shape[2]-dw)//2
    return a[:,py:py+dh,px:px+dw]
def energy(a):
    y=.299*a[0]+.587*a[1]+.114*a[2]
    return (np.diff(y,axis=0).__pow__(2).mean()+np.diff(y,axis=1).__pow__(2).mean())/2
