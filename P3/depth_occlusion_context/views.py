"""Build context canvases on the existing corner crop lattice, without changing its samples."""
import math
from data import *
import predict as ref
import torch
import torch.nn.functional as F

def reflect(pos,n):
    phase=np.mod(pos,2*(n-1))
    return np.minimum(phase,2*(n-1)-phase)

def layouts(size,core_hw):
    w,h=size
    ch,cw=core_hw
    bh,bw=[14*old.rounded(v*4/(3*14)) for v in core_hw]
    layouts_=[]
    for x,y,u,v in old.boxes(w,h):
        ox=bw-cw if x else 0
        oy=bh-ch if y else 0
        dx=(u-x)/cw;dy=(v-y)/ch
        source=(x-ox*dx,y-oy*dy,u+(bw-ox-cw)*dx,v+(bh-oy-ch)*dy)
        assert source[0]>=0 and source[1]>=0 and source[2]<=w and source[3]<=h
        layouts_.append(dict(box=[x,y,u,v],net_box=[ox,oy,ox+cw,oy+ch],source=source,
                        native_per_encoder=[dx,dy],core_hw=[ch,cw],canvas_hw=[bh,bw]))
    return layouts_

def canvas(image,core,layout,which,resample):
    x,y,u,v=layout["box"];a,b,d,e=layout["source"]
    # Enclosing integer tile, then continuous Pillow box. Both arms use identical geometry.
    left,top,right,bottom=math.floor(a),math.floor(b),math.ceil(d),math.ceil(e)
    if which=="R":
        tile=image.crop((left,top,right,bottom))
    else:
        assert which=="P"
        tx=reflect(np.arange(left,right)-x,u-x)
        ty=reflect(np.arange(top,bottom)-y,v-y)
        tile=Image.fromarray(np.asarray(image.crop((x,y,u,v)))[ty[:,None],tx])
    bh,bw=layout["canvas_hw"]
    out=np.array(tile.resize((bw,bh),resample=resample,box=(a-left,b-top,d-left,e-top)))
    xx,yy,xx1,yy1=layout["net_box"]
    out[yy:yy1,xx:xx1]=core
    return out

def tensors(pp,image,which):
    boxes=old.boxes(*image.size)
    rasters=[ref.raster(pp,image.crop(b)).transpose(1,2,0) for b in boxes]
    info=layouts(image.size,rasters[0].shape[:2])
    imgs=[Image.fromarray(canvas(image,q,s,which,pp.resample)) for q,s in zip(rasters,info)]
    batch=pp(images=imgs,do_resize=False,return_tensors="pt").pixel_values
    assert tuple(batch.shape[-2:])==tuple(info[0]["canvas_hw"])
    return batch,info

def extended(pp,model,image,which):
    x,info=tensors(pp,image,which)
    with torch.inference_mode():
        p=model(pixel_values=x.cuda()).predicted_depth
        assert p.dtype==torch.float32 and tuple(p.shape[1:])==tuple(x.shape[-2:])
        parts=[]
        for i,layout in enumerate(info):
            a,b,d,e=layout["net_box"];sx,sy,su,sv=layout["box"]
            parts.append(p[i,b:e,a:d])
        cropped=torch.stack(parts)[:,None]
        maps=F.interpolate(cropped,size=(sv-sy,su-sx),mode="bicubic",align_corners=False)[:,0].cpu().numpy()
    assert np.isfinite(maps).all()
    return maps,x,info

def audit_input(pp,image,records):
    core=ref.tensor(pp,[image.crop(b) for b in old.boxes(*image.size)])
    hashes=[]
    for i,layout in enumerate(records["R"][1]):
        a,b,d,e=layout["net_box"]
        rr=records["R"][0][i];p=records["P"][0][i]
        assert torch.equal(rr[:,b:e,a:d],core[i]) and torch.equal(p[:,b:e,a:d],core[i])
        ext=rr.shape[1]*rr.shape[2]-(d-a)*(e-b)
        hashes.append(dict(core_sha256=ah(core[i].numpy()),R_sha256=ah(rr.numpy()),P_sha256=ah(p.numpy()),
             changed_outer_pixels=int(torch.any(rr!=p,dim=0).sum()),outer_pixels=ext))
    return hashes

def coordinate_test(pp):
    h,w=321,481
    yy,xx=np.indices((h,w))
    img=Image.fromarray(np.uint8(np.stack([xx%251,yy%241,(2*xx+yy)%239],axis=2)))
    res={i:tensors(pp,img,i) for i in ("R","P")}
    assert np.array_equal(reflect(np.arange(-3,8),5),[3,2,1,0,1,2,3,4,3,2,1])
    for info in res["R"][1]:
        x,y,u,v=info["box"];l,t,r,b=info["source"];a,c,d,e=info["net_box"];H,W=info["canvas_hw"]
        for i,j in ((0,0),(d-a-1,e-c-1),(7,9)):
            assert abs((x+(i+.5)*(u-x)/(d-a))-(l+(a+i+.5)*(r-l)/W))<1e-8
            assert abs((y+(j+.5)*(v-y)/(e-c))-(t+(c+j+.5)*(b-t)/H))<1e-8
        # Synthetic coordinate-valued prediction: crop on the output grid then map back exactly as in inference.
        ramp=torch.arange(W,dtype=torch.float32)[None,None,None,:].expand(1,1,H,W)
        crop=ramp[:,:,c:e,a:d]
        warped=F.interpolate(crop,size=(v-y,u-x),mode="bicubic",align_corners=False)[0,0]
        expected=a+(torch.arange(u-x)+.5)*(d-a)/(u-x)-.5
        assert (warped[:,2:-2]-expected[2:-2]).abs().max()<.06
    return audit_input(pp,img,res)
