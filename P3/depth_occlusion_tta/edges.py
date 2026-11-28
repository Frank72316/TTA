"""GT-only coverage pass. Uses no model predictions or errors."""
import argparse,concurrent.futures
import numpy as np
from io_data import *

def axis_edges(z,valid,scale,vertical):
    h,w=z.shape
    pair=valid[:,:-1]&valid[:,1:]
    delta=z[:,1:]-z[:,:-1]
    y,x=np.where(pair&(np.abs(delta)>=.15))
    stat=dict(adjacent=h*(w-1),both_valid=int(pair.sum()),coarse_jump=len(x))
    # interval midpoint is at x+0.5 in pixel-index coordinates
    offsets=np.array([1,2,3])*scale
    left=np.floor(x[:,None]+1-offsets).astype(int)
    right=np.floor(x[:,None]+1+offsets).astype(int)
    inside=(left.min(axis=1)>=0)&(right.max(axis=1)<w)
    y,x,left,right=y[inside],x[inside],left[inside],right[inside]
    invalid=np.pad(np.cumsum(~valid,axis=1,dtype=np.int32),((0,0),(1,0)))
    ok=(invalid[y,right.max(axis=1)+1]-invalid[y,left.min(axis=1)])==0
    stat.update(in_bounds=len(x),no_holes=int(ok.sum()))
    y,x,left,right=y[ok],x[ok],left[ok],right[ok]
    l=z[y[:,None],left];r=z[y[:,None],right]
    lm=np.median(l,axis=1);rm=np.median(r,axis=1);near_left=lm<rm
    nm=np.minimum(lm,rm);fm=np.maximum(lm,rm);thr=np.maximum(.15,.1*nm)
    jump=delta[y,x]
    ok=((fm-nm)>=thr)&(np.abs(jump)>=thr)&((jump>0)==near_left)
    stable=(np.abs(l-lm[:,None]).max(axis=1)<=np.maximum(.07,.05*lm))&(np.abs(r-rm[:,None]).max(axis=1)<=np.maximum(.07,.05*rm))
    stat["consistent_jump"]=int(ok.sum());ok&=stable;stat["stable"]=int(ok.sum())
    y,x,left,right,nm,fm,near_left=(a[ok] for a in (y,x,left,right,nm,fm,near_left))
    signs=np.zeros((h,w-1),np.int8)
    signs[y,x]=np.where(near_left,1,-1)
    step=np.where(near_left,-1,1);start=np.where(near_left,x,x+1)
    alive=np.ones(len(x),bool);count=np.zeros(len(x),int)
    cap=int(np.ceil(24*scale));tol=np.maximum(.07,.05*nm)
    for k in range(cap):
        p=start+step*k
        bounded=p.clip(0,w-1)
        alive&=(p>=0)&(p<w)&valid[y,bounded]&(abs(z[y,bounded]-nm)<=tol)
        count+=alive
    far_edge=x+step*count
    at=far_edge.clip(0,w-2);span=count/scale
    narrow=(span>=4)&(span<=12)&(far_edge==at)&(signs[y,at]==-signs[y,x])
    wide=count==cap
    def flat(xx):
        yy=y.reshape((-1,)+(1,)*(xx.ndim-1))
        return (xx*h+yy if vertical else yy*w+xx).astype("i4")
    near=np.where(near_left[:,None],left,right)
    far=np.where(near_left[:,None],right,left)
    return dict(anchor=flat(x),axis=np.full(len(x),int(vertical),np.int8),near_direction=step.astype("i1"),
                near=flat(near),far=flat(far),span=span.astype("f4"),censored=wide,
                width=np.where(narrow,1,np.where(wide,2,0)).astype("i1"),jump=(fm-nm)),stat

def geometry(z,hw):
    h,w=z.shape;valid=np.isfinite(z)&(z>0)&(z<=20)
    output=[];counts=[]
    for axis in (0,1):
        d,c=axis_edges(z.T if axis else z,valid.T if axis else valid,(h/hw[0] if axis else w/hw[1]),axis)
        output.append(d);counts.append(c)
    data={k:np.concatenate([r[k] for r in output]) for k in output[0]}
    c={k:sum(t[k] for t in counts) for k in counts[0]}
    return data,c
def synthetic():
    z=np.full((120,180),4.,np.float32);z[:,40:48]=2;z[:,90:145]=2
    g,c=geometry(z,(120,180))
    assert c["stable"]==480 and sum(g["width"]==1)==240 and sum(g["width"]==2)==240
    assert np.all(z.ravel()[g["near"]]==2) and np.all(z.ravel()[g["far"]]==4)
    assert all(g["axis"]==0)
    z[:,42]=np.nan;g,_=geometry(z,(120,180))
    assert not np.any((g["anchor"]%180)==39)
    assert boxes(101,63)[-1]==(40,25,101,63)
    assert .001*200*1000/(40+10)==4.
    return True
def run_one(r):
    from transformers import AutoImageProcessor
    pp=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    im=rgb(r);z=depth(r)
    assert z.shape==im.size[::-1]
    hw=pp(images=im,return_tensors="np").pixel_values.shape[-2:]
    data,c=geometry(z,hw)
    for name,img in [("N",im),("L",degrade(im,hw))]:
        a=np.array(img,dtype="f4")/255
        lum=a[:,:,0]*.2126+a[:,:,1]*.7152+a[:,:,2]*.0722
        nf=np.median(lum.ravel()[data["near"]],axis=1);ff=np.median(lum.ravel()[data["far"]],axis=1)
        data["contrast_"+name]=np.abs(ff-nf)
    c.update(dataset=r["dataset"],id=r["id"],height=z.shape[0],width=z.shape[1],base_h=hw[0],base_w=hw[1],
             native_pixels=z.size,depth_positive_finite=int((np.isfinite(z)&(z>0)).sum()),
             out_of_model_range=int((np.isfinite(z)&(z>20)).sum()),valid_pixels=int(((z>0)&(z<=20)).sum()))
    for label,select in [("all",np.ones(len(data["width"]),bool)),("narrow",data["width"]==1),("wide",data["width"]==2)]:
        c[label+"_edges"]=int(select.sum())
        for side in ("near","far"):
            c[label+"_"+side+"_pixels"]=len(np.unique(data[side][select]))
    c["both_eligible"]=int(min(c[k+"_pixels"] for k in ("narrow_near","narrow_far","wide_near","wide_far"))>=20)
    d=work(r);d.mkdir(parents=True,exist_ok=True)
    p=d/"edges.npz";tmp=p.with_suffix(".tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,**data)
    tmp.replace(p);c["edges_sha256"]=sha(p);c["gt_sha256"]=sha(paths(r)[1]);c["calibration_sha256"]=sha(paths(r)[2])
    c["geometry_source"]=sha(Path(__file__));save(d/"coverage.json",c)
    return c
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--jobs",type=int,default=6);args=ap.parse_args()
    assert synthetic()
    rows=[r for r in read(HERE/"expected.csv") if all(p.exists() for p in paths(r))]
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as ex:result=list(ex.map(run_one,rows))
    write(HERE/"coverage.csv",result)
