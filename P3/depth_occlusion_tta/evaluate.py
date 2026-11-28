"""Offline metrics on frozen positions. Image-level estimates in metres; bootstrap separately."""
import argparse,concurrent.futures
import numpy as np
from scipy.ndimage import distance_transform_edt
from io_data import *

def fuse(base,parts,boxes_,factors):
    den=np.zeros_like(base);z=np.zeros_like(base)
    for p,(x,y,x1,y1),k in zip(parts,boxes_,factors):
        h,w=p.shape
        cx=(np.arange(w)+.5)/w;cy=(np.arange(h)+.5)/h
        wt=np.clip(np.minimum(np.minimum(cx,1-cx)[None,:],np.minimum(cy,1-cy)[:,None])*10,.05,1).astype("f4")
        z[y:y1,x:x1]+=p*k*wt;den[y:y1,x:x1]+=wt
    return z/den
def plane_sets(r,z,hw):
    masks={};folder=paths(r)[1].parent
    if r["dataset"]!="ibims":return masks
    h,w=z.shape
    img=np.asarray(rgb(r),dtype="f4")/255;l=img@np.array([.2126,.7152,.0722],dtype="f4")
    gy,gx=np.gradient(l);mag=np.hypot(gx,gy)
    valid=(z>0)&(z<=20)&np.isfinite(z)
    for name in ("wall","floor","table"):
        path=folder/("mask_"+name+".png")
        if not path.exists():continue
        a=np.asarray(Image.open(path));assert a.shape==z.shape
        region=np.zeros_like(valid)
        for label in np.unique(a[a>0]):
            region|=distance_transform_edt(np.pad(a==label,1),sampling=(hw[0]/h,hw[1]/w))[1:-1,1:-1]>5
        region&=valid
        if not region.any():continue
        ys,xs=np.where(region&(mag>=np.quantile(mag[region],.9)))
        xx=np.abs(gx[ys,xs])>=np.abs(gy[ys,xs]);dx=xx*rounded(2*w/hw[1]);dy=(~xx)*rounded(2*h/hw[0])
        ok=(xs-dx>=0)&(xs+dx<w)&(ys-dy>=0)&(ys+dy<h)
        ys,xs,dx,dy=(t[ok] for t in (ys,xs,dx,dy))
        ok=valid[ys-dy,xs-dx]&valid[ys+dy,xs+dx]&(a[ys-dy,xs-dx]==a[ys,xs])&(a[ys+dy,xs+dx]==a[ys,xs])
        masks[name]=(region,((ys-dy)[ok],(xs-dx)[ok]),((ys+dy)[ok],(xs+dx)[ok]))
    return masks
def edgesets(e,z,r,hw):
    sets={}
    masks={"all":np.ones(len(e["axis"]),bool),"narrow":e["width"]==1,"wide":e["width"]==2}
    path=paths(r)[1].parent/"edges.png"
    if r["dataset"]=="ibims" and path.exists():
        official=np.asarray(Image.open(path))!=0;assert official.shape==z.shape
        dist=distance_transform_edt(~official,sampling=(hw[0]/z.shape[0],hw[1]/z.shape[1]))
        next_pixel=e["anchor"]+np.where(e["axis"]==1,z.shape[1],1)
        masks["official"]=(np.minimum(dist.ravel()[e["anchor"]],dist.ravel()[next_pixel])<=1)
    for g,s in masks.items():
        sets[g]=(s,np.unique(e["near"][s]),np.unique(e["far"][s]))
    return sets
def measure(p,z,valid,e,sets,planes):
    if not np.isfinite(p).all():raise ValueError("non-finite prediction on fixed evaluation positions")
    zf=z.ravel();pf=p.ravel()
    t=(p[valid].astype("f8")-z[valid])
    out=dict(global_mae=np.abs(t).mean(),absrel=(np.abs(t)/z[valid]).mean())
    for g,(s,n,f) in sets.items():
        if not len(n) or not len(f):continue
        ne=pf[n].astype("f8")-zf[n];fe=pf[f].astype("f8")-zf[f]
        out[g+"_near_mae"]=abs(ne).mean();out[g+"_far_mae"]=abs(fe).mean();out[g+"_mae"]=(abs(ne).mean()+abs(fe).mean())/2
        out[g+"_near_signed"]=ne.mean();out[g+"_far_signed"]=fe.mean()
        diff=np.median(pf[e["far"][s]],axis=1).astype("f8")-np.median(pf[e["near"][s]],axis=1)-e["jump"][s]
        out[g+"_jump_mae"]=np.abs(diff).mean();out[g+"_jump_signed"]=diff.mean()
    for name,(mask,i,j) in planes.items():
        out["plane_"+name+"_mae"]=np.mean(abs(p[mask]-z[mask]),dtype=np.float64)
        if len(i[0]):out["plane_"+name+"_texture_false_jump"]=np.mean(abs((p[i]-p[j])-(z[i]-z[j])),dtype=np.float64)
    return out
def one(r):
    d=work(r);info=get(d/"inference.json");cov=get(d/"coverage.json")
    assert sha(d/"predictions.npz")==info["prediction_sha256"] and sha(d/"edges.npz")==cov["edges_sha256"]
    with np.load(d/"edges.npz") as g:e=dict(g)
    z=depth(r);valid=np.isfinite(z)&(z>0)&(z<=20)
    hw=info["conditions"]["N"]["base_hw"];assert hw==[cov["base_h"],cov["base_w"]]
    sets=edgesets(e,z,r,hw);planes=plane_sets(r,z,hw)
    rows=[];wins=[]
    def add(c,s,p):rows.append(dict(**r,condition=c,strategy=s,**measure(p,z,valid,e,sets,planes)))
    with np.load(d/"predictions.npz") as ck:
        for c in ("N","L"):
            b=ck[c+"B"];parts=ck[c+"W"];bxs=info["boxes"];ks=[]
            for j,((x,y,u,v),p) in enumerate(zip(bxs,parts)):
                h,w=p.shape;yy=(np.arange(h)+.5)/h;xx=(np.arange(w)+.5)/w
                sl=np.ix_((yy>=.1)&(yy<=.9),(xx>=.1)&(xx<=.9))
                bc=b[y:v,x:u][sl];wc=p[sl];ok=(bc>0)&(wc>0)&np.isfinite(bc)&np.isfinite(wc)
                rat=bc[ok]/wc[ok];ku=float(np.median(rat)) if rat.size else 1.;k=float(np.clip(ku,.5,2))
                assert abs(k-info["conditions"][c]["k"][j])<1e-6;ks.append(k)
                lr=np.log(rat.astype("f8")) if rat.size else np.array([])
                wh=info["conditions"][c]["window_hw"]
                wins.append(dict(**r,condition=c,window=j,x=x,y=y,x1=u,y1=v,net_h=wh[0],net_w=wh[1],
                      factor_x=wh[1]/w/(hw[1]/z.shape[1]),factor_y=wh[0]/h/(hw[0]/z.shape[0]),positive_ratio_count=rat.size,
                      k_unclipped=ku,k=k,at_lower=int(k==.5),at_upper=int(k==2),fallback=int(not rat.size),
                      median_signed_log_B_over_window=float(np.median(lr)) if rat.size else None,
                      abs_log_before=float(np.median(abs(lr))) if rat.size else None,
                      abs_log_after=float(np.median(abs(lr-np.log(k)))) if rat.size else None))
            add(c,"B",b)
            for method,factors in [("raw",[1]*4),("scaled",ks)]:
                stitched=fuse(b,parts,bxs,factors)
                add(c,"local_"+method,stitched)
                add(c,"T" if method=="scaled" else "fusion_raw",(b+stitched)*.5)
            if c=="N":bn=b;kn=list(ks)
            else:add("counterfactual","BN_kN_Lwindows",(bn+fuse(bn,parts,bxs,kn))*.5)
        if "H" in ck:add("N","H",ck["H"])
    return rows,wins
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--jobs",type=int,default=6);n=ap.parse_args().jobs
    rs=[r for r in read(HERE/"expected.csv") if (work(r)/"inference.json").exists() and (work(r)/"coverage.json").exists()]
    with concurrent.futures.ProcessPoolExecutor(n) as ex:out=list(ex.map(one,rs))
    write(HERE/"per_image.csv",[r for m,w in out for r in m]);write(HERE/"per_window.csv",[r for m,w in out for r in w])
