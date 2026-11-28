"""Evaluate all completed strategies on exactly the same GT pixels; save image-level values in metres."""
from concurrent.futures import ProcessPoolExecutor
from data import *
from coverage import edge_path
from evaluate import measure, fuse

def read_maps(r):
    d=cache(r);info=get(d/"inference.json")
    with np.load(d/"predictions.npz") as f:
        parts={m:f[m] for m in ("P","R")}
        if r["year"]!="2014":
            b=f["B"];parts["T"]=f["T"]
    if r["year"]=="2014":
        with np.load(old.work(legacy_row(r))/"predictions.npz") as ck:
            b=ck["NB"];parts["T"]=ck["NW"]
    return b,parts,info

def maps(r,requested):
    b,parts,j=read_maps(r)
    boxes=[v["box"] for v in j["layouts"]]
    for m in requested:
        if m=="B":
            yield m,b
            continue
        name="T" if m in ("C","T25") else m.split("_")[0]
        scales="T" if m.endswith("_fixed") else name
        local=fuse(b,parts[name],boxes,j["k"][scales])
        if m=="C":est=local
        elif m=="T25":est=.75*b+.25*local
        else:est=(b+local)*.5
        yield m,est

def one(r):
    d=cache(r);j=get(d/"inference.json")
    assert sha(d/"predictions.npz")==j["prediction_sha256"]
    z=metric(r);valid=np.isfinite(z)&(z>0)&(z<=20)
    with np.load(edge_path(r)) as f:e=dict(f)
    mask=np.ones(len(e["jump"]),bool)
    sets=dict(all=(mask,np.unique(e["near"]),np.unique(e["far"])))
    basic={k:r[k] for k in ("year","id","cluster")}
    metrics=[];movement=[];vis=[]
    for m,p in maps(r,("B","T","T25","C","P","R","P_fixed","R_fixed")):
        metrics.append(dict(**basic,method=m,**measure(p,z,valid,e,sets,{})))
        if m=="B":
            orig=p
        if m=="T":
            for side,ii in zip(("near","far"),sets["all"][1:]):
                bb=orig.ravel()[ii].astype(float);tt=p.ravel()[ii].astype(float);gt=z.ravel()[ii].astype(float)
                dx=tt-bb;err=abs(tt-gt)-abs(bb-gt)
                for direction,keep in (("nearer",dx<0),("farther",dx>0),("unchanged",dx==0)):
                    n=int(keep.sum())
                    movement.append(dict(**basic,side=side,direction=direction,total=len(ii),count=n,
                        fraction=n/len(ii) if len(ii) else None,improved=int((err[keep]<0).sum()),
                        worsened=int((err[keep]>0).sum()),tied=int((err[keep]==0).sum()),
                        err_change_mm=1000*np.mean(err[keep]) if n else None,depth_shift_mm=1000*np.mean(dx[keep]) if n else None))
    roll={row["method"]:row for row in metrics}
    if r["year"]=="2014":
        legacy={v["strategy"]:v for v in read(LEGACY/"per_image.csv") if v["condition"]=="N" and v["id"]==r["id"]}
        for meth,source in (("B","B"),("T","T"),("C","local_scaled")):
            for k in ("all_mae","all_near_mae","all_far_mae","all_jump_mae","global_mae"):
                assert abs(roll[meth][k]-float(legacy[source][k]))<2e-6
    diffs={}
    for a,b in (("T","R"),("B","R"),("P","R"),("P_fixed","R_fixed"),("T","P"),("B","T"),("T25","R"),("B","T25"),("T","T25")):
        for suffix in ("all_near_mae","all_far_mae","all_mae","all_jump_mae","global_mae","all_near_signed","all_far_signed"):
            if suffix in roll[a]:
                diffs[f"{a}_minus_{b}_{suffix}_mm"]=1000*(roll[a][suffix]-roll[b][suffix])
    if mask.any():
        diffs["posthoc_nearB_farT_mae_mm"]=500*(roll["B"]["all_near_mae"]+roll["T"]["all_far_mae"])
    for i,jj in enumerate(j["layouts"]):
        h,w=j["base_hw"];xx,yy,u,v=jj["box"]
        yy0,xx0=jj["core_hw"];yy1,xx1=jj["canvas_hw"];l,t,rr,bt=jj["source"]
        vis.append(dict(**basic,corner=i,box=" ".join(map(str,jj["box"])),encoded_core=" ".join(map(str,jj["net_box"])),
            real_source_box=" ".join(map(str,jj["source"])),source_fov_x=(rr-l)/z.shape[1],source_fov_y=(bt-t)/z.shape[0],
            core_scale_x=xx0/(u-xx)/(w/z.shape[1]),core_scale_y=yy0/(v-yy)/(h/z.shape[0]),
            core_h=yy0,core_w=xx0,canvas_h=yy1,canvas_w=xx1,k_T=j["k"]["T"][i],k_R=j["k"]["R"][i],k_P=j["k"]["P"][i],
            **j["input_audit"][i]))
    time_rows=[dict(**basic,method=m,ms=j["ms"][m],peak_MiB=j["peak_MiB"][m],uuid=j["model"]["uuid"],
        provenance=("old native B/T" if m in ("B","T") else "old B plus paired new stage") if r["year"]=="2014" else "new shared baseline and stage",
        blend_timing="component estimate" if m=="T25" else "measured") for m in ("B","T","T25","P","R")]
    index=dict(**basic,path=str(d/"predictions.npz"),prediction_sha256=j["prediction_sha256"],edge_sha256=sha(edge_path(r)),
        rgb_sha256=r["rgb_sha256"],depth_sha256=r["depth_sha256"],calibration_sha256=r["calib_sha256"],
        new=j["new"],reused=j["prior"],inference_sha256=sha(d/"inference.json"),
        shared_legacy_path=str(old.work(legacy_row(r))/"predictions.npz") if r["year"]=="2014" else "")
    return metrics,[dict(**basic,**diffs)],movement,vis,time_rows,[index]

if __name__=="__main__":
    with ProcessPoolExecutor(4) as ex:out=list(ex.map(one,included()))
    for i,name in enumerate(("per_scene","paired_differences","pixel_movements","windows","latencies","cache_index")):
        write(ROOT/(name+".csv"),[r for result in out for r in result[i]])
