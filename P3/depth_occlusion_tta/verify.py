"""CPU-only independent verification using direct gathers at sampled pixels."""
import struct
from io_data import *

def sample_T(indices,b,wins,rect,ks):
    yy,xx=np.divmod(indices,b.shape[1]);den=np.zeros(len(indices));total=np.zeros(len(indices))
    for (x,y,u,v),w,k in zip(rect,wins,ks):
        ok=(xx>=x)&(xx<u)&(yy>=y)&(yy<v)
        lx,ly=xx[ok]-x,yy[ok]-y
        dist=np.minimum.reduce([(lx+.5)/(u-x),(u-x-lx-.5)/(u-x),(ly+.5)/(v-y),(v-y-ly-.5)/(v-y)])
        weight=(10*dist).clip(.05,1)
        total[ok]+=w[ly,lx]*weight*k;den[ok]+=weight
    assert (den>0).all()
    return (b.ravel()[indices]+total/den)/2
def check_window_coordinates():
    from predict import stitch
    yy,xx=np.indices((63,101))
    src=np.where((xx>=40)&(xx<48),2.,4.)+.001*yy+.0001*xx
    src=src.astype("float32")
    rect=boxes(src.shape[1],src.shape[0])
    scales=[.75,1.,1.25,1.5]
    windows=[src[y:v,x:u]/s for (x,y,u,v),s in zip(rect,scales)]
    fused,measured=stitch(src,windows,rect)
    np.testing.assert_allclose(measured,scales,rtol=2e-7)
    np.testing.assert_allclose(fused,src,rtol=4e-7)
    gathered=sample_T(np.arange(src.size),src,windows,rect,measured)
    np.testing.assert_allclose(gathered,src.ravel(),rtol=2e-7)
    return float(np.max(np.abs(fused-src)))

def main():
    synthetic_residual=check_window_coordinates()
    rng=np.random.default_rng(804)
    rows=[];errors=[];pfm_checks=0
    output={(r["id"],r["condition"],r["strategy"]):r for r in read(HERE/"per_image.csv")}
    strata=[];contrast=[]
    for r in read(HERE/"expected.csv"):
        d=work(r)
        if not (d/"edges.npz").is_file():continue
        z=depth(r)
        if r["dataset"]=="middlebury":
            im,fp,cal=paths(r)
            v=dict(l.strip().split("=",1) for l in cal.read_text().splitlines() if "=" in l)
            with fp.open("rb") as f:
                assert f.readline().strip()==b"Pf"
                w,h=map(int,f.readline().split());endi="<" if float(f.readline())<0 else ">"
                start=f.tell()
                for y,x in zip(rng.integers(h,size=64),rng.integers(w,size=64)):
                    f.seek(start+4*((h-1-int(y))*w+int(x)))
                    val=struct.unpack(endi+"f",f.read(4))[0];den=val+float(v["doffs"])
                    if np.isfinite(val) and den>0:
                        metric=(float(v["baseline"])/1000)*float(v["cam0"][1:].split()[0])/den
                        assert np.isclose(metric,z[y,x],rtol=3e-7)
                    else:assert not np.isfinite(z[y,x])
                    pfm_checks+=1
        with np.load(d/"edges.npz") as g:e=dict(g)
        if not len(e["axis"]):continue
        for code,name in [(0,"other"),(1,"narrow"),(2,"wide")]:
            m=e["width"]==code
            if m.any():contrast.append(dict(**r,group=name,edges=int(m.sum()),median_span=float(np.median(e["span"][m])),
                        right_censored=int(e["censored"][m].sum()),median_GT_jump_m=float(np.median(e["jump"][m])),
                        mean_contrast_N=float(e["contrast_N"][m].mean()),mean_contrast_L=float(e["contrast_L"][m].mean())))
        info=get(d/"inference.json");side=[np.unique(e[s]) for s in ("near","far")]
        scores={}
        with np.load(d/"predictions.npz") as ck:
            for c in ("N","L"):
                b,ps=ck[c+"B"],ck[c+"W"]
                for key in ("B","T"):
                    mae=[]
                    for ii in side:
                        est=b.ravel()[ii] if key=="B" else sample_T(ii,b,ps,info["boxes"],info["conditions"][c]["k"])
                        mae.append(np.mean(abs(est.astype("f8")-z.ravel()[ii])))
                    scores[c+key]=sum(mae)/2
                    errors.append(abs(scores[c+key]-float(output[r["id"],c,key]["all_mae"])))
        rows.append(dict(id=r["id"],dataset=r["dataset"],G_N=(scores["NB"]-scores["NT"])*1000,G_L=(scores["LB"]-scores["LT"])*1000))
    assert max(errors)<2e-6
    summary={(r["dataset"],r["metric"]):r for r in read(HERE/"summary.csv")}
    examples=read(HERE/"cases.csv")
    for ds in set(r["dataset"] for r in rows):
        rr=[r for r in rows if r["dataset"]==ds]
        vals=np.array([[r["G_N"],r["G_L"],r["G_N"]-r["G_L"]] for r in rr])
        # Image is the cluster for all available formal cases. Extend with known room IDs if provided.
        groups={}
        for i,r in enumerate(rr):
            label=get(paths(r)[2]).get("room_id") if ds=="ibims" else r["id"]
            groups.setdefault(label or r["id"],[]).append(i)
        centres=np.stack([vals[indices].mean(axis=0) for indices in groups.values()])
        gen=np.random.default_rng(2026);draws=[]
        for _ in range(10000):
            draws.append(centres[gen.choice(len(centres),size=len(centres))].mean(axis=0))
        for j,k in enumerate(("G_N","G_L","delta_G")):
            vals2=[centres[:,j].mean(),*np.quantile(np.array(draws)[:,j],[.025,.975])]
            assert max(abs(a-float(summary[ds,k][b])) for a,b in zip(vals2,("mean","lo","hi")))<.002
        g=vals[:,0];idx=[np.argmax(g),np.argmin(abs(g-np.median(g))),np.argmin(g)]
        assert [rr[i]["id"] for i in idx]==[c["id"] for c in examples if c["dataset"]==ds]
    write(HERE/"edge_diagnostics.csv",contrast)
    save(HERE/"verification.json",dict(images=len(rows),mae_recalculations=len(errors),pfm_direct_file_offsets=pfm_checks,
             max_MAE_difference_m=max(errors),synthetic_window_alignment_max_m=synthetic_residual,
             bootstrap_draws=10000,case_IDs_verified=True))
if __name__=="__main__":main()
