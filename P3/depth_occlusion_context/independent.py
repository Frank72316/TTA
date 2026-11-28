"""Independent pointwise reconstruction and two-stage paired inference, CPU only."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from data import *
from coverage import edge_path

def recompute(r):
    pos=cache(r);j=get(pos/"inference.json")
    assert sha(pos/"predictions.npz")==j["prediction_sha256"]
    assert [sha(ROOT/n) for n in ("views.py","score.py")]==j["signature"]["code"]
    with np.load(edge_path(r)) as f:
        ids=[np.unique(f[side]) for side in ("near","far")]
    q=metric(r);yy,xx=np.divmod(np.concatenate(ids),q.shape[1])
    true=q[yy,xx].astype("float64")
    if r["year"]!="2014":
        p,d,c=sources(r)
        counts=np.asarray(Image.open(d))[yy,xx].astype("float64")
        assert np.all(counts>0)
        np.testing.assert_allclose(true,598.4/(int(c.read_text())+counts),rtol=1e-7)
    with np.load(pos/"predictions.npz") as f:
        ws={m:f[m] for m in ("R","P")}
        if r["year"]!="2014":
            b=f["B"];ws["T"]=f["T"]
    if r["year"]=="2014":
        with np.load(old.work(legacy_row(r))/"predictions.npz") as f:
            b=f["NB"];ws["T"]=f["NW"]
    bp=b[yy,xx].astype("float64")
    curves={"B":bp}
    for m,parts in ws.items():
        numer=np.zeros((2,len(xx)));den=np.zeros(len(xx))
        for i,p in enumerate(parts):
            x,y,u,v=j["layouts"][i]["box"]
            nx=u-x;ny=v-y;central=lambda n:np.nonzero((np.arange(n)+.5>=.1*n)&(np.arange(n)+.5<=.9*n))[0]
            inner=np.ix_(central(ny),central(nx))
            aa=b[y:v,x:u][inner];bb=p[inner];ok=(aa>0)&(bb>0)&np.isfinite(aa)&np.isfinite(bb)
            scale=np.clip(np.median(aa[ok]/bb[ok]),.5,2) if np.any(ok) else 1
            assert abs(scale-j["k"][m][i])<1e-6
            keep=(xx>=x)&(xx<u)&(yy>=y)&(yy<v)
            lx,ly=xx[keep]-x,yy[keep]-y
            dist=np.minimum.reduce([(lx+.5)/nx,(nx-lx-.5)/nx,(ly+.5)/ny,(ny-ly-.5)/ny])
            wt=np.clip(dist*10,.05,1)
            wp=p[ly,lx].astype("float64")*wt
            numer[0,keep]+=wp*scale;numer[1,keep]+=wp*j["k"]["T"][i];den[keep]+=wt
        assert np.all(den>0)
        c=numer[0]/den
        curves[m]=.5*(bp+c)
        if m=="T":
            curves["C"]=c;curves["T25"]=.75*bp+.25*c
        else:curves[m+"_fixed"]=(bp+numer[1]/den)/2
    result={}
    for m,est in curves.items():
        err=est-true;ne,fa=np.split(err,[len(ids[0])])
        result[m]=[abs(ne).mean(),abs(fa).mean(),(abs(ne).mean()+abs(fa).mean())/2,ne.mean(),fa.mean()]
    return r,result

def main():
    rows=included();assert len(rows)==len({key(r) for r in rows})==50
    truth={(key(r),r["method"]):r for r in read(ROOT/"per_scene.csv")}
    published={(r["batch"],r["metric"]):r for r in read(ROOT/"summary.csv") if r["method"]=="paired"}
    checks=[];stats=[];maxres=0
    cols=["all_near_mae","all_far_mae","all_mae","all_near_signed","all_far_signed"]
    with ProcessPoolExecutor(3) as ex:data=list(ex.map(recompute,rows))
    for r,result in data:
        for m,vals in result.items():
            row=truth[key(r),m];checks.extend(abs(v-float(row[c])) for v,c in zip(vals,cols))
    assert max(checks)<2e-6
    for batch,years in (("2014",["2014"]),("2005+2006",["2005","2006"]),("2005",["2005"]),("2006",["2006"])):
        by=defaultdict(list)
        for r,vals in data:
            if r["year"] in years:
                by[r["cluster"]].append([1000*(vals[a][col]-vals[b][col]) for a,b,col in
                    (("T","R",0),("B","R",0),("P","R",0),("P_fixed","R_fixed",0),("B","R",1),("T","R",1),("B","R",2))])
        cluster=np.stack([np.mean(by[k],axis=0) for k in sorted(by)])
        gen=np.random.default_rng(2026);replicas=[]
        for _ in range(10000):
            replicas.append(cluster[gen.choice(len(cluster),len(cluster))].mean(axis=0))
        estim=np.vstack([cluster.mean(axis=0),np.quantile(replicas,[.025,.975],axis=0)])
        for k,field in enumerate(("T_minus_R_all_near_mae_mm","B_minus_R_all_near_mae_mm","P_minus_R_all_near_mae_mm",
                "P_fixed_minus_R_fixed_all_near_mae_mm","B_minus_R_all_far_mae_mm","T_minus_R_all_far_mae_mm","B_minus_R_all_mae_mm")):
            rr=published[batch,field]
            assert int(rr["clusters"])==len(cluster) and sum(map(len,by.values()))==int(rr["scenes"])
            residual=np.abs(estim[:,k]-[float(rr[col]) for col in ("mean","lo","hi")])
            maxres=max(maxres,float(residual.max()));stats.append(field)
    assert maxres<.002
    idx=read(ROOT/"cache_index.csv")
    counts=get(ROOT/"worker_0.json"),get(ROOT/"worker_1.json")
    assert sum(int(r["new"]) for r in idx)==sum(r["new"]+r["hits"] for r in counts)==535
    assert sum(int(r["reused"]) for r in idx)==115
    for shard,rec in enumerate(counts):
        assert rec["new"]+rec["hits"]==sum(int(r["new"]) for r in idx[shard::2])
        if rec["hits"]==0:assert rec["prior"]==sum(int(r["reused"]) for r in idx[shard::2])
    syn=sum(v["forwards"] for i in range(2) for f in ("synthetic","blend_time") for v in get(ROOT/f"{f}_{i}.json")["rows"])
    assert syn==260
    errs={(key(r),r["method"]):float(r["all_near_mae"]) for r in truth.values()}
    oldrows=[r for r in rows if r["year"]=="2014"]
    near_worse=sum(errs[key(r),"T"]>errs[key(r),"B"] for r in oldrows)
    far_better=sum(float(truth[key(r),"T"]["all_far_mae"])<float(truth[key(r),"B"]["all_far_mae"]) for r in oldrows)
    assert (near_worse,far_better)==(22,20)
    save(ROOT/"verification.json",dict(images=len(data),per_method_metrics_checked=len(checks),max_pixelwise_reconstruction_error_m=max(checks),
        paired_statistics=len(stats),maximum_CI_or_mean_difference_mm=maxres,legacy_near_worsened=near_worse,legacy_far_improved=far_better,
        formal_new=535,legacy_predictions_reused=115,synthetic_forwards=syn,formal_cache_resume_hits=sum(r["hits"] for r in counts),
        groups={"2014":23,"2005+2006":17,"2005":6,"2006":11},nonfinite_method_failures=0))
if __name__=="__main__":main()
