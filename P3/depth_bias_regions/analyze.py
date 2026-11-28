import csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def run():
    with (HERE/"frame_results.csv").open() as f:r=list(csv.DictReader(f))
    a=lambda k:np.array([float(i[k]) if i[k] else np.nan for i in r])
    assert len(r)==128
    groups=[np.array([j for j,i in enumerate(r) if i["scene"]==s]) for s in sorted({i["scene"] for i in r})]
    assert len(groups)==95
    rng=np.random.default_rng(42);counts=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
    size=counts@np.array([len(g) for g in groups])
    def component(k):
        x=a(k);samples=(counts@np.array([x[g].sum() for g in groups]))/size
        return dict(mean=x.mean(),ci=np.quantile(samples,[.025,.975]).tolist(),positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()))
    info={k:component(k) for k in ("G_mean","G_space","G_old","boundary_weighted","interior_weighted","undefined_weighted")}
    reg={}
    for k in ("boundary","interior","undefined"):
        reg[k]={f:float(np.nanmean(a(k+"_"+f))) for f in ("fraction","gain","mse2","mse_shift","weighted")}
        reg[k]["frames"]=int(np.isfinite(a(k+"_gain")).sum())
    mean=a("G_mean");sp=a("G_space")
    info.update(regions=reg,opposite=int((mean*sp<0).sum()),max_telescope=max(abs(a("telescope_error"))),
        max_prediction_space_residual=max(a("direct_space_error")),max_mean_depth_residual=max(a("mean_depth_check")),
        cases={k:next((dict(frame_id=x["frame_id"],mean=x["G_mean"],space=x["G_space"],total=x["G_old"],scene=x["scene"]) for x in r if sign*float(x["G_space"])>0),None) for k,sign in (("positive",1),("negative",-1))})
    (HERE/"summary.json").write_text(json.dumps(info,indent=2))
if __name__=="__main__":run()
