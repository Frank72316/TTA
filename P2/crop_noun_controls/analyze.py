from pathlib import Path
import sys
import numpy as np
from freeze import HERE,read,put
sys.path.insert(0,str(HERE.parent/"photometric_pairing"))
from measures import dump,quant,table_rates,bootstrap,cases
def run():
    old=HERE.parent/"spatial_crop_tta"; base=read(old/"scores.csv")
    bank={(r["pair_id"],r["side"],r["half"],r["role"]):float(r["score"]) for r in base}
    new=read(HERE/"scores.csv");alt={(r["seed"],r["pair_id"],r["side"],r["half"],r["role"]):float(r["score"]) for r in new}
    assert len(bank)==512 and len(alt)==len(new)
    source={r["pair_id"]:r for r in read(old/"pair_results.csv")}
    details=[];stats={}
    for seed in ("42","43","44"):
        group=[];arms=("true","true_x","true_y","both_alt")
        for mp in read(HERE/"mappings.csv"):
            if mp["seed"]!=seed or not mp["donor"]:continue
            pid=mp["pair_id"];p=source[pid];d=dict(seed=seed,pair_id=pid,donor=mp["donor"],x=mp["x"],y=mp["y"],alt_x=mp["alt_x"],alt_y=mp["alt_y"],n_clear=int(p["n_clear"]),true_left=p["true_left"])
            for arm in arms:
                margins={}
                for side in "AB":
                    def score(half,role):
                        key=(pid,side,half,role)
                        return bank[key] if arm=="true" or arm=="true_"+role else alt[(seed,*key)]
                    margins[side]=score("left","x")-score("right","x")+score("right","y")-score("left","y")
                    d[arm+"_"+side]=margins[side]
                d[arm]=margins[p["true_left"]]-margins[p["true_right"]]
            assert d["true"]==float(p["local_margin"])
            group.append(d);details.append(d)
        ar={a:np.array([r[a] for r in group]) for a in arms};n=len(group);strata={}
        def outcomes(x):return [int((x>0).sum()),int((x<0).sum()),int((x==0).sum())]
        for k in range(3):
            ix=np.array([r["n_clear"]==k for r in group]);strata[k]={a:outcomes(x[ix]) for a,x in ar.items()}
        stats[seed]=dict(n=n,counts={a:outcomes(x) for a,x in ar.items()},strata=strata,
            true_minus_alt_pp=bootstrap(100*((ar["true"]>0).astype(float)-(ar["both_alt"]>0))),
            margin_true_minus_alt=quant(ar["true"]-ar["both_alt"]),
            transitions={a:table_rates(ar["true"][:,None],ar[a][:,None]) for a in arms[1:]},
            examples=cases([p["pair_id"] for p in group],ar["both_alt"][:,None],ar["true"][:,None]))
    assert stats["42"]["counts"]["true"][0]==52
    put(HERE/"pair_results.csv",details);dump(HERE/"summary.json",stats)
if __name__=="__main__":run()
