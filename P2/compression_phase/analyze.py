import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"photometric_pairing"))
from measures import read,write,dump,bootstrap,quant,table_rates,preference,cases,result
def run():
    phases=("00","40","04","44");rows=read(HERE/"scores.csv");pairs=read(HERE.parent/"translation/pairs_frozen.csv")
    ids,m=preference(rows,pairs,["O",*[k+p for k in "JG" for p in phases]])
    ds=[]; orig=[];jpeg=[];blur=[];ranges=[];singles=[]
    for pid in ids:
        o=m[pid,"O"];j=np.array([m[pid,"J"+p] for p in phases]);g=np.array([m[pid,"G"+p] for p in phases])
        assert np.ptp(g)==0
        r=(j+o)/2;b=(g+o)/2
        d=dict(pair_id=pid,m0=o,single_J_range=np.ptp(j),aggregate_J_range=np.ptp(r),m_G=g[0],
               G_aggregate=b[0],G_outcome=result(b[0]),J_correct=int((r>0).sum()),J_wrong=int((r<0).sum()),J_tie=int((r==0).sum()))
        for k,p in enumerate(phases):d["m_J"+p]=j[k];d["aggregate_J"+p]=r[k];d["outcome_J"+p]=result(r[k])
        ds.append(d);orig.append([o]*4);jpeg.append(r);blur.append(b);ranges.append(np.ptp(r));singles.append(np.ptp(j))
    write(HERE/"pair_results.csv",ds)
    o,j,g=map(np.array,(orig,jpeg,blur));jc=(j>0).sum(1);jw=(j<0).sum(1);jt=(j==0).sum(1)
    def selected(mask):return np.array(ids)[mask].tolist()
    stats=dict(single_range=quant(singles),aggregate_range=quant(ranges),both_correct_noncorrect=selected((jc>0)&(jc<4)),
        strict_both_signs=selected((jc>0)&(jw>0)),phase_counts_by_n_correct={str(n):int((jc==n).sum()) for n in range(5)},
        rescue_all=selected((o[:,0]<0)&(jc==4)),rescue_some=selected((o[:,0]<0)&(jc>0)&(jc<4)),
        blur_rescues=selected((o[:,0]<0)&(g[:,0]>0)),overlap=selected((o[:,0]<0)&(g[:,0]>0)&(jc>0)),
        phase_tie_pairs=selected(jt>0),examples=cases(ids,o,j),strategies={},calibration={})
    for name,arr in (("JPEG",j),("Gaussian",g)):
        stats["strategies"][name]=dict(expected=[(arr>0).sum()/4,(arr<0).sum()/4,(arr==0).sum()/4],
            diff_pp=bootstrap(100*((arr>0).mean(1)-(o>0).mean(1))),transitions=table_rates(o,arr))
    calibration=read(HERE/"calibration.csv");radius=json.loads((HERE/"strength.json").read_text())["radius"]
    for name in ("J","G"):
        group=[r for r in calibration if r["family"]==name and (name=="J" or float(r["parameter"])==radius)]
        by={}
        for r in group:by.setdefault((r["pair_id"],r["side"]),[]).append(r)
        vals=np.array([[np.mean([float(r[k]) for r in v]) for k in ("relative_energy","rgb_mse")] for v in by.values()])
        stats["calibration"][name]=dict(energy_quartiles=quant(vals[:,0]),mse_quartiles=quant(vals[:,1]),
            energy_up=int((vals[:,0]>0).sum()),energy_down=int((vals[:,0]<0).sum()))
    dump(HERE/"summary.json",stats)
if __name__=="__main__":run()
