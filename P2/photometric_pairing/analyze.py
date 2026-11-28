import platform
from time import perf_counter
from pathlib import Path
import numpy as np
from measures import read,write,sha,dump,preference,bootstrap,table_rates,cases
HERE=Path(__file__).resolve().parent
def run():
    tick=perf_counter();src=HERE.parent/"translation";pairs=read(src/"pairs_frozen.csv");scores=read(src/"scores.csv")
    family={}
    for f in "BC":
        params={float(r["param"]):r["view"] for r in scores if r["view"].startswith(f)}
        assert set(params)=={.9,.95,1.05,1.1}
        family[f]=[params[k] for k in (1.05,.95,1.1,.9)]
    ids,m=preference(scores,pairs,["O",*family["B"],*family["C"]])
    assert len(ids)==256 and sum(m[i,"O"]>0 for i in ids)==175
    stats={};details=[]
    for f,views in family.items():
        same=[];cross=[];orig=[];signs=[];v1=[];v2=[]
        for i,pid in enumerate(ids):
            o=m[pid,"O"];p,n,P,N=[m[pid,k] for k in views]
            light=(p-n)/2;heavy=(P-N)/2
            same_sum=np.array([o+p+P,o+n+N]);cross_sum=np.array([o+p+N,o+n+P])
            assert same_sum.sum()==cross_sum.sum()
            a=same_sum/3;b=cross_sum/3
            av1=np.var(a);av2=np.var(b)
            assert np.isclose(a.mean(),b.mean(),rtol=0,atol=1e-16)
            assert np.isclose(av1,(light+heavy)**2/9,rtol=0,atol=1e-18) and np.isclose(av2,(light-heavy)**2/9,rtol=0,atol=1e-18)
            same.append(a);cross.append(b);orig.append([o,o]);signs.append(np.sign(light*heavy));v1.append(av1);v2.append(av2)
            details.append(dict(pair_id=pid,human=pairs[i]["human_winner_P0"],family=f,m0=o,m_plus5=p,m_minus5=n,m_plus10=P,m_minus10=N,
                    d_light=light,d_heavy=heavy,product=light*heavy,same_plus=a[0],same_minus=a[1],cross_plus=b[0],
                    cross_minus=b[1],expected_margin=(2*o+p+n+P+N)/6,var_same=av1,var_cross=av2))
            if i==1: assert len({p["image_"+s] for p in pairs[:2] for s in "AB"})==4
        a,b=np.array(same),np.array(cross)
        stats[f]=dict(var_same=np.mean(v1),var_cross=np.mean(v2),reduction=1-np.mean(v2)/np.mean(v1),
            pos=signs.count(1),neg=signs.count(-1),zero=signs.count(0),
            counts={k:[(x>0).sum()/2,(x<0).sum()/2,(x==0).sum()/2] for k,x in (("same",a),("cross",b))},
            diff_pp=bootstrap(100*((b>0).mean(1)-(a>0).mean(1))),O_to_same=table_rates(orig,a),
            O_to_cross=table_rates(orig,b),same_to_cross=table_rates(a,b),examples=cases(ids,a,b))
    write(HERE/"pair_results.csv",details);dump(HERE/"summary.json",stats)
    trans=[]
    for f,s in stats.items():
        for key in ("O_to_same","O_to_cross","same_to_cross"):
            trans.extend(dict(family=f,comparison=key,transition=k,expected_pairs=v) for k,v in s[key].items())
    write(HERE/"transitions.csv",trans)
    dump(HERE/"environment.json",dict(source_scores_sha=sha(src/"scores.csv"),source_pairs_sha=sha(src/"pairs_frozen.csv"),
        protocol_sha=sha(HERE/"protocol.md"),old_environment_sha=sha(src/"environment.json"),model="frozen HPS v2.1",scores_read=len(scores),
        used_records=256*2*9,python=platform.python_version(),numpy=np.__version__,forwards=0,wall_seconds=perf_counter()-tick,smoke=ids[:2]))
if __name__=="__main__":run()
