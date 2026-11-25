"""All differences are computed from finite fp16 outputs promoted to float64."""
from collections import Counter
import numpy as np
from common import *

BINS=("r<1","r=1","1<r<2","r=2","r>2","initial_tie")


def pick(v):
    return "A" if v>0 else "B" if v<0 else "tie"


def compare(d0,d):
    if d0==0:
        return "initial_tie"
    if d==0:
        return "new_tie"
    return "flip" if pick(d0)!=pick(d) else "same"


def margins(cells,i,c):
    a0,a1,b0,b1=(float(cells[i,s,c,l]) for s,l in CELLS)
    d0,d1=a0-b0,a1-b1
    delta=d1-d0
    da,db=a1-a0,b1-b0
    dt=(d0+d1)/2
    assert delta==da-db and dt==(a0+a1)/2-(b0+b1)/2
    v,w=compare(d0,d1),compare(d0,dt)
    # Classify by signs. The ratio is only a descriptive field.
    if d0==0: bucket="initial_tie"
    elif v=="same": bucket="r<1"
    elif v=="new_tie": bucket="r=1"
    elif w=="same": bucket="1<r<2"
    elif w=="new_tie": bucket="r=2"
    else: bucket="r>2"
    return dict(A0=a0,A1=a1,B0=b0,B1=b1,d0=d0,d1=d1,delta=delta,d_tta=dt,
                delta_A=da,delta_B=db,common_drift=(da+db)/2,differential_drift=da-db,
                pick0=pick(d0),pick1=pick(d1),pick_tta=pick(dt),change_L1=v,change_TTA=w,
                r=-delta/d0 if d0 else "",r_bin=bucket)


def paired(cells,conds):
    rows=[]
    for i,p in PAIRS.items():
        row=dict(pair_id=i,image_A=p["image_A"],image_B=p["image_B"])
        ds=[]
        for c in conds:
            m=margins(cells,i,c)
            row.update({c+"_"+k:v for k,v in m.items()})
            ds.append(m["delta"])
        row.update(I=ds[1]-ds[0],abs_I=abs(ds[1]-ds[0]),G=abs(ds[1])-abs(ds[0]))
        rows.append(row)
    return rows


def summary(name, rows, conds, bootstrap=False):
    assert rows
    v=np.array([[r[conds[0]+"_delta"],r[conds[1]+"_delta"],r["abs_I"],r["G"]] for r in rows],dtype=np.float64)
    u=np.abs(v[:,:3])
    g=v[:,3]
    q=np.quantile(g,[.25,.5,.75])
    s=dict(group=name,n=len(g),median_amp0=np.median(u[:,0]),median_amp1=np.median(u[:,1]),
           abs_I_q1=np.quantile(u[:,2],.25),abs_I_median=np.median(u[:,2]),abs_I_q3=np.quantile(u[:,2],.75),
           G_q1=q[0],G_median=q[1],G_q3=q[2],G_positive=int((g>0).sum()),G_negative=int((g<0).sum()),G_zero=int((g==0).sum()),
           bootstrap_low="",bootstrap_high="")
    if bootstrap:
        # Resample pair IDs. Each selected row carries both prompts and both brightness conditions.
        draw=np.random.default_rng(42).integers(len(g),size=(10000,len(g)))
        boot=np.median(np.abs(v[draw,1])-np.abs(v[draw,0]),axis=1)
        s["bootstrap_low"],s["bootstrap_high"]=np.quantile(boot,[.025,.975])
    for j,c in enumerate(conds):
        prefix=f"c{j}_"
        bins=Counter(r[c+"_r_bin"] for r in rows)
        assert sum(bins[b] for b in BINS)==len(rows)
        for k,b in enumerate(BINS):
            s[prefix+"bin"+str(k)]=bins[b]
        for mode,raw in (("L1","d1"),("TTA","d_tta")):
            counter=Counter(r[c+"_change_"+mode] for r in rows)
            assert sum(counter.values())==len(rows)
            for vname in ("same","initial_tie","new_tie","flip"):
                s[prefix+mode+"_"+vname]=counter[vname]
            s[prefix+mode+"_end_tie"]=sum(r[c+"_"+raw]==0 for r in rows)
    return s


def audit_old():
    assert digest(OLD/"scores.csv")==jsonfile(ROOT/"freeze.json")["old_scores_sha256"]
    scores=read_csv(OLD/"scores.csv")
    result=paired(index_scores(scores, PAIRS, ("P0","P1")),("P0","P1"))
    original={r["pair_id"]:r for r in read_csv(OLD/"pair_results.csv")}
    audit={r["pair_id"]:r for r in read_csv(ROOT/"old_prompt_audit.csv")}
    assert len(original)==len(audit)==64 and original.keys()==audit.keys()==PAIRS.keys()
    for r in result:
        i=r["pair_id"]
        old=original[i]
        assert old["P0"]==PAIRS[i]["P0"] and old["P1"]==PAIRS[i]["P1"]
        for c in ("P0","P1"):
            for field in ("d0","d1","delta","d_tta","pick0","pick1","pick_tta","change_L1","change_TTA"):
                new=r[c+"_"+field]
                value=old[c+"_"+field]
                if field.startswith("change"):
                    new="tie" if new in ("initial_tie","new_tie") else new
                assert new==(float(value) if isinstance(new,float) else value),(i,c,field)
        assert float(old["I"])==r["I"]
        r["human_P0"]=PAIRS[i]["human_winner_P0"]
        for k in ("pick0","pick_tta"):
            winner=r["P0_"+k]
            r["P0_"+k+"_human"]="tie" if winner=="tie" else "correct" if winner==r["human_P0"] else "wrong"
        info=audit[i]
        r["group"]="truncated" if info["P1_complete"]=="0" else "lit_complete" if info["light"]=="1" else "unlit_complete"
        r["identical"]=int(info["identical"])
        if r["identical"]:
            assert all(r["P0_"+k]==r["P1_"+k] for k in ("A0","A1","B0","B1")) and r["I"]==0
    assert sum(r["group"] in ("unlit_complete","lit_complete","truncated") for r in result)==64
    write_csv(ROOT/"old_audit.csv",result)
    return result


def main():
    old=audit_old()
    prompts={r["pair_id"]:r for r in frozen()}
    fresh=read_csv(ROOT/"scores_round2.csv")
    for r in fresh:
        p=prompts[r["pair_id"]]
        assert r["text_token_sha"]==p[r["prompt_condition"]+"_token_sha"]
        assert float(r["factor"])==(1.0 if r["light"]=="L0" else 1.1) and r["source"]=="round2_fresh"
    new=paired(index_scores(fresh,PAIRS,("Q0","Q1")),("Q0","Q1"))
    for r in new:
        r["semantic_conflict"]=int(prompts[r["pair_id"]]["semantic_conflict"])
    write_csv(ROOT/"pair_results_round2.csv",new)
    sums=[summary("old_all",old,("P0","P1"))]
    for g in ("unlit_complete","lit_complete","truncated"):
        sums.append(summary(g,[r for r in old if r["group"]==g],("P0","P1")))
    sums.append(summary("identical",[r for r in old if r["identical"]],("P0","P1")))
    for name,rows in [("new_all",new),("no_conflict",[r for r in new if not r["semantic_conflict"]])]:
        sums.append(summary(name,rows,("Q0","Q1"),True))
    write_csv(ROOT/"summary.csv",sums)
    print([(s["group"],s["n"],s["G_median"],s["bootstrap_low"],s["bootstrap_high"]) for s in sums])

if __name__=="__main__":
    main()

