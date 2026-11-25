"""Pair-level three-score decisions. Run after the complete score cache is exported."""
from collections import Counter,defaultdict
import numpy as np
from common import *

def median(values):
    return float(np.median(values)) if len(values) else ""

def main():
    pairs=frozen(); n=len(pairs); pmap={p["pair_id"]:p for p in pairs}
    scores=read_csv(ROOT/"scores.csv"); bank={}
    for r in scores:
        i,s,v=r["pair_id"],r["side"],r["view"];p=pmap[i]
        value=float(r["score"]);key=i,s,v
        assert key not in bank and np.isfinite(value) and r["P0"]==p["P0"] and r["image_id"]==p["image_"+s] and r["prompt_id"]==p["prompt_id"]
        assert r["native_dtype"]=="float16" and value==float(np.float16(value))
        bank[key]=value
    assert len(bank)==len(scores)==26*n
    combos=[]; results=[]; shifts=[]; contributions=[]; directions=[]
    for p in pairs:
        i=p["pair_id"]; human=p["human_winner_P0"]
        d={v:bank[i,"A",v]-bank[i,"B",v] for v in VIEWS}; d0=d["O"]
        pick0=choice(d0);base=outcome(pick0,human)
        delta={v:x-d0 for v,x in d.items()};counts={}
        for v in VIEWS[1:]:
            shifts.append(dict(pair_id=i,view=v,family=v[0],d0=d0,view_margin=d[v],delta=delta[v],abs_delta=abs(delta[v])))
        for st in STRATS:
            count=Counter();size=1 if st=="O" else len(combinations(st))
            for x,y in ([("","")] if st=="O" else combinations(st)):
                vv=["O",x,y] if x else ["O"]
                tot=sum(bank[i,"A",v] for v in vv)-sum(bank[i,"B",v] for v in vv)
                if x:assert tot==3*d0+delta[x]+delta[y]
                z=tot/len(vv);w=choice(tot);res=outcome(w,human)
                changed=w!=pick0;flip=tot*d0<0
                count[res]+=1;count["change"]+=int(changed);count["flip"]+=int(flip)
                if st!="O":combos.append(dict(pair_id=i,strategy=st,view1=x,view2=y,original_pick=pick0,human=human,d0=d0,margin=z,pick=w,state=res,changed=int(changed),strict_flip=int(flip)))
            row=dict(pair_id=i,human=human,d0=d0,abs_d0=abs(d0),original_pick=pick0,original_state=base,strategy=st,n_combinations=size,
                     **{s:count[s] for s in STATES},changed=count["change"],strict_flips=count["flip"],correct_fraction=count["correct"]/size)
            assert sum(count[s] for s in STATES)==size
            results.append(row);counts[st]=row
        gain=counts["BT"]["correct_fraction"]-counts["BC"]["correct_fraction"]
        pc=dict(pair_id=i,human=human,d0=d0,abs_d0=abs(d0),original_pick=pick0,original_state=base,bt_minus_bc_expected_correct=gain,
                contribution_pp=100*gain/n)
        for st in STRATS[1:]:
            for tag in STATES+["changed","strict_flips"]:pc[st+"_"+tag]=counts[st][tag]
        contributions.append(pc)
        if counts["BT"]["changed"]:
            for t in FAMILIES["T"]:
                dv=[r for r in combos[-16:] if r["view2"]==t]
                assert len(dv)==4 and all(r["pair_id"]==i and r["strategy"]=="BT" for r in dv)
                z=(d0+d[t])/2
                directions.append(dict(pair_id=i,original_state=base,direction=t,d0=d0,delta_t=delta[t],O_plus_T_margin=z,O_plus_T_pick=choice(z),
                      O_plus_T_changes=int(choice(z)!=pick0),n_changed=sum(r["changed"] for r in dv),n_strict_flips=sum(r["strict_flip"] for r in dv),
                      changed_brightness=" ".join(r["view1"] for r in dv if r["changed"]),**{s:sum(r["state"]==s for r in dv) for s in STATES}))
    by_state=[]; summaries=[]
    for st in STRATS:
        rr=[r for r in results if r["strategy"]==st]; k=rr[0]["n_combinations"]
        for state in STATES:
            sel=[r for r in rr if r["original_state"]==state]
            row=dict(strategy=st,original_state=state,pairs=len(sel),combinations_per_pair=k)
            for s in STATES:
                total=sum(r[s] for r in sel);row[s+"_cells"]=total;row[s+"_expected"]=total/k
            by_state.append(row)
            assert sum(row[s+"_cells"] for s in STATES)==k*len(sel)
        row=dict(strategy=st,pairs=n,total_cells=k*n,changed_pairs=sum(r["changed"]>0 for r in rr),strict_flip_pairs=sum(r["strict_flips"]>0 for r in rr))
        for s in STATES:
            total=sum(r[s] for r in rr);row[s+"_cells"]=total;row[s+"_expected"]=total/k;row[s+"_percent"]=100*total/(k*n)
        summaries.append(row)
    nz=sorted((r for r in contributions if r["d0"]!=0),key=lambda r:(r["abs_d0"],r["pair_id"]))
    groups={f"Q{j}":[nz[int(k)] for k in inds] for j,inds in enumerate(np.array_split(np.arange(len(nz)),4),1)}
    groups["original_tie"]=[r for r in contributions if not r["d0"]]
    shiftmap={(r["pair_id"],r["view"]):r["abs_delta"] for r in shifts}
    bins=[]
    for name,sub in groups.items():
        for r in sub:r["margin_group"]=name
        lead=[r["abs_d0"] for r in sub]
        rec=dict(group=name,n=len(sub),min_abs_d0=min(lead,default=""),median_abs_d0=median(lead),max_abs_d0=max(lead,default=""))
        med={}
        for fam in ("B","C","T"):
            vals=[[shiftmap[r["pair_id"],v] for v in FAMILIES[fam]] for r in sub];med[fam]=np.array([median(a) for a in vals])
            rec["median_abs_delta_"+fam]=median(np.ravel(vals))
        rec.update(T_gt_C=int(np.sum(med["T"]>med["C"])),T_eq_C=int(np.sum(med["T"]==med["C"])),T_lt_C=int(np.sum(med["T"]<med["C"])),
                   any_BT_BC_change=sum(bool(r["BT_changed"] or r["BC_changed"]) for r in sub))
        for st in ["BT","BC"]:
            rec[st+"_changed_pairs"]=sum(r[st+"_changed"]>0 for r in sub)
            rec[st+"_strict_flip_pairs"]=sum(r[st+"_strict_flips"]>0 for r in sub)
        rec["BT_minus_BC_pp"]=100*sum(r["bt_minus_bc_expected_correct"] for r in sub)/len(sub) if sub else ""
        bins.append(rec)
    for name,rows in [("combination_results",combos),("pair_strategy_results",results),("single_view_shifts",shifts),("pair_contributions",contributions),
                      ("strategy_summary",summaries),("transitions",by_state),("margin_groups",bins),("direction_counts",directions)]:
        if rows:write_csv(ROOT/(name+".csv"),rows)
        elif name=="direction_counts":
            (ROOT/(name+".csv")).write_text("pair_id,original_state,direction,d0,delta_t,O_plus_T_margin,O_plus_T_pick,O_plus_T_changes,n_changed,n_strict_flips,changed_brightness,correct,wrong,tie\n")
    diff=np.array([r["bt_minus_bc_expected_correct"] for r in contributions])
    indices=np.random.default_rng(2026).integers(0,n,(10000,n))
    def ci(v):return np.percentile(v,[2.5,97.5]).tolist()
    by_base={}
    for base in STATES:
        mask=np.array([r["original_state"]==base for r in contributions])
        for s in STATES:
            v=np.array([(r["BT_"+s]-r["BC_"+s])/16 for r in contributions])*mask
            by_base[base+"_to_"+s]=dict(expected_pairs=float(v.sum()),ci_expected_pairs=ci(v[indices].sum(axis=1)))
    assert sum(by_base[st+"_to_correct"]["expected_pairs"] for st in STATES)==sum(diff)
    write_json(ROOT/"endpoint.json",dict(n=n,bt_minus_bc_pp=float(100*diff.mean()),ci_pp=ci(100*diff[indices].mean(axis=1)),net_expected_correct=float(diff.sum()),
              ci_expected_correct=ci(diff[indices].sum(axis=1)),source_contrasts=by_base,combinations=len(combos),raw_scores=len(scores),
              bootstrap_seed=2026,bootstrap_resamples=10000,identity_checks=len(combos)))
    print(n,"pairs;",len(combos),"identities;",len(directions)//4,"BT changed pairs;",100*diff.mean(),"pp")

if __name__=="__main__":main()

