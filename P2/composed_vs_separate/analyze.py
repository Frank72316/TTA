"""Offline paired decisions on the native HPS outputs."""
from collections import Counter
import json
import numpy as np
from study import HERE, COMBOS, VIEWS, read, write


def outcome(m):
    return "correct" if m>0 else "wrong" if m<0 else "tie"


def main():
    rows=read(HERE/"scores.csv")
    frozen=read(HERE.parent/"translation/pairs_frozen.csv")
    bank={(r["pair_id"],r["side"],r["view"]):r for r in rows}
    assert len(rows)==len(bank)==7680 and sum("+" in r["view"] for r in rows)==4096
    strategies=("original","original_B","original_T","separate","composed")
    details=[]; per_pair=[]; crossings=Counter(); examples={}; rates=Counter()
    for pair in frozen:
        pid=pair["pair_id"]; human=pair["human_winner_P0"]
        assert human==("A" if int(pair["rank_A"])<int(pair["rank_B"]) else "B")
        signed=1 if human=="A" else -1
        margin={}; step={}
        for v in VIEWS:
            scores=[]
            for side in "AB":
                r=bank[pid,side,v]
                assert (r["human"],r["P0"],r["prompt_id"],r["image_id"])==(human,pair["P0"],pair["prompt_id"],pair["image_"+side])
                value=float(r["score"])
                assert np.isfinite(value) and float(np.float16(value))==value
                scores.append(value)
            margin[v]=signed*(scores[0]-scores[1])
            step[v]=max(float(np.spacing(np.float16(s))) for s in scores)
        m0=margin["O"]; counts=Counter()
        for b,t in COMBOS:
            bt=b+"+"+t; mb,mt,mbt=(margin[v] for v in (b,t,bt))
            margins=(m0,(m0+mb)/2,(m0+mt)/2,(mb+mt)/2,(m0+mbt)/2)
            I=mbt-mt-mb+m0
            assert margins[-1]-margins[-2]==I/2
            ulp=max(step[v] for v in ("O",b,t,bt))
            prev,nxt=map(outcome,margins[-2:])
            crossings[prev,nxt]+=1
            if (prev,nxt) in (("wrong","correct"),("correct","wrong")):
                examples.setdefault(prev+"_to_"+nxt,dict(pair_id=pid,human=human,P0=pair["P0"],B=b,T=t,
                    m_O=m0,m_B=mb,m_T=mt,m_BT=mbt,m_separate=margins[-2],m_composed=margins[-1]))
            entry=dict(pair_id=pid,human=human,B=b,T=t,m_O=m0,m_B=mb,m_T=mt,m_BT=mbt,I=I,
                abs_I_over_abs_mO=abs(I/m0) if m0 else "",max_score_ulp=ulp,within_one_ulp=int(abs(I)<=ulp))
            for strategy,m in zip(strategies,margins):
                result=outcome(m)
                counts[strategy,result]+=1; rates[strategy,result]+=1
                entry["m_"+strategy]=m; entry[strategy]=result
            details.append(entry)
        per_pair.append(dict(pair_id=pid,human=human,m_O=m0,**{s+"_p_"+k:counts[s,k]/8
                    for s in strategies for k in ("correct","wrong","tie")}))
    write(HERE/"combinations.csv",details); write(HERE/"pair_results.csv",per_pair)
    case_rows=[dict(event=key,**r) for key,r in examples.items()]
    if case_rows: write(HERE/"cases.csv",case_rows)
    diff=np.array([r["composed_p_correct"]-r["separate_p_correct"] for r in per_pair])
    assert len(diff)==256 and rates["original","correct"]==175*8
    draws=np.random.default_rng(42).integers(256,size=(10000,256))
    ci=np.quantile(100*diff[draws].mean(axis=1),[.025,.975])
    q=lambda x:np.quantile(x,[.25,.5,.75]).tolist()
    changed=[r for r in details if r["separate"]!=r["composed"]]
    strict=[r for r in changed if "tie" not in (r["separate"],r["composed"])]
    stats=dict(records=len(rows),pairs=len(diff),contrasts=len(details),diff_pp=100*diff.mean(),ci_pp=ci.tolist(),
        transitions={x:{y:crossings[x,y] for y in ("correct","wrong","tie")} for x in ("correct","wrong","tie")},
        expected={s:{k:rates[s,k]/8 for k in ("correct","wrong","tie")} for s in strategies},
        abs_I_quartiles=q([abs(r["I"]) for r in details]),
        abs_I_ratio_quartiles=q([r["abs_I_over_abs_mO"] for r in details if r["m_O"]]),
        zero_m0_pairs=sum(r["m_O"]==0 for r in per_pair),zero_I=sum(r["I"]==0 for r in details),
        within_one_native_ulp=sum(r["within_one_ulp"] for r in details),
        ulp_values=sorted(set(r["max_score_ulp"] for r in details)),
        changed=len(changed),changed_pairs=len({r["pair_id"] for r in changed}),
        strict_flips=len(strict),strict_flip_pairs=len({r["pair_id"] for r in strict}),
        max_identity_error=max(abs(r["m_composed"]-r["m_separate"]-r["I"]/2) for r in details),
        source_counts=dict(Counter(r["source"] for r in rows)))
    (HERE/"summary.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
