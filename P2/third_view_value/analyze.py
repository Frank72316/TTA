"""Exact finite parameter enumeration; every resampling unit is a prompt."""
import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common_io import HERE, ROOT, SEED, VIEWS, read, save, text_hash, write

B = VIEWS[1:5]
C = VIEWS[5:9]
BC = list(itertools.product(C,B))
CC = list(itertools.permutations(C,2))
BB = list(itertools.combinations(B,2))
RECIPES = {"O":[("O",)],"OC":[("O",c) for c in C],"OCB":[("O",*p) for p in BC],
           "OCC":[("O",*p) for p in CC],"OBB":[("O",*p) for p in BB]}
STATE = {-1:"wrong",0:"tie",1:"correct"}

def bounds(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if np.all(x==0):
        return [0., -100*(-math.expm1(math.log(.025)/n)), 100*(-math.expm1(math.log(.025)/n)),
                100*(-math.expm1(math.log(.05)/n)),100*(-math.expm1(math.log(.001)/n))]
    rng = np.random.default_rng(SEED)
    draws = np.concatenate([x[rng.integers(n,size=(250,n))].mean(axis=1) for _ in range(40)])
    return [100*x.mean(), *list(100*np.quantile(draws,[.025,.975,.95,.999]))]

def binom_upper(k,n,alpha):
    if k==n:
        return 1.
    if k==0:
        return -math.expm1(math.log(alpha)/n)
    # Invert Pr_p(X<=k) = alpha; log space keeps either endpoint stable.
    from math import lgamma, log, log1p, exp
    log_choose = [lgamma(n+1)-lgamma(j+1)-lgamma(n-j+1) for j in range(k+1)]
    lo,hi = 0.,1.
    for _ in range(58):
        p = (lo+hi)/2
        cdf = sum(exp(v+j*log(p)+(n-j)*log1p(-p)) for j,v in enumerate(log_choose))
        if cdf>alpha:
            lo=p
        else:
            hi=p
    return hi

def load_scores(pairs, path, dev):
    by_id = {p["pair_id"]:p for p in pairs}
    lookup = defaultdict(dict)
    for row in read(path):
        if row["view"] not in VIEWS and dev:
            continue
        k = row["pair_id"]
        p = by_id[k]
        side,v = row["side"],row["view"]
        assert side in ("A","B") and v in VIEWS and (side,v) not in lookup[k]
        assert row["image_id"]==p["image_"+side] and row["prompt_id"]==p["prompt_id"]
        if "P0" in row:
            assert row["P0"]==p["P0"]
        assert v=="O" or float(row["param"])==int(v[1:])/100
        val = float(row["score"])
        assert row["native_dtype"]=="float16" and np.isfinite(val) and float(np.float16(val))==val
        lookup[k][side,v] = val
    assert set(lookup)==set(by_id) and all(len(a)==18 for a in lookup.values())
    return lookup

def paired_cells(before,after):
    cells = Counter(zip(before,after))
    return {f"{STATE[s]}_to_{STATE[t]}":cells[s,t] for s,t in itertools.product(STATE,repeat=2)}

def analyze(pairs, scores, dev=False):
    raw = load_scores(pairs,scores,dev)
    details, trans, threshold = [],[],[]
    for p in pairs:
        pid = p["pair_id"]
        winner = p["human_winner_P0"]
        assert int(p["rank_A"])!=int(p["rank_B"]) and (winner=="A")== (int(p["rank_A"])<int(p["rank_B"]))
        other = "B" if winner=="A" else "A"
        values = raw[pid]
        m = {v:values[winner,v]-values[other,v] for v in VIEWS}
        state = {}
        d = dict(pair_id=pid,prompt_id=p["prompt_id"],winner=winner,m_O=m["O"],original_state=STATE[np.sign(m["O"])])
        for name,recipes in RECIPES.items():
            vals = [sum(m[v] for v in r) for r in recipes]
            # Independent check using separate per-candidate means then comparing A/B.
            decisions = [np.sign(sum(values[winner,v] for v in r)/len(r)-sum(values[other,v] for v in r)/len(r)) for r in recipes]
            assert np.array_equal(np.sign(vals),decisions)
            state[name] = np.sign(vals).astype(int)
            for s,label in STATE.items():
                d[name+"_"+label] = np.mean(state[name]==s)
        for strategy,configs in [("OCB",BC),("OCC",CC)]:
            stopped = [state["OC"][C.index(c)] for c,z in configs]
            cell = paired_cells(stopped,state[strategy])
            trans.append(dict(pair_id=pid,original=d["original_state"],comparison="OC_to_"+strategy,recipes=len(configs),**cell))
            d[strategy+"_strict_rescue"] = cell["wrong_to_correct"]
            d[strategy+"_strict_harm"] = cell["correct_to_wrong"]
            for (c,z),src,dst in zip(configs,stopped,state[strategy]):
                before,extra = m["O"]+m[c],m[z]
                crossed = before*extra<0 and abs(extra)>abs(before)
                assert crossed == (int(src)*int(dst)==-1)
                threshold.append(dict(pair_id=pid,original=d["original_state"],strategy=strategy,c1=c,third=z,
                                      m_O=m["O"],m_C1=m[c],m_third=extra,stop_sum=before,three_sum=before+extra,
                                      stop=STATE[src],final=STATE[dst],strict_cross=int(crossed)))
        coupled = [(state["OCB"][i],state["OCC"][j]) for i,(c,b) in enumerate(BC)
                   for j,(cc,c2) in enumerate(CC) if c==cc]
        assert len(coupled)==48
        trans.append(dict(pair_id=pid,original=d["original_state"],comparison="OCB_to_OCC",recipes=48,
                          **paired_cells(*zip(*coupled))))
        d["L"] = d["OCB_correct"]-d["OC_correct"]
        details.append(d)
    return details,trans,threshold

def compute(pairs,score_path,dev):
    out,trans,combos = analyze(pairs,score_path,dev)
    n = len(out)
    label = "dev_" if dev else ""
    def emit(name,rows):
        write(HERE/(label+name+".csv"),rows)
    baseline = Counter(r["original_state"] for r in out)
    summary = []
    for s in RECIPES:
        row = dict(strategy=s,recipe_count=len(RECIPES[s]),n=n)
        for t in STATE.values():
            row[t] = sum(r[s+"_"+t] for r in out)
            row[t+"_pct"] = 100*row[t]/n
        for t0 in STATE.values():
            for t1 in STATE.values():
                row[t0+"_to_"+t1] = sum(r[s+"_"+t1] for r in out if r["original_state"]==t0)
        assert np.isclose(sum(row[t] for t in STATE.values()),n)
        summary.append(row)
    intervals = []
    def ci(name,values,threshold_pp=None):
        est,lo,hi,u,adj=bounds(values)
        intervals.append(dict(comparison=name,estimate_pp=est,low95_pp=lo,high95_pp=hi,upper95_pp=u,
                              upper_family_pp=adj if threshold_pp is not None else "",
                              threshold_pp=threshold_pp if threshold_pp is not None else "",
                              upper_method="exact_zero_events" if all(v==0 for v in values) else "paired_percentile"))
    ci("primary_OCB_minus_OC",[r["L"] for r in out])
    for src in ("OBB","OCB"):
        ci("accuracy_loss_"+src+"_to_OCC",[r[src+"_correct"]-r["OCC_correct"] for r in out],.5)
        ci("extra_harm_"+src+"_to_OCC",[(r["OCC_wrong"]-r[src+"_wrong"]) if r["original_state"]=="correct" else 0 for r in out],.5)
    wrong = [r for r in out if r["original_state"]=="wrong"]
    missed = [r["pair_id"] for r in wrong if r["OBB_correct"]>0 and r["OCC_correct"]==0]
    k,w = len(missed),len(wrong)
    intervals.append(dict(comparison="BB_only_among_original_wrong",estimate_pp=100*k/w,low95_pp="",high95_pp="",
                          upper95_pp=100*binom_upper(k,w,.05),upper_family_pp=100*binom_upper(k,w,.001),
                          threshold_pp=3,upper_method="Clopper-Pearson"))
    if dev:
        assert n==256 and baseline["correct"]==175 and baseline["wrong"]==81
        cb = {s:{r["pair_id"] for r in wrong if r[s+"_correct"]>0} for s in ["OCC","OBB"]}
        assert len(cb["OBB"])==5 and cb["OCC"]-cb["OBB"]=={"N020","N164"} and cb["OBB"]<=cb["OCC"]
        sd=np.std([r["L"] for r in out],ddof=1)
        save(HERE/"dev_precision.json",dict(n=n,sd_per_prompt=float(sd),approx_halfwidth_at_400_pp=100*1.96*sd/20,
                                            n_for_halfwidth_0_5pp=math.ceil((1.96*sd/.005)**2)))
    emit("strategy_summary",summary)
    emit("intervals",intervals)
    if dev:
        return
    emit("prompt_results",out)
    emit("recipe_margins",combos)
    emit("prompt_transitions",trans)
    totals=[]
    for name in ("OC_to_OCB","OC_to_OCC","OCB_to_OCC"):
        for grp in ("all","wrong","correct","tie"):
            subset=[r for r in trans if r["comparison"]==name and (grp=="all" or r["original"]==grp)]
            cells={f"{s}_to_{t}":sum(r[f"{s}_to_{t}"]/r["recipes"] for r in subset)
                   for s,t in itertools.product(STATE.values(),repeat=2)}
            assert np.isclose(sum(cells.values()),len(subset))
            totals.append(dict(comparison=name,original_group=grp,n=len(subset),**cells))
    emit("transition_summary",totals)
    # Compare either third action under exactly the same stopped decision.
    conditional = []
    for pid, group in itertools.groupby(combos,key=lambda r:r["pair_id"]):
        rows = list(group)
        for c in C:
            b = [r for r in rows if r["c1"]==c and r["strategy"]=="OCB"]
            cc = [r for r in rows if r["c1"]==c and r["strategy"]=="OCC"]
            assert len(b)==4 and len(cc)==3
            paired = list(itertools.product([r["final"] for r in b],[r["final"] for r in cc]))
            cell = Counter(paired)
            conditional.append(dict(pair_id=pid,c1=c,original=rows[0]["original"],stop=b[0]["stop"],
                              B_correct_of_4=sum(r["final"]=="correct" for r in b),
                              C_correct_of_3=sum(r["final"]=="correct" for r in cc),
                              **{f"{x}_to_{y}_of_12":cell[x,y] for x,y in itertools.product(STATE.values(),repeat=2)}))
    emit("fixed_c1_actions",conditional)
    coverage=[]
    for r in wrong:
        reachable=[r[s+"_correct"]>0 for s in ("OCB","OCC")]
        rescued=[r[s+"_strict_rescue"]>0 for s in ("OCB","OCC")]
        group=lambda z: ["neither","contrast_only","brightness_only","both"][2*int(z[0])+int(z[1])]
        coverage.append(dict(pair_id=r["pair_id"],any_correct=group(reachable),stop_wrong_rescue=group(rescued),
                             OCB_correct_of_16=round(16*r["OCB_correct"]),OCC_correct_of_12=round(12*r["OCC_correct"]),
                             OCB_rescue_of_16=r["OCB_strict_rescue"],OCC_rescue_of_12=r["OCC_strict_rescue"],
                             OBB_correct_of_6=round(6*r["OBB_correct"])))
    emit("original_errors",coverage)
    only={r["pair_id"] for r in coverage if r["stop_wrong_rescue"]=="brightness_only"}
    cases=[]
    for kind in ["brightness_only_rescue","brightness_introduces_error"]:
        for r in combos:
            if r["strategy"]!="OCB":
                continue
            if ((kind=="brightness_only_rescue" and r["pair_id"] in only and r["stop"]=="wrong" and r["final"]=="correct")
                or (kind=="brightness_introduces_error" and r["original"]=="correct" and r["stop"]=="correct" and r["final"]=="wrong")):
                cases.append(dict(case=kind,**r))
                break
    save(HERE/"cases.json",cases)
    save(HERE/"audit.json",dict(pairs=n,unique_scores=n*18,unique_images=2*n,expected_recipes={s:len(r) for s,r in RECIPES.items()},
                                aggregate_sign_disagreements=0,crossing_formula_checks=len(combos),
                                shared_c1_replacement_draws=n*48,baseline=dict(baseline),pure_B_missed_by_C=missed,
                                reachable_original_errors=dict(Counter(r["any_correct"] for r in coverage)),
                                strict_rescue_original_errors=dict(Counter(r["stop_wrong_rescue"] for r in coverage))))
    print(intervals,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dev",action="store_true")
    a=ap.parse_args()
    base=ROOT/"P2/translation" if a.dev else HERE
    compute(read(base/"pairs_frozen.csv"),base/"scores.csv",a.dev)
