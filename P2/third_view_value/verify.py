"""Reconcile trial-level, prompt-level and timed decisions independently of analysis."""
import json
from collections import Counter,defaultdict
from itertools import product
import numpy as np
from common_io import HERE, VIEWS, read, save, text_hash

def main():
    pairs={r["pair_id"]:r for r in read(HERE/"pairs_frozen.csv")}
    n=len(pairs)
    freeze=json.loads((HERE/"freeze.json").read_text())
    assert n==freeze["samples"] and text_hash(HERE/"pairs_frozen.csv")==freeze["pairs_sha256_lf"]
    assert text_hash(HERE/"protocol.md")==freeze["protocol_sha256_lf"]
    env=json.loads((HERE/"environment.json").read_text())
    assert text_hash(HERE/"run_gpu.py")==env["script_sha256_lf"]
    assert text_hash(HERE/"common_io.py")==env["common_sha256_lf"]
    scores={(r["pair_id"],r["side"],r["view"]):r for r in read(HERE/"scores.csv")}
    assert len(read(HERE/"scores.csv"))==len(scores)==n*18
    assert all((p,s,v) in scores for p,s,v in product(pairs,"AB",VIEWS))
    for r in read(HERE/"smoke_scores.csv"):
        assert r==scores[r["pair_id"],r["side"],r["view"]]
    for pid,p in pairs.items():
        assert int(p["rank_A"])!=int(p["rank_B"])
        for s in "AB":
            group=[scores[pid,s,v] for v in VIEWS]
            assert all(r["prompt_id"]==p["prompt_id"] and r["image_id"]==p["image_"+s] for r in group)
            assert len({r["source_sha256"] for r in group})==1
        assert len({scores[pid,s,v]["token_sha256"] for s,v in product("AB",VIEWS)})==1
    info={r["pair_id"]:r for r in read(HERE/"prompt_results.csv")}
    sums={r["strategy"]:r for r in read(HERE/"strategy_summary.csv")}
    for r in read(HERE/"transition_summary.csv"):
        if r["original_group"]!="all":
            continue
        a,b=r["comparison"].split("_to_")
        for i in ("correct","wrong","tie"):
            assert np.isclose(sum(float(r[f"{i}_to_{j}"]) for j in ("correct","wrong","tie")),float(sums[a][i]))
            assert np.isclose(sum(float(r[f"{j}_to_{i}"]) for j in ("correct","wrong","tie")),float(sums[b][i]))
    timings=read(HERE/"timings.csv")
    assert len(timings)==64*3
    ms=defaultdict(dict)
    for r in timings:
        p=r["pair_id"]
        k=int(r["k"])
        recipe=["O",r["c1"],r["b1"]][:k]
        a,b=[sum(float(scores[p,s,v]["score"]) for v in recipe) for s in "AB"]
        assert r["winner"]==("A" if a>b else "B" if a<b else "tie")
        assert p not in ms[k]
        ms[k][p]=float(r["ms"])
    seq=list(pairs)[:64]
    assert all(set(ms[k])==set(seq) for k in (1,2,3))
    z={str(k):dict(median_ms=float(np.median(list(ms[k].values()))),
                   p95_ms=float(np.percentile(list(ms[k].values()),95))) for k in (1,2,3)}
    diff=np.asarray([ms[3][p]-ms[2][p] for p in seq])
    z["paired_3_minus_2"]=dict(median_ms=float(np.median(diff)),p95_ms=float(np.percentile(diff,95)),
                                faster_pairs=int(sum(diff>0)),median_relative_saved_pct=float(100*np.median(diff/[ms[3][p] for p in seq])))
    save(HERE/"latency_summary.json",z)
    trials=read(HERE/"fixed_c1_actions.csv")
    lifts=[]
    for name,rules in [
        ("stopped_wrong_B_repairs_C_strict_wrong",("wrong","correct_to_wrong_of_12")),
        ("stopped_wrong_B_repairs_C_ties",("wrong","correct_to_tie_of_12")),
        ("stopped_correct_B_breaks_C_keeps",("correct","wrong_to_correct_of_12")),
        ("stopped_correct_B_breaks_C_ties",("correct","wrong_to_tie_of_12"))]:
        stop,col=rules
        by=Counter()
        for r in trials:
            if r["stop"]==stop and int(r[col]):
                by[r["pair_id"]]+=int(r[col])
        lifts.append(dict(event=name,affected_ids=list(by),count_by_pair_out_of_48=dict(by),expected_pairs=sum(by.values())/48))
    save(HERE/"replacement_cases.json",lifts)
    runs=json.loads((HERE/"runs.json").read_text())
    assert any(r["pairs"]==n for r in runs)
    assert all(r["new_scores"]+r["cache_hits"]==18*r["pairs"] for r in runs)
    ordered_keys=set(info)
    assert ordered_keys==set(pairs) and len(trials)==n*4
    print(f"Verified {n} prompts, {n*18} unique records, all timing decisions and all transition marginals.")

if __name__=="__main__":
    main()
