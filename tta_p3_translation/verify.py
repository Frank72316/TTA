"""Independent checks using exact integer score units and the source ranking metadata."""
from collections import Counter,defaultdict
from html.parser import HTMLParser
from itertools import combinations as unordered_pairs,product
import struct
import numpy as np
from common import *

def validate():
    ps=frozen();n=len(ps);pmap={p["pair_id"]:p for p in ps};score=read_csv(ROOT/"scores.csv")
    sf=2**24
    def integer(r):
        v=float(r["score"]);unit=round(v*sf);assert v==unit/sf
        return unit
    s={(r["pair_id"],r["side"],r["view"]):integer(r) for r in score}
    assert len(s)==len(score)==26*n
    for p in ps:
        cached=file_json(ROOT/"cache"/(p["pair_id"]+".json"))
        assert cached["protocol"]==digest(ROOT/"environment.json")
        assert len(cached["rows"])==26
        for r in cached["rows"]:
            assert s[p["pair_id"],r["side"],r["view"]]==integer(r)
            assert r["image_id"]==p["image_"+r["side"]] and r["P0"]==p["P0"] and r["prompt_id"]==p["prompt_id"]
    pools={f:[v for v in VIEWS if v.startswith(f)] for f in "BCT"}
    counts=defaultdict(Counter);comb=read_csv(ROOT/"combination_results.csv");comb_key=set();shift=read_csv(ROOT/"single_view_shifts.csv")
    for r in comb:
        i,st,x,y=[r[k] for k in ("pair_id","strategy","view1","view2")];key=i,st,x,y
        assert key not in comb_key;comb_key.add(key)
        marg=lambda v:s[i,"A",v]-s[i,"B",v]
        d=marg("O");num=d+marg(x)+marg(y);pick=choice(num);state=outcome(pick,pmap[i]["human_winner_P0"])
        assert r["pick"]==pick and r["state"]==state and float(r["margin"])==num/(3*sf)
        assert float(r["d0"])==d/sf and int(r["changed"])==(pick!=choice(d)) and int(r["strict_flip"])==(num*d<0)
        counts[i,st][state]+=1
    expected={(i,st,x,y) for i in pmap for st in STRATS[1:] for x,y in (unordered_pairs(pools[st[0]],2) if st[0]==st[1] else product(pools[st[0]],pools[st[1]]))}
    assert expected==comb_key
    assert len(shift)==12*n==len({(r["pair_id"],r["view"]) for r in shift})
    for r in shift:
        i,v=r["pair_id"],r["view"]
        assert float(r["delta"])==(s[i,"A",v]-s[i,"B",v]-s[i,"A","O"]+s[i,"B","O"])/sf
    rr=read_csv(ROOT/"pair_strategy_results.csv");assert len(rr)==6*n==len({(r["pair_id"],r["strategy"]) for r in rr})
    for r in rr:
        i,st=r["pair_id"],r["strategy"]
        if st=="O":counts[i,st][outcome(choice(s[i,"A","O"]-s[i,"B","O"]),pmap[i]["human_winner_P0"])]=1
        assert all(counts[i,st][k]==int(r[k]) for k in STATES)
    b=np.array([counts[i,"BT"]["correct"]-counts[i,"BC"]["correct"] for i in pmap])
    e=file_json(ROOT/"endpoint.json");assert sum(b)*100/(16*n)==e["bt_minus_bc_pp"]
    rng=np.random.default_rng(2026)
    ci=np.percentile([b[rng.choice(n,n,replace=True)].sum()*100/(16*n) for _ in range(10000)],[2.5,97.5])
    assert np.array_equal(ci,e["ci_pp"])
    bins=read_csv(ROOT/"margin_groups.csv");assert sum(int(r["n"]) for r in bins)==n
    direc=read_csv(ROOT/"direction_counts.csv")
    assert sum(int(r["n_changed"]) for r in direc)==sum(int(r["changed"]) for r in rr if r["strategy"]=="BT")
    totals=read_csv(ROOT/"strategy_summary.csv");trans=read_csv(ROOT/"transitions.csv")
    for a in totals:
        trs=[r for r in trans if r["strategy"]==a["strategy"]]; assert sum(int(r["pairs"]) for r in trs)==n
        for k in STATES:assert int(a[k+"_cells"])==sum(int(r[k+"_cells"]) for r in trs)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.img=[];self.href=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=="img":self.img.append(a["src"])
            if tag=="a" and a.get("href","").startswith("assets/"):self.href.append(a["href"])
    reader=Links();folder=ROOT/"old_cases";reader.feed((folder/"index.html").read_text(encoding="utf8"))
    assert len(set(reader.img))==50 and len(set(reader.href))==10
    from PIL import Image
    for im in reader.img+reader.href:
        with Image.open(folder/im) as pic:
            if im in reader.img:assert pic.size==(224,224)
            pic.verify()
    for fig in ("strategy_transitions","margin_groups"):
        with Image.open(ROOT/"figures"/(fig+".png")) as pic:pic.verify()
    assert len(list((ROOT/"figures").glob("*.png")))==2
    runs=file_json(ROOT/"runs.json")
    assert sum(r["unique_new_scores"] for r in runs)==len(s) and runs[0]["limit"]==2 and runs[0]["reference_check_extra_forwards"]==52
    old_paths=(ROOT/"logs/prior_inputs.sha256").read_text().splitlines()
    for line in old_paths:
        sha,path=line.split(maxsplit=1);assert digest(ROOT/path)==sha
    info=dict(independent_pairs=n,unique_images=2*n,raw_scores=len(s),unique_strategy_combinations=len(comb_key),score_checkpoints=n,
              exact_integer_decisions=True,bootstrap_reproduced=True,transitions_reconcile=True,old_input_hashes_unchanged=len(old_paths),
              original_case_views_verified=len(reader.img),original_webps_verified=len(reader.href),statistical_figures=2)
    write_json(ROOT/"verification.json",info);print(info)
if __name__=="__main__":validate()

