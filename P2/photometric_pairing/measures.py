"""CPU statistics on matched pairs; each formulation produces one strict decision."""
from collections import Counter
import csv,hashlib,json
import numpy as np
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,rows[0],lineterminator="\n");w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,data):p.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf8")
def result(v):return "correct" if v>0 else "wrong" if v<0 else "tie"
def quant(x):return np.quantile(x,[0,.25,.5,.75,1]).tolist()
def bootstrap(x):
    x=np.asarray(x,float); idx=np.random.default_rng(42).integers(len(x),size=(10000,len(x)))
    return [x.mean(),*np.quantile(x[idx].mean(1),[.025,.975])]
def table_rates(a,b):
    d={}
    for m,n in zip(a,b):
        for x,y in zip(m,n):
            k=result(x)+"_to_"+result(y);d[k]=d.get(k,0)+1/len(m)
    return {i+"_to_"+j:d.get(i+"_to_"+j,0) for i in ("correct","wrong","tie") for j in ("correct","wrong","tie")}
def cases(ids,baseline,intervention):
    picked={}
    for pid,a,b in zip(ids,baseline,intervention):
        for j,(m,n) in enumerate(zip(a,b)):
            if m<0<n or n<0<m:
                key="correction" if n>0 else "loss"
                picked.setdefault(key,dict(pair_id=pid,combination=j,old=float(m),new=float(n)))
    return {k:picked.get(k) for k in ("correction","loss")}
def preference(rows,pairs,names):
    bank={(r["pair_id"],r["side"],r["view"]):r for r in rows}
    assert len(bank)==len(rows)
    ids=[r["pair_id"] for r in pairs];assert len(set(ids))==len(ids)
    ms={}
    for p in pairs:
        pid=p["pair_id"];winner=p["human_winner_P0"];sign=1 if winner=="A" else -1
        assert winner==("A" if int(p["rank_A"])<int(p["rank_B"]) else "B")
        for k in names:
            scores=[]
            for s in "AB":
                r=bank[pid,s,k]
                assert r["image_id"]==p["image_"+s] and r.get("human",winner)==winner
                assert r.get("P0",p["P0"])==p["P0"]
                val=float(r["score"]);assert float(np.float16(val))==val
                scores.append(val)
            ms[pid,k]=sign*(scores[0]-scores[1])
    return ids,ms
