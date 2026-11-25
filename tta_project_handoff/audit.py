"""Reconcile the reported subset against read-only caches, with no model or resampling experiment."""
from pathlib import Path
import csv,json,hashlib,itertools
from collections import Counter
import numpy as np
R=Path(__file__).resolve().parent; W=R.parent
P="tta_pilot"; Q="tta_round2"; H="tta_round3_analysis"; N="tta_round4_repairing"; E="tta_p2_complementarity"; F="tta_p3_translation"
sources={}
def rows(directory,name):
    p=W/directory/name;sources[str(p.relative_to(W)).replace(chr(92),"/")]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def index(rs):return {r["pair_id"]:r for r in rs}
def pick(d):return "A" if d>0 else "B" if d<0 else "同分"
def check(v):assert v
def main():
    old=index(rows(P,"pairs.csv"));new=index(rows(F,"pairs_frozen.csv"));q=index(rows(Q,"prompts_round2.csv"));pilotres=index(rows(P,"pair_results.csv"));qres=index(rows(Q,"pair_results_round2.csv"));dec=index(rows(H,"pair_decomposition.csv"))
    rank=lambda p:"A" if int(p["rank_A"])<int(p["rank_B"]) else "B"
    assert len(old)==64 and len(new)==256 and old.keys()==q.keys()==qres.keys()==dec.keys()
    for p in [*old.values(),*new.values()]:assert p["human_winner_P0"]==rank(p) and p["rank_A"]!=p["rank_B"]
    assert not {p["prompt_id"] for p in old.values()} & {p["prompt_id"] for p in new.values()}
    suffix=" The scene is at night, with dim lighting."
    for i,p in old.items():
        assert q[i]["original_prompt"]==p["P0"] and q[i]["Q1"]==q[i]["Q0"]+suffix and p["P1"]==p["P0"]+suffix
        for s in "AB":assert p["image_"+s]==q[i]["image_"+s]==dec[i]["image_"+s]
    a=rows(Q,"old_prompt_audit.csv")
    assert [sum(int(r["light"]) for r in a),sum(1-int(r["P1_complete"]) for r in a),sum(int(r["identical"]) for r in a)]==[23,9,3]
    def raw(d,name):
        rr=rows(d,name);kv={};assert len(rr)==512
        for r in rr:
            i,s,t,l=[r[x] for x in ["pair_id","side","prompt_condition","light"]]
            assert old[i]["image_"+s]==r["image_id"] and (i,s,t,l) not in kv
            kv[i,s,t,l]=float(r["score"])
            if d==Q:assert r["text_token_sha"]==q[i][t+"_token_sha"]
        return kv
    bat=raw(P,"scores.csv");short=raw(Q,"scores_round2.csv")
    lit={};hs=[];gs=[]
    for i in old:
        prev={}
        for l in ("L0","L1"):
            prev[l]=bat[i,"A","P0",l]-bat[i,"B","P0",l]
        avg=(prev["L0"]+prev["L1"])/2
        assert [prev["L0"],prev["L1"],avg]==[float(pilotres[i][k]) for k in ("P0_d0","P0_d1","P0_d_tta")]
        lit[i]=[prev["L0"],prev["L1"],avg]
        b={(s,t):short[i,s,t,"L1"]-short[i,s,t,"L0"] for s in "AB" for t in ("Q0","Q1")}
        h=[b[s,"Q1"]-b[s,"Q0"] for s in "AB"];hh=abs(b["A","Q1"]-b["B","Q1"])-abs(b["A","Q0"]-b["B","Q0"])
        assert h==[float(dec[i]["h_"+s]) for s in "AB"] and hh==float(qres[i]["G"]);hs.append(h);gs.append(hh)
    assert sum(pick(lit[i][0])==old[i]["human_winner_P0"] for i in old)==43
    assert sum(x[0]*x[1]<0 for x in lit.values())==2 and sum(x[0]*x[2]<0 for x in lit.values())==0
    assert [sum(v>0 for v in gs),sum(v<0 for v in gs),sum(v==0 for v in gs)]==[23,34,7]
    a,b=np.array(hs).T;den=sum(a*a+b*b);n=len(a)
    r=.5+sum(a*b)/den;random=.5+(sum(a)*sum(b)-sum(a*b))/((n-1)*den)
    m=int(sum(a*b>0));random_m=sum(a[i]*b[j]>0 for i in range(n) for j in range(n) if i!=j)/(n-1)
    ss={x["metric"]:x for x in rows(N,"summary.csv") if x["scope"]=="all" and x["design"]=="derangement"}
    for k,v,z in [("R",r,random),("same_nonzero",m,random_m)]:
        assert np.isclose(float(ss[k]["observed"]),v) and np.isclose(float(ss[k]["exact_null_mean"]),z)
    fixed=rows(N,"null_distribution.csv")
    for k in ss:
        vv=[o for o in fixed if o["scope"]=="all" and o["design"]=="derangement" and o["metric"]==k];assert sum(int(o["frequency"]) for o in vv)==1000000
    views={"B":["B090","B095","B105","B110"],"C":["C090","C095","C105","C110"],"T":["Tleft","Tright","Tup","Tdown"]}
    case={};mix={};detail={};allresults={}
    for directory,pairs in [(E,old),(F,new)]:
        rs=rows(directory,"scores.csv");kv={};checks=rows(directory,"combination_results.csv")
        saved={(x["pair_id"],x["strategy"],x["view1"],x["view2"]):x for x in checks}
        assert len(saved)==len(checks)
        for v in rs:
            i,s,view=[v[x] for x in ("pair_id","side","view")];p=pairs[i]
            assert p["image_"+s]==v["image_id"] and p["P0"]==v["P0"] and (i,s,view) not in kv
            kv[i,s,view]=float(v["score"])
        assert len(kv)==(2176 if directory==E else 6656)
        for i,p in pairs.items():
            base=kv[i,"A","O"]-kv[i,"B","O"];human=p["human_winner_P0"];cnt={};matrix=[];combo_data={}
            for st in ("BB","CC","TT","BC","BT"):
                vset=itertools.combinations(views[st[0]],2) if st[0]==st[1] else itertools.product(views[st[0]],views[st[1]])
                tags=Counter()
                for x,y in vset:
                    av=[sum(kv[i,s,t] for t in ("O",x,y))/3 for s in "AB"]
                    margin=sum(kv[i,"A",t]-kv[i,"B",t] for t in ("O",x,y))/3
                    prior=saved[i,st,x,y];assert margin==float(prior.get("margin",prior.get("d_aug")))
                    state=pick(margin);tags["right" if state==human else "tie" if state=="同分" else "wrong"]+=1
                    tags["changed"]+=int(state!=pick(base))
                    combo_data[x+" "+y]=[av[0],av[1],margin,state]
                    if st=="BT":matrix.append(state)
                cnt[st]=dict(tags)
            mix[i]=(pick(base),human,cnt,base)
            if i in ["003","008","N032","N141","053","005"]:
                case[i]=dict(prompt=p["P0"],human=human,orig=[kv[i,s,"O"] for s in "AB"],d0=base,rank=[p["rank_A"],p["rank_B"]],matrix=matrix,combo=combo_data,
                    single={v:[kv[i,s,v] for s in "AB"] for v in ["O",*views["B"],*views["T"]]},counts=cnt)
    assert [mix["003"][2][s].get("right",0) for s in ("TT","BT")]==[4,4] and mix["008"][2]["BT"]["wrong"]==16
    assert [(i,c["BC"].get("right",0)) for i,(k,h,c,d) in mix.items() if i in old and k!=h and c["BC"].get("right",0)]==[("053",1)]
    unique=[];and_tt=[];change=[];budget={};mov={}
    for st in ("BT","BC"):
        tr=Counter()
        for i in new:
            p,h,c,d=mix[i];src="right" if p==h else "wrong";assert p!="同分"
            for x in ["right","wrong","tie"]:tr[src+"_"+x]+=c[st].get(x,0)
        budget[st]={k:v/16 for k,v in tr.items()}
    diffs=[]
    for i in new:
        p,h,c,d=mix[i]
        if p!=h and c["BT"].get("right",0)>0 and not any(c[st].get("right",0) for st in ["BB","CC","BC"]):
            unique.append(i)
            if c["TT"].get("right"):and_tt.append(i)
        if c["BT"]["changed"] or c["BC"]["changed"]:change.append(i)
        diffs.append((c["BT"].get("right",0)-c["BC"].get("right",0))/16)
    assert len(unique)==7 and and_tt==unique
    assert len(change)==21 and set(change)<=set(sorted(new,key=lambda i:(abs(mix[i][3]),i))[:64])
    ci=np.percentile(np.asarray(diffs)[np.random.default_rng(2026).integers(0,256,(10000,256))].mean(1)*100,[2.5,97.5])
    endpoint=json.loads((W/F/"endpoint.json").read_text());assert np.array_equal(ci,endpoint["ci_pp"]) and np.mean(diffs)*100==endpoint["bt_minus_bc_pp"]
    for i,mat in [("N032",["B","B","A","B"]*4),("N141",["A"]*16)]:assert case[i]["matrix"]==mat
    for i,p in case.items():
        if i in old:p["pilot"]=lit[i]
    direct=rows(F,"direction_counts.csv");nd={}
    for i in {x["pair_id"] for x in direct}:
        d=[int(x["n_changed"]) for x in direct if x["pair_id"]==i];nd[i]=d
    assert len(nd)==17 and sum(sum(i>0 for i in v)>=2 for v in nd.values())==9 and sum(max(v)>1 for v in nd.values())==16
    bins=rows(F,"margin_groups.csv");assert [int(x["any_BT_BC_change"]) for x in bins]==[21,0,0,0,0]
    for k in (E,F):
        for x in ("pair_strategy_results.csv","strategy_summary.csv"):rows(k,x)
    rows(Q,"summary.csv");rows(H,"summary.csv");rows(F,"transitions.csv")
    snapshots={d:[str(x.relative_to(W)).replace(chr(92),"/") for x in (W/d).glob("*.py")] for d in (P,Q,H,N,E,F)}
    for d in snapshots:
        for fn in ("README.md","report.md"):
            fnp=W/d/fn;sources[f"{d}/{fn}"]=hashlib.sha256(fnp.read_bytes()).hexdigest()
    out=dict(cases=case,true_R=r,null_R=random,true_same=m,null_same=random_m,medianG=float(np.median(gs)),new_cases=unique,
             endpoint=endpoint,budget=budget,margin_groups=bins,random_reference=ss,source_hashes=sources,input_files=snapshots)
    (R/"audit.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print("Checked 6 stages, 64 + 256 independent prompt IDs, 28 display inputs in separate pixel audit.",budget,ci)
if __name__=="__main__":main()

