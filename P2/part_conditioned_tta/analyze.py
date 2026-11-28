"""Analyze genuine HPS scores in CSV with image_id,view,query,score. Run: python analyze.py scores.csv"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
VIEWS=('original','crown','wing','center')
QUERIES=('crown','wing')
METHODS=('original','center','matched','swapped')

def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def run(manifest,data,out):
    groups=defaultdict(list)
    for r in read(manifest):
        groups[r['pair_id']].append(r)
    scores={}
    for r in read(data):
        key=(r['image_id'],r['view'],r['query'])
        if key in scores: raise ValueError(f'duplicate score {key}')
        val=float(r['score'])
        if not np.isfinite(val):raise ValueError(key)
        scores[key]=val
    expected={(r['image_id'],v,q) for rows in groups.values() for r in rows for v in VIEWS for q in QUERIES}
    if scores.keys()!=expected:raise ValueError('score rows disagree with manifest × views × queries')
    results=[]; strata=[]
    for pid,rows in sorted(groups.items()):
        if len(rows)!=2 or {r['side'] for r in rows}!={'A','B'}:raise ValueError(pid)
        a,b=sorted(rows,key=lambda r:r['side'])
        if a['species']!=b['species']:raise ValueError('expected same-species pairs')
        both_definite=all(int(r[k])==4 for r in rows for k in ('crown_certainty','wing_certainty'))
        item=dict(pair_id=pid,species=a['species'],color=a['color'],definite=int(both_definite))
        for q in QUERIES:
            if {a[q+'_present'],b[q+'_present']}!={'0','1'}:raise ValueError(pid)
            direction=1 if a[q+'_present']=='1' else -1
            def margin(v):
                return direction*(scores[a['image_id'],v,q]-scores[b['image_id'],v,q])
            m0=margin('original')
            for method,view in [('original','original'),('center','center'),('matched',q),('swapped','wing' if q=='crown' else 'crown')]:
                item[method+'_'+q]=m0 if method=='original' else (m0+margin(view))/2
        strata.append(a['species']);results.append(item)
    out.mkdir(exist_ok=True,parents=True)
    with (out/'pair_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    vectors={}; summary={}
    for method in METHODS:
        a=np.array([[r[method+'_'+q] for q in QUERIES] for r in results])
        correct=(a>0)
        vectors[method]=correct.mean(axis=1)
        summary[method]=dict(crown_correct=int(correct[:,0].sum()),wing_correct=int(correct[:,1].sum()),
                             tie_questions=int((a==0).sum()),both_correct=int(correct.all(axis=1).sum()),
                             strict_accuracy=float(correct.mean()))
    lab=np.array(strata); clusters=[np.flatnonzero(lab==s) for s in sorted(set(lab))]
    rng=np.random.default_rng(20261128)
    boot=[np.concatenate([clusters[k] for k in rng.integers(len(clusters),size=len(clusters))]) for _ in range(10000)]
    for ref in ('swapped','original','center'):
        d=vectors['matched']-vectors[ref]
        summary['matched_minus_'+ref]={'delta_pp':float(d.mean()*100),'ci_pp':(100*np.quantile([d[ix].mean() for ix in boot],[.025,.975])).tolist()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scores',type=Path);p.add_argument('--manifest',type=Path,default=ROOT/'pairs.csv');p.add_argument('--out',type=Path,default=ROOT/'results');a=p.parse_args();run(a.manifest,a.scores,a.out)
