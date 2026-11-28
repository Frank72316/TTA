"""Freeze an annotation-only CUB task. python freeze_pairs.py --meta ./metadata --out pairs.csv"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

COLORS=('black','white','brown','grey','yellow','red','blue','green','orange')
MAX_PAIRS=96
CAP_PER_COLOR=16
SEED=20261128

def read_rows(root,name):
    with (root/name).open() as f:
        for line in f:
            if line.strip():
                yield line.split()

def main(root,out):
    test={int(i) for i,s in read_rows(root,'train_test_split.txt') if s=='0'}
    classes={int(i):int(s) for i,s in read_rows(root,'image_class_labels.txt')}
    names={int(i):s for i,s in read_rows(root,'images.txt')}
    attnames={int(i):s for i,s in read_rows(root,'attributes.txt')}
    attrs={i:(s.split('::')[0][4:-6],s.split('::')[1]) for i,s in attnames.items()
           if s.startswith(('has_crown_color::','has_wing_color::')) and s.split('::')[1] in COLORS}
    assert len(attrs)==len(COLORS)*2
    positions=defaultdict(dict)
    for ix,k,x,y,v in read_rows(root,'part_locs.txt'):
        i,k=int(ix),int(k)
        if i in test and k in (5,9,13) and v=='1':
            positions[i][k]=(float(x),float(y))
    visible={i for i,p in positions.items() if 5 in p and (9 in p or 13 in p)}
    facts={}
    for row in read_rows(root,'image_attribute_labels.txt'):
        i,k,p,c=map(int,row[:4])
        if i in visible and k in attrs:
            key=(i,*attrs[k]);val=(p,c)
            if key in facts: raise ValueError(f'Duplicate fact: {key}')
            facts[key]=val
    pools={c:[[],[]] for c in COLORS}
    for c in COLORS:
        for i in sorted(visible):
            h,w=(facts.get((i,part,c)) for part in ('crown','wing'))
            if not h or not w or min(h[1],w[1]) < 3 or h[0]==w[0]:continue
            pools[c][0 if h[0] else 1].append(i)
    rng=random.Random(SEED)
    for left,right in pools.values():
        rng.shuffle(left);rng.shuffle(right)
    taken=set();chosen=[];counts=Counter()
    for within in (True,False):
        while len(chosen)<MAX_PAIRS:
            advanced=False
            for color in COLORS:
                if counts[color]>=CAP_PER_COLOR or len(chosen)>=MAX_PAIRS:continue
                left,right=pools[color]
                lookup=defaultdict(list)
                for j in right:
                    if j not in taken:lookup[classes[j]].append(j)
                eligible_right=[j for j in right if j not in taken]
                opt=None
                for i in left:
                    if i in taken:continue
                    if within:
                        if lookup[classes[i]]:opt=(i,lookup[classes[i]][0]);break
                    else:
                        j=next((j for j in eligible_right if classes[j]!=classes[i]),None)
                        if j is not None:opt=(i,j);break
                if opt:
                    taken.update(opt);chosen.append((color,*opt));counts[color]+=1;advanced=True
            if not advanced:break
    rng.shuffle(chosen)
    a_head=[True]*(len(chosen)//2)+[False]*(len(chosen)-len(chosen)//2);rng.shuffle(a_head)
    result=[]
    for no,(color,i,j) in enumerate(chosen):
        a,b=(i,j) if a_head[no] else (j,i)
        for role,idx in [('A',a),('B',b)]:
            pt=positions[idx];wing=9 if 9 in pt else 13
            h,w=facts[idx,'crown',color],facts[idx,'wing',color]
            result.append(dict(pair_id=f'C{no+1:03}',side=role,image_id=idx,path=names[idx],species=classes[idx],
                               same_species=int(classes[a]==classes[b]),color=color,crown_present=h[0],wing_present=w[0],
                               crown_certainty=h[1],wing_certainty=w[1],crown_x=pt[5][0],crown_y=pt[5][1],
                               wing_id=wing,wing_x=pt[wing][0],wing_y=pt[wing][1]))
    with out.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(result[0]));wr.writeheader();wr.writerows(result)
    print('official test',len(test),'with visible points',len(visible),'pairs',len(chosen),'within species',sum(classes[i]==classes[j] for c,i,j in chosen),'species',len({classes[i] for i in taken}),'colors',dict(counts))
    print('pools', {c:tuple(map(len,v)) for c,v in pools.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--meta',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.meta,a.out)
