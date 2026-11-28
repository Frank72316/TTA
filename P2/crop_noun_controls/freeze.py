"""Match entire noun pairs to eligible recipients without score access."""
import argparse,csv,hashlib,json,random
from collections import defaultdict,Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def put(p,records):
    with p.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,records[0],lineterminator="\n");w.writeheader();w.writerows(records)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(path):
    src=HERE.parent/"prompt_aware_flip"; orig=json.loads((src/"freeze.json").read_text())
    assert sha(path)==orig["annotation_sha256"]
    ann=json.loads(path.read_bytes()); names={c["id"]:c["name"] for c in ann["categories"]}
    present=defaultdict(set)
    for a in ann["annotations"]:present[a["image_id"]].add(names[a["category_id"]])
    pairs=read(src/"pairs.csv"); out=[]; audit=[]
    for seed in (42,43,44):
        rng=random.Random(seed); n=len(pairs); neighbors={}
        for i,p in enumerate(pairs):
            excluded=present[int(p["image_A"])]|present[int(p["image_B"])]
            assert {p["x"],p["y"]}<=excluded
            neighbors[i]=[j for j,q in enumerate(pairs) if j!=i and not ({q["x"],q["y"]}&excluded)]
            rng.shuffle(neighbors[i])
        owner={}
        def match(i,visited):
            for j in neighbors[i]:
                if j in visited:continue
                visited.add(j)
                if j not in owner or match(owner[j],visited):
                    owner[j]=i;return True
            return False
        order=list(range(n));rng.shuffle(order)
        for i in order:match(i,set())
        donors={i:j for j,i in owner.items()}
        for i,p in enumerate(pairs):
            q=pairs[donors[i]] if i in donors else None
            out.append(dict(seed=seed,pair_id=p["pair_id"],donor=q["pair_id"] if q else "",x=p["x"],y=p["y"],
                alt_x=q["x"] if q else "",alt_y=q["y"] if q else "",eligible_donors=len(neighbors[i]),
                excluded="|".join(sorted(present[int(p["image_A"])]|present[int(p["image_B"])]))))
        if len(owner)==n:
            assert all(Counter(p[k] for p in pairs)==Counter(pairs[j][k] for j in owner) for k in ("x","y"))
        audit.append(dict(seed=seed,matched=len(owner),unmatched=[pairs[i]["pair_id"] for i in range(n) if i not in donors]))
    put(HERE/"mappings.csv",out)
    (HERE/"freeze.json").write_text(json.dumps(dict(annotations=sha(path),mappings=sha(HERE/"mappings.csv"),
        pairs=sha(src/"pairs.csv"),protocol=sha(HERE/"protocol.md"),runs=audit),indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("annotations",type=Path);run(ap.parse_args().annotations)
