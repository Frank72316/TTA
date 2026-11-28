"""Reconstruct the historical seven from per-image scores before scoring the new pool."""
import argparse
import csv
import hashlib
from itertools import combinations, product
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def freeze(p3=None):
    source = HERE.parent / "translation"
    pairs = read(source/"pairs_frozen.csv")
    scores = read(source/"scores.csv")
    bank = {(r["pair_id"],r["side"],r["view"]):r for r in scores}
    assert len(bank)==len(scores)==6656
    pools = {f:sorted({r["view"] for r in scores if r["view"].startswith(f)}) for f in "BCT"}
    old = {(r["pair_id"],r["strategy"]):r for r in read(source/"pair_strategy_results.csv")}
    chosen=[]
    for p in pairs:
        i=p["pair_id"]
        win=p["human_winner_P0"]
        assert win==("A" if int(p["rank_A"])<int(p["rank_B"]) else "B")
        m={}
        for view in ["O",*(v for vs in pools.values() for v in vs)]:
            a,b=(bank[i,s,view] for s in "AB")
            assert a["image_id"]==p["image_A"] and b["image_id"]==p["image_B"] and p["P0"]==a["P0"]==b["P0"]
            m[view]=(float(a["score"])-float(b["score"]))*(1 if win=="A" else -1)
        counts={}
        for st in ("BB","CC","BC","BT","TT"):
            x,y=st
            specs=list(combinations(pools[x],2) if x==y else product(pools[x],pools[y]))
            counts[st]=sum(m["O"]+m[s]+m[t]>0 for s,t in specs)
            assert counts[st]==int(old[i,st]["correct"])
        if m["O"]<0 and counts["BT"]>0 and not any(counts[s] for s in ("BB","CC","BC")):
            chosen.append(dict(pair_id=i,prompt_id=p["prompt_id"],P0=p["P0"],human=win,m0=m["O"],**counts))
    assert len(chosen)==7
    out=HERE/"historical_seven.csv"
    if out.exists():
        assert read(out)==[{k:str(v) for k,v in r.items()} for r in chosen]
    else:
        write(out,chosen)
    print("Frozen", " ".join(r["pair_id"] for r in chosen),hashlib.sha256(out.read_bytes()).hexdigest())
    if p3:
        original={(r["pair_id"],r["side"]):r for r in read(p3/"scores.csv") if r["view"]=="O"}
        assert len(original)==2*len(pairs)
        for p in pairs:
            for side in "AB":
                row=original[p["pair_id"],side]
                assert (row["P0"],row["prompt_id"],row["image_id"])==(p["P0"],p["prompt_id"],p["image_"+side])
        selected={r["pair_id"] for r in chosen}
        prior=[r for r in read(p3/"pair_results.csv") if r["pair_id"] in selected]
        assert len(prior)==28 and {(r["pair_id"],int(r["budget_per_image"])) for r in prior}==set(product(selected,(1,2,4,8)))
        for r in prior:
            i=r["pair_id"]
            assert float(r["original_margin_a_minus_b"])==float(original[i,"A"]["score"])-float(original[i,"B"]["score"])
        target=HERE/"historical_p3_budget.csv"
        if target.exists():
            assert read(target)==prior
        else:
            write(target,prior)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--p3",type=Path,help="P3/results/development-3 directory from the archived study")
    freeze(parser.parse_args().p3)
