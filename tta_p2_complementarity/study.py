from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/"tta_pilot"
sys.dont_write_bytecode=True
sys.path.append(str(PILOT))
from pilot_io import digest, read_csv, write_csv, write_json

FAMILIES={"B":["B090","B095","B105","B110"],"C":["C090","C095","C105","C110"],
          "J":["J85","J89","J93","J97"],"T":["Tleft","Tright","Tup","Tdown"]}
VERSIONS=["O",*(v for pool in FAMILIES.values() for v in pool)]
PURE=["BB","CC","JJ","TT"]
MIX=["BC","BJ","BT","CJ","CT","JT"]
PRIMARY=MIX[:3]

def file_json(file):
    return json.loads(file.read_text(encoding="utf8"))

def pairs():
    rows=read_csv(PILOT/"pairs.csv")
    assert digest(PILOT/"pairs.csv")==file_json(PILOT/"manifest.json")["pairs_sha256"]
    assert len(rows)==len({r["P0"] for r in rows})==len({r["pair_id"] for r in rows})==64
    assert len({r["image_"+s] for r in rows for s in ("A","B")})==128
    for r in rows:
        assert int(r["rank_A"])!=int(r["rank_B"]) and r["human_winner_P0"]==("A" if int(r["rank_A"])<int(r["rank_B"]) else "B")
    return rows

def choice(d):
    return "A" if d>0 else "B" if d<0 else "tie"

def outcome(w, human):
    return "tie" if w=="tie" else "correct" if w==human else "wrong"

def combinations(strategy):
    a,b=strategy
    if a==b: return list(itertools.combinations(FAMILIES[a],2))
    return list(itertools.product(FAMILIES[a],FAMILIES[b]))

def pilot_scores():
    ps=pairs()
    m={p["pair_id"]:p for p in ps}
    records=[r for r in read_csv(PILOT/"scores.csv") if r["prompt_condition"]=="P0"]
    bank={}
    for r in records:
        i,s,l=r["pair_id"],r["side"],r["light"]
        k=i,s,{"L0":"O","L1":"B110"}[l]
        assert k not in bank and r["image_id"]==m[i]["image_"+s] and r["native_dtype"]=="float16"
        bank[k]=float(r["score"])
    assert len(bank)==256
    counts=Counter(outcome(choice(bank[p["pair_id"],"A","O"]-bank[p["pair_id"],"B","O"]),p["human_winner_P0"]) for p in ps)
    assert counts=={"correct":43,"wrong":21}
    return bank

if __name__=="__main__":
    pilot_scores()
    print("64 pairs, baseline 43 correct / 21 wrong / 0 ties")

