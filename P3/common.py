from pathlib import Path
import sys
import unicodedata
import json
import hashlib
import itertools

ROOT=Path(__file__).resolve().parent
P2=ROOT.parent/"P2"
PILOT=ROOT.parent/"P1/baseline"
sys.dont_write_bytecode=True
sys.path.extend([str(P2),str(PILOT)])
from study import FAMILIES, combinations, choice, outcome, read_csv, write_csv, write_json, digest, file_json
from views import original,make_view,final_rgb

VIEWS=["O",*FAMILIES["B"],*FAMILIES["C"],*FAMILIES["T"]]
STRATS=["O","BB","CC","TT","BC","BT"]
STATES=["correct","wrong","tie"]
def norm(text):
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())

def frozen():
    m=file_json(ROOT/"freeze.json")
    assert digest(ROOT/"pairs_frozen.csv")==m["pairs_sha256"] and digest(ROOT/"protocol.md")==m["protocol_sha256"]
    rows=read_csv(ROOT/"pairs_frozen.csv")
    old=read_csv(PILOT/"pairs.csv")
    assert len(rows)==m["n"]==len({r["prompt_id"] for r in rows})==len({norm(r["P0"]) for r in rows})==len({r["pair_id"] for r in rows})
    for key in ("prompt_id","P0"):
        fn=norm if key=="P0" else str
        assert not {fn(r[key]) for r in rows}&{fn(r[key]) for r in old}
    ids=lambda rs:{Path(p["image_"+s]).name for p in rs for s in ("A","B")}
    assert len(ids(rows))==2*len(rows) and not ids(rows)&ids(old)
    assert all(int(r["rank_A"])!=int(r["rank_B"]) and r["human_winner_P0"]==("A" if int(r["rank_A"])<int(r["rank_B"]) else "B") for r in rows)
    return rows

