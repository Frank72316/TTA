"""Read and check the frozen upstream files without modifying them."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/"part_conditioned_tta"
sys.path.insert(0, str(OLD))
from prepare import as_json, digest, read, write

VIEWS = ("original","crown","wing")
QUERIES = ("crown","wing")
SMOKE = ("C001","C002")
REV = "3cafdb28e17aa7d51197c520885423f010783746"

def frozen():
    spec=json.loads((ROOT/"source.json").read_text(encoding="utf8"))
    assert spec["commit"]==REV
    for name,h in spec["sha256"].items():
        assert digest(ROOT.parent/name)==h, name
    return spec

def wording(q):
    return f"The bird has plumage on its {q}."

def check_scores(rows,photos,neutral):
    by_id={p["image_id"]:p for p in photos}
    data={}
    for s in rows:
        i,v,q=(s[k] for k in ("image_id","view","query"))
        p=by_id[i]
        assert (s["pair_id"],s["side"])==(p["pair_id"],p["side"])
        assert s["prompt"]==(wording(q) if neutral else f"The bird has {p['color']} plumage on its {q}.")
        assert (i,v,q) not in data
        data[i,v,q]=s
    assert set(data)=={(i,v,q) for i in by_id for v in VIEWS for q in QUERIES}
    return data
