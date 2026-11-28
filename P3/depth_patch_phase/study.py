"""Frozen design and small IO helpers. This module does not import a model."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "depth_adaptive_budget"
SEED = 20261128
AMPS = (11,17,25,31)
SHIFTS = (0, *[d for a in AMPS for d in (-a,a)])
PAIRS = ((11,25),(17,31),(11,31),(17,25))
ARMS = ("reflect","real")
BORDER = 42
TRIM = 70

def sha(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    with open(path,encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def write(path,rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,list(rows[0]),lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(path)

def save(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode())
    tmp.replace(path)

def load(path):
    return json.loads(path.read_text(encoding="utf8"))

def key(dx):
    return "O" if dx==0 else ("R" if dx>0 else "L")+str(abs(dx))

def phase(a):
    return (a+7)%14-7
