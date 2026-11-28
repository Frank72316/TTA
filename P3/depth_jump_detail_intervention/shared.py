"""Shared paths and read-only dependencies, pinned in source_sha256.txt."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent / "depth_occlusion_context"
PREVIOUS = HERE.parent / "depth_jump_conditional_assignment"
sys.path.insert(0, str(PREVIOUS))
sys.path.insert(0, str(CONTEXT))
from data import ah, cache, get, included, legacy_row, old, read, save, sha, sources, write
from common import COHORTS, array_path, random_mae

STORE = Path(os.environ.get("DETAIL_DATA", "/home/data/users/shenc/tta_jump_detail_intervention_v1"))
RESULTS = HERE / "results"

def local_cache(record):
    return STORE / record["year"] / record["id"]

def ident(record):
    return {key:record[key] for key in ("year","id","cluster")}

def load_freeze():
    freeze=get(HERE/"freeze.json")
    assert sha(HERE/"protocol.md")==freeze["protocol"] and sha(HERE/"source_sha256.txt")==freeze["sources"]
    assert sha(CONTEXT/"scenes.csv")==freeze["scenes"]
    for line in (HERE/"source_sha256.txt").read_text().splitlines():
        digest,name=line.split()
        assert sha(HERE/name)==digest,name
    return freeze

def signed_base_and_h(record):
    """Open just the previously used arrays (immutable FP32 windows and full image)."""
    import numpy as np
    base=None
    with np.load(cache(record)/"predictions.npz") as f:
        native=f["R"]
        if record["year"]!="2014":
            base=f["B"]
    if base is None:
        with np.load(old.work(legacy_row(record))/"predictions.npz") as f:
            base=f["NB"]
    return base,native
