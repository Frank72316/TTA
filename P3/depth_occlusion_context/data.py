"""Frozen Middlebury scene inventory, metric depth, and read-only reference paths."""
import hashlib
import os
import re
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "depth_occlusion_tta"
sys.path.insert(0, str(LEGACY))
import io_data as old
from io_data import ah, get, read, save, sha, write

DATA = Path(os.environ.get("CTX_DATA", "/home/data/users/shenc/tta_occlusion_context"))
SCENES = {
    "2005": "Art Books Dolls Laundry Moebius Reindeer",
    "2006": "Aloe Baby1 Baby2 Baby3 Bowling1 Bowling2 Cloth1 Cloth2 Cloth3 Cloth4 Flowerpots Lampshade1 Lampshade2 Midd1 Midd2 Monopoly Plastic Rocks1 Rocks2 Wood1 Wood2",
}

def key(r):
    return r["year"] + "/" + r["id"]

def legacy_row(r):
    return {"dataset": "middlebury", "id": r["id"]}

def cache(r):
    return DATA / "pred" / key(r)

def sources(r):
    if r["year"] == "2014":
        return old.paths(legacy_row(r))
    d = DATA / "source" / key(r)
    return d / "view1.png", d / "disp1.png", d / "dmin.txt"

def metric(r):
    if r["year"] == "2014":
        return old.depth(legacy_row(r))
    _, disp_path, offset_path = sources(r)
    disp = np.array(Image.open(disp_path))
    assert disp.dtype == np.uint8 and disp.ndim == 2
    dmin = int(offset_path.read_text().strip())
    z = np.full(disp.shape, np.nan, np.float32)
    valid = disp > 0
    z[valid] = (.160 * 3740 / (disp[valid].astype("float64") + dmin)).astype("float32")
    return z

def all_rows():
    rows = []
    for year,names in SCENES.items():
        for name in names.split():
            rows.append(dict(year=year,id=name,cluster=year+"/"+re.sub(r"\d+$","",name),
                        camera="view1",exposure="Illum1/Exp"+("1" if year=="2005" else "2")))
    for row in read(LEGACY/"cache_index.csv"):
        rows.append(dict(year="2014",id=row["id"],cluster="2014/"+row["id"],camera="im0",exposure="default perfect"))
    return sorted(rows,key=lambda r:hashlib.sha256(key(r).encode()).digest())

def included():
    return [r for r in read(ROOT/"scenes.csv") if r["include"]=="1"]
