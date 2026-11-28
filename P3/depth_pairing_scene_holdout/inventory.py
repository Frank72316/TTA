"""Audit existing NYU usage in the fixed, pre-holdout repository snapshot."""
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
REV="8a8d0cfcb8702c2df5be01b4e67c6a2f252e9fe4"
PRED="3e0ea7c340a5bf27faf784b6e16a45a27571a7f0"

def git(*args):
    return subprocess.check_output(["git",*args],cwd=HERE.parents[1])

def inventory():
    regex=r"nyu_[0-9]{4}"
    found=git("grep","-l","-I","-E",regex,REV,"--","*.csv","*.json","*.jsonl","*.tsv").decode().splitlines()
    files=defaultdict(set)
    for entry in found:
        _,name=entry.split(":",1)
        blob=git("show",REV+":"+name)
        for fid in set(re.findall(rb"nyu_[0-9]{4}(?![0-9])",blob)):files[fid.decode()].add(name)
    ref=git("show",REV+":P3/results/depth/depth_moments.json")
    early={Path(r["frame"]).stem for r in json.loads(ref)}
    later={r["frame_id"] for r in csv.DictReader(git("show",REV+":P3/depth_adaptive_budget/frames.csv").decode().splitlines())}
    assert len(early)==64 and len(later)==128 and not early&later and early|later<=set(files)
    with (HERE/"history.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(["frame_id","early64","later128","sources"])
        for fid,paths in sorted(files.items()):w.writerow([fid,int(fid in early),int(fid in later),"|".join(sorted(paths))])
    paths={"P3/depth_translation_pairing/"+n:PRED for n in ("run.py","metrics.py")}
    paths.update({"P3/depth_order_pairing/"+n:REV for n in ("order_data.py","summarize.py","verify.py","summary.csv","region_pairs.csv")})
    paths.update({"P3/depth_adaptive_budget/"+n:REV for n in ("scene_metadata.py","frames.csv","environment.json")})
    obj={"prediction_commit":PRED,"evaluation_and_inventory_commit":REV,
         "source_files":{n:hashlib.sha256(git("show",v+":"+n)).hexdigest() for n,v in paths.items()}}
    (HERE/"sources.json").write_bytes((json.dumps(obj,indent=2)+"\n").encode())
    print(len(files),"distinct historical frames from",len(found),"structured artifacts")
if __name__=="__main__":inventory()
