import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
P3=HERE.parent
ORDER=P3/"depth_order_pairing"
TRANS=P3/"depth_translation_pairing"
sys.path.insert(0,str(ORDER))
from order_data import window_records,pair_record,VIEWS,valid_truth,read,write,save,sha,CHANGES
from summarize import frame_stats,cluster_stats,SCOPES

def frozen_rows():
    f=json.loads((HERE/"freeze.json").read_text())
    for name,h in f["hashes"].items():assert sha(HERE/name)==h,name
    rows=read(HERE/"frames.csv")
    assert len(rows)==len({r["frame_id"] for r in rows})==f["retained_frames"]
    return rows
