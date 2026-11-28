"""Freeze the complement of all previously used scenes, using only file names and metadata."""
import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from scipy.io import loadmat
from holdout_lib import HERE,P3,read,write,save,sha

def table(path,fields,rows):
    with path.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)

def freeze(a):
    src=json.loads((HERE/"sources.json").read_text())
    old_env=json.loads((P3/"depth_adaptive_budget/environment.json").read_text())
    assert sha(a.archive)==old_env["archive_sha"] and sha(a.split)==old_env["split_sha"]
    # The official HDF5 helper is used separately to populate official_test_scenes.csv.
    # Some Asus historical workspaces keep the metadata helper only in the Git checkout.
    for name,h in src["source_files"].items():
        path=P3.parent/name
        if path.exists():assert sha(path)==h,name
    manifest=read(HERE/"official_test_scenes.csv")
    matlab={int(x) for x in loadmat(a.split)["testNdxs"].ravel()}
    with ZipFile(a.archive) as z:names=set(z.namelist())
    photos={n for n in names if "/images/val/" in n and n.endswith(".jpg")}
    assert len(manifest)==len(photos)==len(matlab)==654 and {r["frame"] for r in manifest}==photos
    assert {int(Path(n).stem[4:])+1 for n in photos}==matlab
    groups=defaultdict(list);byid={}
    for r in manifest:
        fid=Path(r["frame"]).stem;truth=r["frame"].replace("/images/","/depth/")[:-4]+".png"
        assert truth in names and r["scene"] and "_" in r["scene"]
        row=dict(frame_id=fid,frame=r["frame"],truth=truth,scene=r["scene"],matlab_index=int(fid[4:])+1)
        byid[fid]=row;groups[r["scene"]].append(fid)
    hist=read(HERE/"history.csv");used={r["frame_id"] for r in hist};assert used<=set(byid) and len(used)==len(hist)
    prior=read(P3/"depth_adaptive_budget/frame_results.csv")
    assert all(byid[r["frame_id"]]["scene"]==r["scene"] for r in prior)
    excluded={byid[fid]["scene"] for fid in used}
    keep=[v for k,v in sorted(byid.items()) if v["scene"] not in excluded]
    assert not {r["scene"] for r in keep}&excluded and not used&{r["frame_id"] for r in keep}
    dropped=[]
    for scene in sorted(excluded):
        ids=sorted(groups[scene]);old=sorted(used&set(ids))
        dropped.append(dict(scene=scene,test_frames=len(ids),historical_frames=len(old),used_ids=" ".join(old),all_excluded_ids=" ".join(ids)))
    # Refuse to change an existing freeze; regenerating the same snapshot is safe.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)
        table(dest/"frames.csv",["frame_id","frame","truth","scene","matlab_index"],keep)
        table(dest/"excluded_scenes.csv",["scene","test_frames","historical_frames","used_ids","all_excluded_ids"],dropped)
        hashes={k:sha(HERE/k) for k in ("protocol.md","sources.json","official_test_scenes.csv","history.csv","freeze_holdout.py","inventory.py")}
        hashes.update({k:sha(dest/k) for k in ("frames.csv","excluded_scenes.csv")})
        f=dict(total_scenes=len(groups),total_frames=len(byid),directly_used_frames=len(used),
               excluded_scenes=len(excluded),excluded_frames=sum(r["test_frames"] for r in dropped),
               retained_scenes=len(groups)-len(excluded),retained_frames=len(keep),historical_scene_overlap=[],
               metadata="https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat#scenes",
               split_sha256=sha(a.split),archive_sha256=sha(a.archive),prediction_commit=src["prediction_commit"],
               inventory_and_evaluation_commit=src["evaluation_and_inventory_commit"],bootstrap_replicates=10000,seed=20261128,
               primary="G",key_secondary="A",hashes=hashes)
        assert f["retained_frames"]+f["excluded_frames"]==654
        save(dest/"freeze.json",f)
        for k in ("frames.csv","excluded_scenes.csv","freeze.json"):
            data=(dest/k).read_bytes()
            if (HERE/k).exists():assert (HERE/k).read_bytes()==data,k
            else:(HERE/k).write_bytes(data)
    print({k:v for k,v in f.items() if k.endswith(("frames","scenes","overlap"))})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--split",type=Path,required=True)
    freeze(p.parse_args())
