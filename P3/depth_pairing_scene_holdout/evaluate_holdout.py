"""Use the frozen development estimators on independently selected scenes."""
import argparse
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
from io import BytesIO
import json
import numpy as np
from PIL import Image
from zipfile import ZipFile
from holdout_lib import HERE,ORDER,TRANS,CHANGES,VIEWS,SCOPES,read,write,save,sha,frozen_rows,frame_stats,cluster_stats,valid_truth,window_records,pair_record

def summarize_data(frames,rows,out,dev=False):
    groups=defaultdict(list)
    for r in rows:
        key=r["stratum"];fid=r["frame_id"];groups[key,fid].append(r)
        if key in ("small","medium","large"):groups["main",fid].append(r)
    per_frame=[];summary=[];trans=[]
    for s in (("main",) if dev else SCOPES):
        records=[];totals={path:Counter() for path in CHANGES};any_change=any_cross=any_tie=0
        for f in frames:
            fid=f["frame_id"];rr=groups[s,fid]
            rec,counts=frame_stats(fid,f["scene"],s,rr);records.append(rec)
            for path,c in counts.items():totals[path].update(c)
            if s!="equal":
                for r in rr:
                    moves=[(r["state_"+a],r["state_"+b]) for a,b in CHANGES]
                    any_change+=any(a!=b for a,b in moves)
                    any_cross+=any({a,b}=={"wrong","correct"} for a,b in moves)
                    any_tie+=any("tie" in v for v in moves)
        per_frame.extend(records);stats=cluster_stats(records,s);summary.extend(stats)
        if dev:continue
        for (a,b),counts in totals.items():
            if s=="equal":continue
            n=sum(counts.values());assert n==stats[0]["region_pairs"]
            trans.append(dict(scope=s,before=a,after=b,n=n,corrected=counts["wrong","correct"],harmed=counts["correct","wrong"],
                involving_tie=sum(v for key,v in counts.items() if "tie" in key),
                unchanged_strict=counts["wrong","wrong"]+counts["correct","correct"],
                pairs_changed_either_direction=any_change,pairs_strict_flip_either_direction=any_cross,pairs_tie_either_direction=any_tie,
                **{u+"_to_"+v:counts[u,v] for u in ("correct","wrong","tie") for v in ("correct","wrong","tie")}))
    if not dev:
        write(out/"frame_results.csv",per_frame);write(out/"summary.csv",summary);write(out/"transitions.csv",trans)
    return summary

def development():
    frozen_rows()
    src=json.loads((HERE/"sources.json").read_text())["source_files"]
    for n in ("region_pairs.csv","summary.csv","summarize.py","order_data.py"):
        assert sha(ORDER/n)==src["P3/depth_order_pairing/"+n]
    stats=summarize_data(read(TRANS/"frame_results.csv"),read(ORDER/"region_pairs.csv"),HERE,True)
    old={r["metric"]:r for r in read(ORDER/"summary.csv") if r["scope"]=="main"}
    for r in stats:
        for k in ("estimate","ci_low","ci_high"):assert abs(float(r[k])-float(old[r["metric"]][k]))<1e-12
    write(HERE/"development_replay.csv",stats)
    print([(r["metric"],r["estimate"],r["ci_low"],r["ci_high"]) for r in stats if r["metric"] in ("G","A")])

def extract(args):
    frames=frozen_rows()
    env=json.loads((args.out/"environment.json").read_text());assert sha(args.archive)==env["archive_sha256"]
    assert env["freeze_sha256"]==sha(HERE/"freeze.json")
    files=read(args.out/"prediction_files.csv")
    assert [f["frame_id"] for f in files]==[f["frame_id"] for f in frames]
    audit={(r["frame_id"],r["view"]):r for r in read(args.out/"view_audit.csv")}
    assert len(audit)==5*len(files)
    from hashlib import sha256
    ws=[];rs=[]
    with ZipFile(args.archive) as z:
        for idx,(f,entry) in enumerate(zip(frames,files)):
            fid=f["frame_id"];path=args.out/"predictions"/(fid+".npz")
            assert sha(path)==entry["npz_sha256"]
            with np.load(path,allow_pickle=False) as a:
                assert str(a["signature"])==sha(args.out/"environment.json") and str(a["frame_id"])==fid
                pred={v:a[v] for v in VIEWS}
            for v,x in pred.items():
                assert x.dtype==np.float32 and sha256(x.tobytes()).hexdigest()==audit[fid,v]["aligned_sha256"]
            with Image.open(BytesIO(z.read(f["truth"]))) as im:mm=np.asarray(im)
            g,mask=valid_truth(mm);assert all(x.shape==g.shape for x in pred.values())
            windows,px=window_records(f,mm[:,26:614],g,mask,pred,idx<2)
            ws.extend(windows)
            rs.extend(pair_record(a,b,px) for a,b in combinations(windows,2) if a["eligible"] and b["eligible"])
    assert len(rs)==len({(r["frame_id"],r["A"],r["B"]) for r in rs})
    write(args.out/"windows.csv",ws);write(args.out/"region_pairs.csv",rs)
    summarize_data(frames,rs,args.out)
    proof=dict(frames=len(frames),scenes=len({f["scene"] for f in frames}),windows=len(ws),all_pairs=len(rs),
               main_pairs=sum(r["stratum"] in ("small","medium","large") for r in rs),audited_arrays=len(audit),
               max_product_identity_error=max(r["identity_error"] for r in rs),freeze_sha256=sha(HERE/"freeze.json"))
    save(args.out/"validation.json",proof);print(proof)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["development","holdout"]);p.add_argument("--archive",type=Path)
    p.add_argument("--out",type=Path,default=HERE);a=p.parse_args()
    if a.mode=="development":development()
    else:extract(a)
