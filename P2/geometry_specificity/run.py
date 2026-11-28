"""Run smoke, calibration, freeze, and development in order, then merge disjoint shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from history import HERE, read, write


def execute(args):
    started=perf_counter()
    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    worker_count=1 if args.device=="cuda" else args.workers
    def stage(n,pairs,cohort):
        folder=root/cohort
        folder.mkdir(exist_ok=True)
        def worker(idx,end=None):
            sub=folder/f"shard{idx}"
            sub.mkdir(exist_ok=True)
            start=n*idx//worker_count
            stop=end if end else n*(idx+1)//worker_count
            cmd=[py,"-Bu",str(HERE/"score.py"),"--out",str(sub),"--pairs",str(pairs),"--pilot",str(args.pilot),
                 "--device",args.device,"--batch",str(args.batch),"--start",str(start),"--limit",str(stop)]
            if n==256:
                cmd+=["--frozen-strength",str(root/"strength_freeze.json")]
            with (sub/"log.txt").open("ab") as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        worker(0,2)
        smoke=read(folder/"shard0/scores.csv")
        assert len(smoke)>=132
        for p in read(pairs)[:2]:
            for side in "AB":
                subset=[r for r in smoke if r["pair_id"]==p["pair_id"] and r["side"]==side]
                assert len(subset)==33 and len({r["tensor_sha"] for r in subset})>1
                assert all(float(r["mean_abs_rgb255"])>0 for r in subset if r["view"][0] in "TS")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(worker,range(worker_count)))
        allrows=[]
        runs=[]
        for idx in range(worker_count):
            sub=folder/f"shard{idx}"
            allrows.extend(read(sub/"scores.csv"))
            runs.extend(json.loads((sub/"runs.json").read_text()))
        assert len(allrows)==len({(r["pair_id"],r["side"],r["view"]) for r in allrows})==66*n
        write(folder/"scores.csv",allrows)
        (folder/"environment.json").write_bytes((folder/"shard0/environment.json").read_bytes())
        (folder/"runs.json").write_text(json.dumps(runs,indent=2))
        subprocess.run([py,"-B",str(HERE/"analyze.py"),"analyze","--scores",str(folder/"scores.csv"),
                        "--n",str(n),"--out",str(folder),"--freeze",str(root/"strength_freeze.json")],check=True)
    stage(64,args.old,"old64")
    subprocess.run([py,"-B",str(HERE/"analyze.py"),"freeze","--scores",str(root/"old64/scores.csv"),"--out",str(root)],check=True)
    stage(256,args.dev,"dev256")
    (root/"completed.json").write_text(json.dumps(dict(device=args.device,shards=worker_count,batch=args.batch,
                                        elapsed_s=perf_counter()-started),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("pilot","old","dev","out"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--workers",type=int,default=1)
    execute(p.parse_args())
