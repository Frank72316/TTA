"""Decode real left/right responses into candidate identity before paired comparisons."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from common import ROOT, read_csv, write_csv

def comparisons(path):
    mapping=read_csv(ROOT/"human_eval/mapping_private.csv")
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    answers={r["question_id"]:r["choice"].strip() for r in rows}
    assert len(answers)==len(rows) and answers.keys()=={m["question_id"] for m in mapping}
    decode={"左更好":"left","右更好":"right","难以区分":"tie","left":"left","right":"right","tie":"tie","":""}
    picks={}
    for m in mapping:
        v=decode[answers[m["question_id"]]]
        pick=m[v+"_side"] if v in ("left","right") else v
        picks[m["pair_id"],m["condition"],m["light"]]=pick
    rows=[]
    for i in sorted({k[0] for k in picks}):
        for fixed in ("Q0","Q1","L0","L1"):
            light=fixed.startswith("Q")
            a,b=(picks[i,fixed,"L0"],picks[i,fixed,"L1"]) if light else (picks[i,"Q0",fixed],picks[i,"Q1",fixed])
            if not a or not b: result="missing"
            elif a==b: result="both_tie" if a=="tie" else "same"
            elif "tie" in (a,b): result="tie_transition"
            else: result="strict_switch"
            rows.append(dict(pair_id=i,compare="L0_to_L1" if light else "Q0_to_Q1",fixed=fixed,before=a,after=b,result=result))
    return rows, len([v for v in answers.values() if v])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("annotations",type=Path)
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    rows,n=comparisons(args.annotations)
    print(f"已填写 {n}/64；人工验证待完成" if n<64 else "已填写 64/64")
    print({k:dict(Counter(r["result"] for r in rows if r["fixed"]==k)) for k in ("Q0","Q1","L0","L1")})
    if args.out:
        if args.out.exists():
            ap.error("Output exists; use a fresh output path.")
        write_csv(args.out,rows)

