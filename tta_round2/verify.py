"""Independent integrity checks on margins, frozen inputs, bootstrap, and blinded images."""
import hashlib
import json
import re
from collections import Counter
from itertools import product
import numpy as np
from PIL import Image
from common import *
from import_annotations import comparisons

def main():
    frozen_rows=frozen()
    summaries={s["group"]:s for s in read_csv(ROOT/"summary.csv")}
    declarations=[("old_all",OLD/"scores.csv","old_audit.csv",("P0","P1")),
                  ("new_all",ROOT/"scores_round2.csv","pair_results_round2.csv",("Q0","Q1"))]
    for scope,path,details,conds in declarations:
        scores=index_scores(read_csv(path),PAIRS,conds)
        results=read_csv(ROOT/details)
        assert len(results)==64 and {r["pair_id"] for r in results}==PAIRS.keys()
        gains=[]
        for r in results:
            i=r["pair_id"]
            ds=[]
            for c in conds:
                vals=np.array([[scores[i,s,c,l] for l in ("L0","L1")] for s in ("A","B")],np.float64)
                margins=vals[0]-vals[1]
                changes=vals[:,1]-vals[:,0]
                assert (float(r[c+"_d_tta"])==np.diff(-vals.mean(axis=1))[0]==margins.mean())
                d=changes[0]-changes[1]
                assert d==margins[1]-margins[0]==float(r[c+"_delta"])
                ds.append(d)
            gain=abs(ds[1])-abs(ds[0])
            assert gain==float(r["G"]) and ds[1]-ds[0]==float(r["I"])
            gains.append(gain)
        s=summaries[scope]
        assert np.median(gains)==float(s["G_median"])
        for j in (0,1):
            assert sum(int(s[f"c{j}_bin{k}"]) for k in range(6))==64
        if scope=="new_all":
            indices=np.random.default_rng(42).choice(64,(10000,64),replace=True)
            ci=np.quantile(np.median(np.array(gains)[indices],axis=1),[.025,.975])
            assert list(ci)==[float(s["bootstrap_low"]),float(s["bootstrap_high"])]
    assert sum(int(summaries[k]["n"]) for k in ("unlit_complete","lit_complete","truncated"))==64
    cache_rows=[r for p in frozen_rows for r in jsonfile(ROOT/"cache"/(p["pair_id"]+".json"))["rows"]]
    scores=read_csv(ROOT/"scores_round2.csv")
    assert [{k:str(v) for k,v in r.items()} for r in cache_rows]==scores
    assert {(r["pair_id"],r["light"],r["prompt_condition"]) for r in scores}==set(product(PAIRS,("L0","L1"),("Q0","Q1")))
    assert all(r["source"]=="round2_fresh" for r in scores)
    blind=ROOT/"human_eval"
    html=(blind/"index.html").read_text(encoding="utf-8")
    data=json.loads(re.search(r"const tasks=(.*?), key=",html).group(1))
    mapping=read_csv(blind/"mapping_private.csv")
    assert [r["question_id"] for r in mapping]==[q["id"] for q in data]
    ids={r["pair_id"] for r in mapping}
    assert len(ids)==16 and len(mapping)==len(data)==64
    assert {(m["pair_id"],m["condition"],m["light"]) for m in mapping}==set(product(ids,("Q0","Q1"),("L0","L1")))
    texts={p["pair_id"]:p for p in frozen_rows}
    expected={}
    for i in sorted(ids):
        for (s,l),img in zip(CELLS,images(i)):
            expected[i,s,l]=hashlib.sha256(img.tobytes()).hexdigest()
    for q,m in zip(data,mapping):
        assert set(q)=={"id","prompt","left","right"} and q["prompt"]==texts[m["pair_id"]][m["condition"]]
        assert {m["left_side"],m["right_side"]}=={"A","B"}
        for side in ("left","right"):
            with Image.open(blind/q[side]) as im:
                assert im.mode=="RGB" and hashlib.sha256(im.tobytes()).hexdigest()==expected[m["pair_id"],m[side+"_side"],m["light"]]
    outcomes,n=comparisons(blind/"annotations_blank.csv")
    assert n==0 and len(outcomes)==64 and all(r["result"]=="missing" for r in outcomes)
    report=(ROOT/"report.md").read_text(encoding="utf-8")
    for p in re.findall(r"\]\(([^)]+)\)",report):
        assert (ROOT/p).is_file(),p
    assert len(read_csv(ROOT/"cases.csv"))==6 and len({r["pair_id"] for r in read_csv(ROOT/"cases.csv")})==6
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for line in (ROOT/"logs/pilot-before.sha256").read_text().splitlines():
        checksum,path=line.split("  ",1)
        assert checksum==sha(ROOT/path),path
    print("PASS: 2 x 512 scores; IDs, A/B, conditions, margins, disjoint partitions, paired bootstrap, 64 checkpoints, 128 blinded image placements, 0 human answers, report links, and unchanged pilot inputs.")

if __name__=="__main__":
    main()

