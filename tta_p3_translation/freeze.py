import random
from collections import defaultdict
from datetime import datetime,timezone
import pyarrow.parquet as pq
from common import *

def main():
    pilot=file_json(PILOT/"manifest.json")
    meta=PILOT/"data/metadata-validation.parquet"
    assert digest(meta)==pilot["metadata_sha256"]
    old=read_csv(PILOT/"pairs.csv")
    exclude_ids={r["prompt_id"] for r in old}
    exclude_text={norm(r["P0"]) for r in old}
    exclude_img={Path(r["image_"+s]).name for r in old for s in ("A","B")}
    rows=pq.read_table(meta).to_pylist()
    assert len({Path(r["image_path"]).name for r in rows})==len(rows)
    groups=defaultdict(list)
    for r in rows:
        text=r["prompt"]
        if (r["prompt_id"] in exclude_ids or not norm(text) or norm(text) in exclude_text
            or Path(r["image_path"]).name in exclude_img or r["rank"] is None or r["rank"]<=0):
            continue
        assert r["image_path"].startswith("images/validation/")
        groups[norm(text),r["prompt_id"],text].append(r)
    pool=defaultdict(dict)
    for (nt,pid,text),rs in groups.items():
        pairs=[(a,b) for a,b in itertools.combinations(sorted(rs,key=lambda x:x["image_path"]),2) if a["rank"]!=b["rank"]]
        if pairs:pool[nt][pid,text]=pairs
    rng=random.Random(2026);selected=[]
    for k,nt in enumerate(rng.sample(sorted(pool),min(256,len(pool))),1):
        pid,text=rng.choice(sorted(pool[nt]))
        a,b=rng.choice(pool[nt][pid,text])
        if rng.getrandbits(1):a,b=b,a
        selected.append(dict(pair_id=f"N{k:03}",prompt_id=pid,P0=text,normalized_prompt=nt,
                   image_A=a["image_path"],image_B=b["image_path"],rank_A=a["rank"],rank_B=b["rank"],
                   human_winner_P0="A" if a["rank"]<b["rank"] else "B"))
    path=ROOT/"pairs_frozen.csv"
    if path.exists():
        assert read_csv(path)==[{k:str(v) for k,v in r.items()} for r in selected]
    else:write_csv(path,selected)
    if not (ROOT/"freeze.json").exists():
        write_json(ROOT/"freeze.json",dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),seed=2026,n=len(selected),
                   eligible_normalized=len(pool),metadata_rows=len(rows),excluded_old_ids=len(exclude_ids),
                   old_images=len(exclude_img),split="validation",revision=pilot["revision"],metadata_sha256=digest(meta),
                   pairs_sha256=digest(path),protocol_sha256=digest(ROOT/"protocol.md")))
    print(len(frozen()),file_json(ROOT/"freeze.json"))
if __name__=="__main__":main()

