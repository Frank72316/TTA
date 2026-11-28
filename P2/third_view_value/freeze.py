"""Build the official test list using metadata and history only."""
import argparse
import itertools
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from common_io import HERE, ROOT, SEED, digest, read, save, text_hash, write

def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())

def identifiers(rows):
    return (set(r["prompt_id"] for r in rows), set(norm(r["P0"]) for r in rows),
            {Path(r["image_"+s]).stem for r in rows for s in "AB"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata", type=Path)
    ap.add_argument("--history", nargs=3, type=Path, default=[
        ROOT/"P1/baseline/pairs.csv", ROOT/"P2/translation/pairs_frozen.csv", ROOT/"P3/results/holdout/pairs.csv"])
    args = ap.parse_args()
    assert digest(args.metadata) == "318b071eb7a53565138ccc215d6cfeaf513b31ee7c03bda89efb590b2b7cd072"
    histories = [read(p) for p in args.history]
    assert list(map(len, histories)) == [64,256,88]
    old_ids, old_text, old_images = identifiers(sum(histories, []))
    raw = pq.read_table(args.metadata).to_pylist()
    groups = defaultdict(list)
    for r in raw:
        assert r["image_path"].startswith("images/test/")
        key = norm(r["prompt"])
        if (r["rank"] and r["rank"] > 0 and key and key not in old_text and r["prompt_id"] not in old_ids
                and Path(r["image_path"]).stem not in old_images):
            groups[key, r["prompt_id"]].append(r)
    eligible = defaultdict(dict)
    for (key, pid), rows in groups.items():
        assert len({r["prompt"] for r in rows}) == 1
        choices = [(a,b) for a,b in itertools.combinations(sorted(rows,key=lambda r:r["image_path"]),2)
                   if a["rank"] != b["rank"] and a["image_path"] != b["image_path"]]
        if choices:
            eligible[key][pid] = choices
    rng = random.Random(SEED)
    pairs = []
    for i, key in enumerate(rng.sample(sorted(eligible),min(400,len(eligible))),1):
        pid = rng.choice(sorted(eligible[key]))
        a,b = rng.choice(eligible[key][pid])
        if rng.getrandbits(1):
            a,b = b,a
        pairs.append(dict(pair_id=f"T{i:03}",prompt_id=pid,P0=a["prompt"],normalized_prompt=key,
                          image_A=a["image_path"],image_B=b["image_path"],rank_A=a["rank"],rank_B=b["rank"],
                          human_winner_P0="A" if a["rank"] < b["rank"] else "B"))
    new = identifiers(pairs)
    assert list(map(len,new)) == [len(pairs),len(pairs),2*len(pairs)]
    checks = []
    for old,p in zip(histories,args.history):
        overlaps = [len(x & y) for x,y in zip(new,identifiers(old))]
        assert overlaps == [0,0,0]
        checks.append(dict(history_n=len(old),sha256_lf=text_hash(p),prompt_id=0,normalized_text=0,image_uuid=0))
    dst = HERE/"pairs_frozen.csv"
    if dst.exists():
        assert read(dst) == [{k:str(v) for k,v in row.items()} for row in pairs]
    else:
        write(dst,pairs)
    obj = dict(revision="c493721e19e296eb615420036f2a2eed08412bb4", split="test", seed=SEED,
               metadata_sha256=digest(args.metadata),official_rows=len(raw),raw_prompt_ids=len({r["prompt_id"] for r in raw}),
               raw_normalized=len({norm(r["prompt"]) for r in raw}),eligible_normalized=len(eligible),
               samples=len(pairs),protocol_sha256_lf=text_hash(HERE/"protocol.md"),pairs_sha256_lf=text_hash(dst),
               historical_overlaps=checks)
    save(HERE/"freeze.json",obj)
    print(obj)

if __name__ == "__main__":
    main()
