"""Freeze the random sample before inference. No model imports or image scores."""
import itertools
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

from pilot_io import DATA, DATA_REV, ROOT, SUFFIX, digest, load_pairs, write_csv, write_json


def prepare():
    table = pq.read_table(DATA / "metadata-validation.parquet")
    rows = table.to_pylist()
    groups = defaultdict(list)
    for row in rows:
        if row["rank"] is not None and row["rank"] > 0:
            assert row["image_path"].startswith("images/validation/")
            groups[(row["prompt"], row["prompt_id"])].append(row)
    eligible = defaultdict(dict)
    for (text, group), records in groups.items():
        paths = sorted(records, key=lambda r: r["image_path"])
        assert len(paths) == len({r["image_path"] for r in paths})
        choices = [(a,b) for a,b in itertools.combinations(paths, 2) if a["rank"] != b["rank"]]
        if choices and text.strip():
            eligible[text][group] = choices
    rng = random.Random(42)
    selected = []
    for i, text in enumerate(rng.sample(sorted(eligible), 64), 1):
        group = rng.choice(sorted(eligible[text]))
        a, b = rng.choice(eligible[text][group])
        if rng.getrandbits(1):
            a, b = b, a
        selected.append(dict(pair_id=f"{i:03}", prompt_id=group, P0=text, P1=text+" "+SUFFIX,
                             image_A=a["image_path"], image_B=b["image_path"],
                             rank_A=a["rank"], rank_B=b["rank"],
                             human_winner_P0="A" if a["rank"] < b["rank"] else "B"))
    path = ROOT / "pairs.csv"
    if path.exists():
        from pilot_io import read_csv
        if read_csv(path) != [{k:str(v) for k,v in r.items()} for r in selected]:
            raise RuntimeError("Existing pairs.csv differs from the seeded draw")
    else:
        write_csv(path, selected)
    manifest = dict(dataset="zai-org/ImageRewardDB", revision=DATA_REV, split="validation",
                    metadata_sha256=digest(DATA/"metadata-validation.parquet"),
                    fields=table.column_names, image_rows=len(rows), prompt_ids=len(groups),
                    distinct_texts=len({r["prompt"] for r in rows}), eligible_texts=len(eligible),
                    rank_evidence="Official ImageRewardDB.py, _generate_examples: rank[idx] < rank[idy] gives img_better=idx",
                    seed=42, pairs_sha256=digest(path))
    write_json(ROOT/"manifest.json", manifest)
    load_pairs()
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__":
    prepare()

