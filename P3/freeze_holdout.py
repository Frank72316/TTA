"""Freeze all eligible validation prompts unused by P1 and P2."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import unicodedata

from budget import PILOT, REPLICATION, SIDES, digest, read_csv, write_csv, write_json

SEED = 30126


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def run(out):
    import pyarrow.parquet as pq

    previous = (PILOT / "pairs.csv", REPLICATION / "pairs_frozen.csv")
    excluded = [row for path in previous for row in read_csv(path)]
    ids = {r["prompt_id"] for r in excluded}
    texts = {normalize(r["P0"]) for r in excluded}
    images = {Path(r[f"image_{side}"]).name for r in excluded for side in SIDES}
    metadata = PILOT / "data" / "metadata-validation.parquet"
    manifest = json.loads((PILOT / "manifest.json").read_text())
    if digest(metadata) != manifest["metadata_sha256"]:
        raise ValueError("Unexpected ImageRewardDB metadata revision")

    groups = defaultdict(list)
    for row in pq.read_table(metadata).to_pylist():
        text = row["prompt"]
        key = normalize(text), row["prompt_id"], text
        if (not key[0] or key[0] in texts or key[1] in ids or not row["rank"]
                or row["rank"] < 0 or Path(row["image_path"]).name in images):
            continue
        groups[key].append(row)

    pool = defaultdict(list)
    for key, records in sorted(groups.items()):
        eligible = [
            (a, b) for a, b in combinations(sorted(records, key=lambda r: r["image_path"]), 2)
            if a["rank"] != b["rank"]
        ]
        if eligible:
            pool[key[0]].append((key, eligible))

    rng = random.Random(SEED)
    rows = []
    for index, normalized in enumerate(sorted(pool), start=1):
        (_, prompt_id, text), eligible = rng.choice(pool[normalized])
        a, b = rng.choice(eligible)
        if rng.getrandbits(1):
            a, b = b, a
        rows.append({
            "pair_id": f"H{index:03}", "prompt_id": prompt_id, "P0": text,
            "normalized_prompt": normalized, "image_A": a["image_path"], "image_B": b["image_path"],
            "rank_A": a["rank"], "rank_B": b["rank"],
            "human_winner_P0": "A" if a["rank"] < b["rank"] else "B",
        })
    if not rows or len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError("Invalid holdout IDs")
    if len({r[f"image_{side}"] for r in rows for side in SIDES}) != 2 * len(rows):
        raise ValueError("Repeated holdout image")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pairs.csv", rows)
    write_json(out / "freeze.json", {
        "pairs": len(rows), "seed": SEED, "dataset_revision": manifest["revision"],
        "metadata_sha256": digest(metadata), "excluded_pair_files_sha256": [digest(path) for path in previous],
        "pairs_sha256": digest(out / "pairs.csv"), "freeze_source_sha256": digest(Path(__file__)),
    })
    print(f"Frozen {len(rows)} unused prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
