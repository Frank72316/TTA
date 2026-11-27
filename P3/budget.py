"""Inputs and view definitions shared by the P3 scoring and analysis commands."""

import math
import sys
from pathlib import Path
from struct import pack, unpack

ROOT = Path(__file__).resolve().parent
P2 = ROOT.parent / "P2"
PILOT = ROOT.parent / "P1" / "baseline"
REPLICATION = P2 / "translation"
sys.path.insert(0, str(PILOT))
from pilot_io import digest, read_csv, write_csv, write_json

CARDINAL = ("Tleft", "Tright", "Tup", "Tdown")
DIAGONAL = {
    "Tup_left": (-1, -1),
    "Tup_right": (1, -1),
    "Tdown_left": (-1, 1),
    "Tdown_right": (1, 1),
}
FULL_DIRECTIONS = (*CARDINAL, *DIAGONAL)
BUDGETS = (1, 2, 4, 8)
SIDES = ("A", "B")


def load_pairs(path):
    pairs = read_csv(path)
    if not pairs:
        raise ValueError(f"No pairs in {path}")
    for field in ("pair_id", "prompt_id", "P0"):
        if len({p[field] for p in pairs}) != len(pairs):
            raise ValueError(f"Repeated {field} in {path}")
    images = [p[f"image_{side}"] for p in pairs for side in SIDES]
    if len(set(images)) != len(images):
        raise ValueError(f"Repeated candidate image in {path}")
    for pair in pairs:
        a, b = int(pair["rank_A"]), int(pair["rank_B"])
        if min(a, b) <= 0 or a == b or pair["human_winner_P0"] != ("A" if a < b else "B"):
            raise ValueError(f"Invalid human ranking for {pair['pair_id']}")
    return pairs


def load_scores(path, pairs, directions):
    pair_map = {p["pair_id"]: p for p in pairs}
    views = ("O", *directions)
    scores = {}
    for row in read_csv(path):
        view = row["view"]
        if view not in views:
            continue
        pair_id, side = row["pair_id"], row["side"]
        key = (pair_id, side, view)
        pair = pair_map.get(pair_id)
        if (key in scores or pair is None or side not in SIDES or row["P0"] != pair["P0"]
                or row["prompt_id"] != pair["prompt_id"] or row["image_id"] != pair[f"image_{side}"]):
            raise ValueError(f"Duplicate or inconsistent score cell {key}")
        score = float(row["score"])
        if row["native_dtype"] != "float16" or not math.isfinite(score):
            raise ValueError(f"Invalid HPS score at {key}")
        try:
            if unpack("<e", pack("<e", score))[0] != score:
                raise ValueError(f"Score is not an exact float16 value at {key}")
        except OverflowError as exc:
            raise ValueError(f"Score is outside the float16 range at {key}") from exc
        scores[key] = row
    for pair_id in pair_map:
        for side in SIDES:
            for view in views:
                if (pair_id, side, view) not in scores:
                    raise ValueError(f"Missing score cell {(pair_id, side, view)}")
    return scores
