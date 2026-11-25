"""Shared file formats and protocol constants. Run scripts from the project folder."""
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUFFIX = "The scene is at night, with dim lighting."
DATA_REV = "c493721e19e296eb615420036f2a2eed08412bb4"
HPS_REV = "697403c78157020a1ae59d23f111aa58ced35b0a"
CLIP_REV = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
CODE_REV = "866735ecaae999fa714bd9edfa05aa2672669ee3"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f.name, path)


def write_json(path, obj):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def load_pairs():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["pairs_sha256"] == digest(ROOT / "pairs.csv")
    pairs = read_csv(ROOT / "pairs.csv")
    assert len(pairs) == len({p["P0"] for p in pairs}) == len({p["pair_id"] for p in pairs}) == 64
    assert all(p["P1"] == p["P0"] + " " + SUFFIX and p["image_A"] != p["image_B"]
               and int(p["rank_A"]) != int(p["rank_B"])
               and p["human_winner_P0"] == ("A" if int(p["rank_A"]) < int(p["rank_B"]) else "B") for p in pairs)
    return pairs

