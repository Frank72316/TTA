import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20261128
FACTORS = (90, 95, 105, 110)
VIEWS = ["O"] + [f"{family}{factor:03d}" for family in "BC" for factor in FACTORS]
ARCHIVES = {"test_1.zip": "73737ce62b3d1aa33a96f788471c263596591c1b749510e686b8bedc7f285e55",
            "test_2.zip": "30f6f38de041b2df4d438ff5452c31fd219f6937508bacc41e92ca81ec284731"}

def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write(path, rows):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def text_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
