import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42
BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATION_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path, data):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    temp.replace(path)


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())
    temp.replace(path)


def valid_rect(w, h):
    scale = 224 / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    # The installed ResizeMaxSize skips both resize and pad when scale == 1.
    if scale == 1 and w != h:
        return None
    return (int((224-nw)//2), int((224-nh)//2), nw, nh)


def choose(x):
    return "A" if x > 0 else "B" if x < 0 else "="


def verified_freeze(out):
    manifest = json.loads((out / "freeze.json").read_bytes())
    for name in ("pairs.csv", "images.csv"):
        assert sha((out / name).read_bytes()) == manifest["hashes"][name], name
    return manifest


def legacy_environment(meta):
    """The two audited runner revisions differ only in preview/checkpoint handling."""
    if (meta["flip_inference_sha"] == "0c5d4654c68b78e3402cbf80e38344d10ce6ebdbae63c17702805eb3090c4b0b"
            and meta["freeze_sha"] in ("1de355c4e590b7fc28205f65d5f4c59fa8962989b84f5b00bff71b4626b6670a",
                                       "2399fa9b99969744474d7650b3829399838801fb2d39592c75901f73bcb8ebf0")):
        return dict(meta, flip_inference_sha="69fc6fdffa0345b8e93e726d56ebe6cbd2cd6c2602f22bf9909a9d77cfd7ba53",
                    freeze_sha="2399fa9b99969744474d7650b3829399838801fb2d39592c75901f73bcb8ebf0")
    return None
