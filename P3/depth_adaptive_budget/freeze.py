import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "results/depth"
SEED = 20261128


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_bytes(rows):
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue().encode("utf8")


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen file differs: {path}")
    else:
        path.write_bytes(data)


def freeze(archive):
    meta = json.loads((OLD / "depth_environment.json").read_text())
    assert sha(archive) == meta["data_sha256"]
    moments = OLD / "depth_moments.json"
    assert sha(moments) == meta["moments_sha256"]
    old = {r["frame"] for r in json.loads(moments.read_text())}
    with ZipFile(archive) as z:
        names = set(z.namelist())
        images = {n for n in names if "/images/val/" in n and PurePosixPath(n).suffix.lower() in {".png",".jpg",".jpeg"}}
        assert len(images) == 654 and len(old) == 64 and old <= images
        selection = random.Random(SEED).sample(sorted(images - old), 128)
        rows = []
        for n in selection:
            gt = str(PurePosixPath(n.replace("/images/", "/depth/")).with_suffix(".png"))
            assert gt in names
            rows.append(dict(frame_id=PurePosixPath(n).stem, frame=n, truth=gt))
    immutable(HERE / "frames.csv", csv_bytes(rows))
    immutable(HERE / "freeze.json", json.dumps(dict(seed=SEED,split_frames=len(images),
        excluded=len(old),remaining=len(images-old),sampled=len(rows),overlap=len(old & set(selection)),
        frames_sha256=sha(HERE/"frames.csv"),archive_sha256=sha(archive),old_moments_sha256=sha(moments)),
        indent=2).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    freeze(ap.parse_args().archive)
