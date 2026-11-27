"""Run official HPS scoring on one idle GPU, checkpointing whole pairs."""
import argparse
import fcntl
import io
import itertools
import json
import math
import os
import platform
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pilot_io import (CLIP_REV, CODE_REV, DATA, DATA_REV, HPS_REV, ROOT, digest,
                      load_pairs, read_csv, write_csv, write_json)

CELLS = list(itertools.product(("A","B"), ("L0","L1")))
WEIGHTS = [
    ("xswu/HPSv2", HPS_REV, "HPS_v2.1_compressed.pt", "c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837"),
    ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", CLIP_REV, "open_clip_pytorch_model.bin", "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"),
]


def select_gpu():
    def query(flag, fields):
        return subprocess.check_output(["nvidia-smi", f"--query-{flag}={fields}",
                                        "--format=csv,noheader,nounits"], text=True).strip()
    busy = set(query("compute-apps","gpu_uuid").splitlines())
    for line in query("gpu","index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version").splitlines():
        idx, uid, name, used, total, util, driver = [s.strip() for s in line.split(",")]
        if uid not in busy and int(used) < 256 and int(util) == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = uid
            return dict(index=int(idx), uuid=uid, name=name, initial_MiB=int(used),
                        total_MiB=int(total), initial_util=int(util), driver=driver)
    raise RuntimeError("No idle GPU found")


def verify_scores(rows, pairs):
    expected = {(p["pair_id"], a, prompt, light):p["image_"+a] for p in pairs
                for a,light in CELLS for prompt in ("P0","P1")}
    actual = {(r["pair_id"],r["side"],r["prompt_condition"],r["light"]):r["image_id"] for r in rows}
    assert len(rows) == len(actual) and actual == expected
    assert all(math.isfinite(float(r["score"])) for r in rows)


def run(limit):
    pairs = load_pairs()
    scorepath = ROOT / "scores.csv"
    rows = read_csv(scorepath) if scorepath.exists() else []
    done = {r["pair_id"] for r in rows}
    verify_scores(rows, [p for p in pairs if p["pair_id"] in done])
    pending = [p for p in pairs[:limit] if p["pair_id"] not in done]
    if not pending:
        return
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).isoformat()
    gpu = select_gpu()
    print("GPU", gpu, flush=True)
    os.environ["HPS_ROOT"] = str(ROOT / "data" / "hps")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import hpsv2
    import torch
    from hpsv2.src.open_clip.tokenizer import _tokenizer, tokenize
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageEnhance

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 1 and torch.cuda.is_available()
    for repo, rev, filename, sha in WEIGHTS:
        pinned = Path(hf_hub_download(repo, filename, revision=rev, local_files_only=True))
        default = Path(hf_hub_download(repo, filename, local_files_only=True))
        assert pinned.samefile(default) and digest(pinned) == sha, filename
    versions = {p:version(p) for p in ("hpsv2","torch","torchvision","pillow","numpy","pyarrow",
                                        "huggingface-hub","transformers","timm","ftfy","regex","pandas",
                                        "einops","clint","webdataset","fsspec","tqdm","requests","braceexpand",
                                        "sentencepiece","protobuf")}
    archive_hashes = {"validation_1.zip":"8eb57656d6c424b9451240d5f401462107889f2c4781fb3e5f827f7e73a197ea",
                      "validation_2.zip":"5349f894b1b1571fbe1aed6a3288bdf9f8e4dbf2d8a6fbe00274477f276da48c"}
    for name, checksum in archive_hashes.items():
        assert digest(DATA/name) == checksum, name
    protocol = dict(pairs_sha256=digest(ROOT/"pairs.csv"), score_script_sha256=digest(Path(__file__)),
                    software=versions, python=platform.python_version(), platform=platform.platform(),
                    host=platform.node(), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu_model=gpu["name"], driver=gpu["driver"], model="HPS v2.1",
                    weights=WEIGHTS, code_revision=CODE_REV, dataset_revision=DATA_REV,
                    validation_archives_sha256=archive_hashes, brightness=1.1,
                    inference="official hpsv2.score; eval, preprocess_val, CUDA autocast, native dot product")
    envfile = ROOT/"environment.json"
    if envfile.exists():
        assert json.loads(envfile.read_text()) == json.loads(json.dumps(protocol)), "Protocol changed"
    elif rows:
        raise RuntimeError("scores.csv lacks provenance")
    else:
        write_json(envfile, protocol)

    for p in pending:
        tick = time.monotonic()
        images = []
        for side in ("A","B"):
            path = Path(p["image_"+side])
            with zipfile.ZipFile(DATA/(path.parent.name+".zip")) as archive:
                with Image.open(io.BytesIO(archive.read(path.name))) as source:
                    rgb = source.convert("RGB")
            images.extend([rgb, ImageEnhance.Brightness(rgb).enhance(1.1)])
        extra = []
        same_tokens = torch.equal(tokenize(p["P0"]), tokenize(p["P1"]))
        for cond in ("P0","P1"):
            count = len(_tokenizer.encode(p[cond])) + 2
            with torch.inference_mode():
                values = hpsv2.score(images, p[cond], hps_version="v2.1")
            assert len(values) == len(CELLS)
            for (side,light),value in zip(CELLS, values):
                extra.append(dict(pair_id=p["pair_id"], side=side, prompt_condition=cond, light=light,
                                  image_id=p["image_"+side], score=float(value), native_dtype=str(value.dtype),
                                  tokens_before_truncation=count, same_P0_P1_tokens=int(same_tokens)))
        verify_scores(extra,[p])
        rows.extend(extra)
        write_csv(scorepath, rows)
        print(f"pair {p['pair_id']} committed, {len(rows)} scores, {time.monotonic()-tick:.2f}s",flush=True)
    verify_scores(rows,[p for p in pairs if p["pair_id"] in {r["pair_id"] for r in rows}])
    runfile = ROOT/"runs.json"
    runs = json.loads(runfile.read_text()) if runfile.exists() else []
    runs.append(dict(limit=limit, started_utc=stamp, ended_utc=datetime.now(timezone.utc).isoformat(),
                     seconds=time.monotonic()-started, gpu=gpu, new_pairs=len(pending), cumulative_scores=len(rows)))
    write_json(runfile, runs)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--limit", type=int, choices=[2,64], required=True)
    args = cli.parse_args()
    with open(ROOT/".score.lock","a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.limit)

