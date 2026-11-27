"""Score the full P3 view bank and benchmark nested budgets on one NVIDIA GPU."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
from time import perf_counter

from budget import (
    BUDGETS, DIAGONAL, FULL_DIRECTIONS, P2, PILOT, REPLICATION, SIDES,
    digest, load_pairs, write_csv, write_json,
)
from score import WEIGHTS, select_gpu

sys.path.insert(0, str(P2))
AXES = {"Tleft": (-1, 0), "Tright": (1, 0), "Tup": (0, -1), "Tdown": (0, 1)}
BRIGHTNESS_VIEWS = ("B095", "B105")


def make_translation(rgb, name, amplitude_pct):
    from views import make_view, translate

    if name == "O" or name in BRIGHTNESS_VIEWS or (amplitude_pct == 3 and name in AXES):
        return make_view(rgb, name)
    if name in DIAGONAL:
        sign_x, sign_y = DIAGONAL[name]
        amplitude_pct /= 2**0.5
    else:
        sign_x, sign_y = AXES[name]
    x, y = (int(n * amplitude_pct / 100 + 0.5) for n in rgb.size)
    dx, dy = sign_x * x, sign_y * y
    return translate(rgb, dx, dy), f"dx={dx};dy={dy}"


def score_pair(pair, model, tokenizer, transform, torch, args):
    from views import original

    directions = list(FULL_DIRECTIONS)
    random.Random(f"{args.seed}:{pair['pair_id']}").shuffle(directions)
    views = ("O", *directions, *(BRIGHTNESS_VIEWS if args.with_brightness else ()))
    start = perf_counter()
    originals = {side: original(pair, side) for side in SIDES}
    tokens = tokenizer([pair["P0"]])
    scores, timings, images, tensors = [], [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        text = model.encode_text(tokens.to("cuda"), normalize=True)
        for k, view in enumerate(views, start=1):
            records, predictions = [], []
            for side in SIDES:
                image, parameter = make_translation(originals[side], view, args.amplitude_pct)
                tensor = transform(image)
                if tensor.shape != (3, 224, 224) or image.size != originals[side].size:
                    raise ValueError(f"Invalid input shape for {pair['pair_id']}, {side}, {view}")
                feature = model.encode_image(tensor.unsqueeze(0).to("cuda"), normalize=True)
                predictions.append((feature @ text.T).reshape(1))
                images.append(image)
                tensors.append(tensor)
                records.append({
                    "pair_id": pair["pair_id"], "prompt_id": pair["prompt_id"], "P0": pair["P0"],
                    "side": side, "image_id": pair[f"image_{side}"], "view": view, "param": parameter,
                    "native_dtype": "float16",
                })
            values = torch.cat(predictions).cpu()
            if values.dtype != torch.float16 or not torch.isfinite(values).all():
                raise ValueError(f"Invalid HPS output for {pair['pair_id']}, {view}")
            for row, value in zip(records, values.tolist()):
                scores.append({**row, "score": value})
            if k in BUDGETS:
                timings.append({
                    "pair_id": pair["pair_id"], "budget_per_image": k,
                    "pair_wall_ms": (perf_counter() - start) * 1000,
                    "views": " ".join(("O", *directions[:k - 1])),
                })
    token_hash = sha256(tokens.numpy().astype("<i8").tobytes()).hexdigest()
    input_hashes = {side: {} for side in SIDES}
    for row, image, tensor in zip(scores, images, tensors):
        tensor_hash = sha256(tensor.numpy().tobytes()).hexdigest()
        seen = input_hashes[row["side"]]
        row.update(token_sha256=token_hash, input_pixels_sha=sha256(image.tobytes()).hexdigest(),
                   model_input_sha=tensor_hash, same_model_input_as=seen.get(tensor_hash, ""))
        seen.setdefault(tensor_hash, row["view"])
    return scores, timings, images


def run(args):
    pairs = load_pairs(args.pairs)
    replication_env = json.loads((REPLICATION / "environment.json").read_text())
    baseline_env = json.loads((PILOT / "environment.json").read_text())
    for name, expected in baseline_env["validation_archives_sha256"].items():
        if digest(PILOT / "data" / name) != expected:
            raise ValueError(f"Validation archive checksum differs for {name}")
    gpu = select_gpu()
    if gpu["name"] != args.gpu_name:
        raise RuntimeError(f"Expected {args.gpu_name}, found {gpu['name']}")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_CACHE=str(PILOT / "data" / "model-cache"),
        HPS_ROOT=str(args.out / "hps"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    import torch
    from hpsv2 import img_score
    from hpsv2.src.open_clip import get_tokenizer
    from huggingface_hub import hf_hub_download

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA device")
    for repo, rev, name, expected in WEIGHTS:
        path = Path(hf_hub_download(repo, name, revision=rev, local_files_only=True))
        main = hf_hub_download(repo, name, local_files_only=True)
        if digest(path) != expected or not path.samefile(main):
            raise ValueError(f"Weight checksum or upstream reference differs for {name}")
        if repo == "xswu/HPSv2":
            checkpoint = path

    img_score.initialize_model()
    model = img_score.model_dict["model"]
    transform = img_score.model_dict["preprocess_val"]
    transform_repr = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(transform))
    if transform_repr != replication_env["preprocess"]:
        raise ValueError("HPS did not initialize the P2 validation preprocessing")
    if args.scorer == "hps":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        model.load_state_dict(state, strict=True)
        del state
    model.eval()
    tokenizer = get_tokenizer("ViT-H-14")
    started = datetime.now(timezone.utc).isoformat()

    warmup, _, images = score_pair(pairs[0], model, tokenizer, transform, torch, args)
    with torch.inference_mode():
        if args.scorer == "hps":
            official = img_score.score(images, pairs[0]["P0"], cp=str(checkpoint), hps_version="v2.1")
        else:
            tokens = tokenizer([pairs[0]["P0"]]).to("cuda")
            official = []
            with torch.autocast("cuda", dtype=torch.float16):
                for image in images:
                    output = model(transform(image).unsqueeze(0).to("cuda"), tokens)
                    official.append((output["image_features"] @ output["text_features"].T).item())
    if [row["score"] for row in warmup] != [float(value) for value in official]:
        raise ValueError(f"Warmup scores differ from the {args.scorer} reference scorer")
    verified = len(official)
    del images, official, warmup

    rows, timings = [], []
    for pair in pairs:
        scores, elapsed, _ = score_pair(pair, model, tokenizer, transform, torch, args)
        rows.extend(scores)
        timings.extend(elapsed)
        print(f"{pair['pair_id']}: {len(rows)} scores", flush=True)

    write_csv(args.out / "scores.csv", rows)
    write_csv(args.out / "timings.csv", timings)
    write_json(args.out / "environment.json", {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "instance_template": args.instance_template, "zone": args.zone,
        "platform": platform.platform(), "linux": platform.freedesktop_os_release().get("PRETTY_NAME"),
        "python": platform.python_version(), "gpu": gpu, "expected_gpu_name": args.gpu_name,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "torch_build": torch.__version__,
        "software": {name: version(name) for name in ("hpsv2", "torchvision", "Pillow", "numpy", "huggingface-hub")},
        "scorer": args.scorer,
        "model": "HPS v2.1 ViT-H-14" if args.scorer == "hps" else "CLIP ViT-H-14 LAION-2B",
        "amplitude_pct": args.amplitude_pct, "with_brightness": args.with_brightness,
        "amp": "CUDA float16",
        "deterministic_algorithms": True, "tf32_matmul": False,
        "pinned_weights": WEIGHTS, "preprocess": transform_repr,
        "dataset_revision": baseline_env["dataset_revision"],
        "validation_archives_sha256": baseline_env["validation_archives_sha256"],
        "pairs_sha256": digest(args.pairs), "scores_sha256": digest(args.out / "scores.csv"),
        "timings_sha256": digest(args.out / "timings.csv"),
        "scorer_source_sha256": digest(Path(img_score.__file__)),
        "code_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("budget.py"), P2 / "views.py")},
        "score_count": len(rows), "warmup_reference_scores": verified,
        "repeated_input_count": sum(bool(row["same_model_input_as"]) for row in rows),
        "timing_seed": args.seed,
        "timing_scope": "Cumulative wall latency per pair, one warmed model and text encoding per pair; "
                        "includes original ZIP reads, image transforms, feature inference, and score transfer; "
                        "excludes warmup, input hashing, and output writes",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=REPLICATION / "pairs_frozen.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--instance-template")
    parser.add_argument("--gpu-name", default="NVIDIA L4", help="Exact device name reported by nvidia-smi")
    parser.add_argument("--amplitude-pct", type=float, default=3, choices=(1.5, 3, 6))
    parser.add_argument("--scorer", choices=("hps", "clip"), default="hps")
    parser.add_argument("--with-brightness", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args)
