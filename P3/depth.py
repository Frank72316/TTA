"""Replicate the P3 budget curve on metric NYU Depth V2 with inverse translation."""

import argparse
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import random
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

from analyze_study import SEED, ci, draw_indices
from budget import BUDGETS, FULL_DIRECTIONS, digest, write_csv, write_json
from score_gpu import make_translation

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
DATA = "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth-png.zip"
VIEWS = ("O", *FULL_DIRECTIONS)


def read_image(archive, name):
    with Image.open(BytesIO(archive.read(name))) as image:
        return image.copy()


def measure(archive_path, out):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, __version__
    from score import select_gpu

    gpu = select_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=[
        "config.json", "preprocessor_config.json", "model.safetensors",
    ]))
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForDepthEstimation.from_pretrained(model_path).eval().cuda()
    if model.config.depth_estimation_type != "metric":
        raise ValueError("The continuous-ground-truth run requires a metric model")
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    with ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist()
                       if "/images/val/" in f"/{name}" and PurePosixPath(name).suffix.lower() in {".png", ".jpg", ".jpeg"})
        if len(names) != 654:
            raise ValueError(f"Expected 654 NYU test frames, found {len(names)}")
        order = random.Random(SEED).sample(names, 64)
        archive_names = set(archive.namelist())
        for index, name in enumerate(order):
            parts = PurePosixPath(name).parts
            marker = parts.index("images")
            depth_name = PurePosixPath(*parts[:marker], "depth", *parts[marker + 1:]).with_suffix(".png").as_posix()
            if depth_name not in archive_names:
                raise ValueError(f"Missing ground truth for {name}")
            image = read_image(archive, name).convert("RGB")
            truth = np.array(read_image(archive, depth_name), dtype=np.float32) / 1000
            if truth.shape != (image.height, image.width):
                raise ValueError(f"RGB and depth shape differ for {name}")

            predictions, offsets, inference_ms = [], [], {}
            started = perf_counter()
            for k, view in enumerate(VIEWS, start=1):
                augmented, params = make_translation(image, view, 3)
                dx, dy = (0, 0) if view == "O" else tuple(int(part.split("=")[1]) for part in params.split(";"))
                inputs = processor(images=augmented, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    result = model(**inputs).predicted_depth[:, None]
                    resized = torch.nn.functional.interpolate(
                        result, size=truth.shape, mode="bicubic", align_corners=False
                    )[0, 0].cpu().numpy()
                if not np.isfinite(resized).all():
                    raise ValueError(f"Nonfinite depth prediction for {name}, {view}")
                predictions.append(resized)
                offsets.append((dx, dy))
                if k in BUDGETS:
                    inference_ms[str(k)] = 1000 * (perf_counter() - started)
            edge_x = max(abs(dx) for dx, _ in offsets)
            edge_y = max(abs(dy) for _, dy in offsets)
            left, right = edge_x, image.width - edge_x
            top, bottom = edge_y, image.height - edge_y
            reference = truth[top:bottom, left:right]
            errors = np.array([
                predicted[top + dy:bottom + dy, left + dx:right + dx] - reference
                for predicted, (dx, dy) in zip(predictions, offsets)
            ])
            valid = (reference > 0) & (reference <= 10)
            regions = {}
            for region, included in {
                "all": valid, "near": valid & (reference <= 3), "far": valid & (reference > 3),
            }.items():
                residuals = errors[:, included].astype(np.float64)
                if residuals.shape[1]:
                    regions[region] = {
                        "pixels": residuals.shape[1], "bias": residuals.mean(axis=1).tolist(),
                        "gram": (residuals @ residuals.T / residuals.shape[1]).tolist(),
                    }
            samples.append({
                "frame": name, "subset": "development" if index < 32 else "holdout",
                "regions": regions, "prefix_inference_ms": inference_ms,
            })
            print(f"{index + 1}/64 {name}", flush=True)
    write_json(out / "depth_moments.json", samples)
    write_json(out / "depth_environment.json", {
        "gpu": gpu, "torch_build": torch.__version__, "transformers": __version__,
        "data_url": DATA, "data_sha256": digest(archive_path), "model": MODEL, "model_revision": REVISION,
        "model_sha256": {p.name: digest(p) for p in model_path.iterdir() if p.is_file()},
        "source_sha256": digest(Path(__file__)), "moments_sha256": digest(out / "depth_moments.json"),
        "float": "float32", "views": VIEWS, "seed": SEED,
        "timing_scope": "Cumulative image transforms, single images in float32, resize and device transfer; "
                        "excludes input I/O, alignment and statistics",
    })


def analyze_depth(out):
    moments = out / "depth_moments.json"
    meta = json.loads((out / "depth_environment.json").read_text())
    if digest(moments) != meta["moments_sha256"]:
        raise ValueError("Depth moments differ from the measured run")
    samples = json.loads(moments.read_text())
    if len(samples) != 64 or sum(s["subset"] == "holdout" for s in samples) != 32:
        raise ValueError("Incomplete depth sample")
    summary, fit, frames = [], [], []
    off = ~np.eye(8, dtype=bool)
    for region in ("all", "near", "far"):
        dev = [s["regions"][region] for s in samples if region in s["regions"] and s["subset"] == "development"]
        test = [s for s in samples if region in s["regions"] and s["subset"] == "holdout"]
        gram = np.mean([s["gram"] for s in dev], axis=0)
        means = np.mean([s["bias"] for s in dev], axis=0)
        cov = gram - np.outer(means, means)
        b = means[1:].mean()
        v = np.diag(cov)[1:].mean()
        rho = cov[1:, 1:][off].mean() / v if v else 0.0
        fit.append({
            "region": region, "development_frames": len(dev), "holdout_frames": len(test),
            "translation_bias_m": b, "single_view_error_variance_m2": v,
            "translation_error_mean_correlation": rho, "asymptotic_floor_m2": b*b + v*rho,
        })
        for sample in samples:
            if region in sample["regions"]:
                m = sample["regions"][region]
                cv = np.array(m["gram"]) - np.outer(m["bias"], m["bias"])
                var = np.diag(cv)[1:].mean()
                frames.append({
                    "frame": sample["frame"], "subset": sample["subset"], "region": region,
                    "pixels": m["pixels"], "translation_bias_m": np.mean(m["bias"][1:]),
                    "translation_variance_m2": var,
                    "translation_mean_correlation": cv[1:, 1:][off].mean() / var if var > 0 else "",
                })

        observed = np.array([s["regions"][region]["gram"] for s in test])
        indices = draw_indices(len(test))
        for anchored in (False, True):
            previous = None
            for k in BUDGETS:
                j = k - int(anchored)
                subsets = [((0, *s) if anchored else s) for s in combinations(range(1, 9), j)]
                loss = np.mean([observed[:, s, :][:, :, s].mean(axis=(1, 2)) for s in subsets], axis=0)
                lo, hi = ci(loss, indices)
                if anchored:
                    prediction = (gram[0, 0] + 2*j*gram[0, 1:].mean() + j*np.diag(gram)[1:].mean()
                                  + j*(j-1)*gram[1:, 1:][off].mean()) / k**2
                else:
                    prediction = b*b + v*(rho + (1-rho)/k)
                denom = 1 + (j-1)*rho
                delta = loss - previous if previous is not None else np.zeros(len(loss))
                delta_lo, delta_hi = ci(delta, indices)
                summary.append({
                    "region": region, "with_original": int(anchored), "budget_per_image": k,
                    "heldout_mse_m2": loss.mean(), "heldout_ci_low": lo, "heldout_ci_high": hi,
                    "predicted_mse_m2": prediction, "effective_translation_views": j / denom if j and denom > 0 else "",
                    "paired_delta_from_previous_m2": delta.mean() if previous is not None else "",
                    "paired_delta_ci_low_m2": delta_lo if previous is not None else "",
                    "paired_delta_ci_high_m2": delta_hi if previous is not None else "",
                    "anchored_prefix_inference_p50_ms": np.median([s["prefix_inference_ms"][str(k)] for s in test])
                    if anchored else "",
                })
                previous = loss
    write_csv(out / "depth_budget.csv", summary)
    write_csv(out / "depth_fit.csv", fit)
    write_csv(out / "depth_frame_error.csv", frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Run model inference before analyzing saved moments")
    args = parser.parse_args()
    if args.archive:
        measure(args.archive, args.out)
    analyze_depth(args.out)
