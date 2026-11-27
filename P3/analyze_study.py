"""Compare matched amplitudes, predict the Brier budget curve, and test holdout policies."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analyze import RESAMPLES, SEED
from budget import (
    BUDGETS, CARDINAL, FULL_DIRECTIONS, REPLICATION, digest, load_pairs,
    load_scores, read_csv, write_csv, write_json,
)

BRIGHTNESS = ("B095", "B105")


def load_run(root, name, pairs_path, amplitude, brightness, scorer="hps"):
    path = root / name
    env = json.loads((path / "environment.json").read_text())
    expected = {
        "pairs_sha256": digest(pairs_path), "scores_sha256": digest(path / "scores.csv"),
        "timings_sha256": digest(path / "timings.csv"), "amplitude_pct": amplitude,
        "scorer": scorer, "with_brightness": brightness,
    }
    if any(env[key] != value for key, value in expected.items()):
        raise ValueError(f"Unexpected experiment inputs for {name}")
    pairs = load_pairs(pairs_path)
    views = (*FULL_DIRECTIONS, *(BRIGHTNESS if brightness else ()))
    scores = load_scores(path / "scores.csv", pairs, views)
    margins = np.array([
        [float(scores[p["pair_id"], "A", view]["score"]) - float(scores[p["pair_id"], "B", view]["score"])
         for view in ("O", *views)]
        for p in pairs
    ])
    human = np.array([p["human_winner_P0"] == "A" for p in pairs])
    return margins, human, env


def draw_indices(n):
    return np.random.default_rng(SEED).integers(0, n, (RESAMPLES, n))


def ci(values, indices):
    return np.percentile(values[indices].mean(axis=1), [2.5, 97.5]).tolist()


def compare_amplitudes(root, runs, out):
    accuracy = {}
    rows = []
    subsets = list(combinations(range(1, 9), 3))
    for pct, (margins, human, _) in runs.items():
        if not np.array_equal(margins[:, 0], runs[3][0][:, 0]):
            raise ValueError(f"Original HPS margins changed in the {pct:g}% amplitude run")
        aligned = (2 * human - 1)[:, None] * margins
        results = np.array([aligned[:, (0, *dirs)].sum(axis=1) > 0 for dirs in subsets])
        accuracy[pct] = 100 * results.mean(axis=0)
    indices = draw_indices(len(accuracy[3]))
    for pct, values in sorted(accuracy.items()):
        name = f"development-{pct:g}"
        base = next(row for row in read_csv(root / name / "budget_summary.csv") if row["budget_per_image"] == "4")
        if not np.isclose(float(base["accuracy_pct"]), values.mean()):
            raise ValueError(f"Amplitude scores and budget analysis disagree for {name}")
        change = values - accuracy[3]
        lo, hi = ci(change, indices)
        rows.append({
            **base, "amplitude_pct": pct, "delta_vs_3_pp": change.mean(),
            "delta_ci_low_pp": lo, "delta_ci_high_pp": hi,
        })
    write_csv(out / "amplitude_summary.csv", rows)


def compare_legacy(root, pairs_path, score_path, out):
    pairs = load_pairs(pairs_path)
    old = load_scores(score_path, pairs, CARDINAL)
    fresh = load_scores(root / "development-3" / "scores.csv", pairs, CARDINAL)
    fields = ("param", "token_sha256", "input_pixels_sha")
    if any(old[key][field] != fresh[key][field] for key in old for field in fields):
        raise ValueError("Legacy and L4 inputs differ in the shared views")
    views = ("O", *CARDINAL)
    matrices = [
        np.array([
            [float(bank[p["pair_id"], "A", v]["score"]) - float(bank[p["pair_id"], "B", v]["score"])
             for v in views]
            for p in pairs
        ])
        for bank in (old, fresh)
    ]
    write_json(out / "legacy_overlap.json", {
        "pairs": len(pairs), "identical_params_tokens_and_rgb_hashes": len(old),
        "identical_fp16_scores": sum(float(old[key]["score"]) == float(fresh[key]["score"]) for key in old),
        "max_absolute_score_delta": max(abs(float(old[key]["score"]) - float(fresh[key]["score"])) for key in old),
        "legacy_scores_sha256": digest(score_path),
    })
    rows = []
    for k in (1, 2, 4):
        subsets = [(0, *s) for s in combinations(range(1, 5), k - 1)]
        left, right = [np.sign([matrix[:, s].sum(axis=1) for s in subsets]) for matrix in matrices]
        delta = left != right
        rows.append({
            "budget_per_image": k, "evaluated_cells": delta.size, "different_cells": delta.sum(),
            "affected_pairs": delta.any(axis=0).sum(),
        })
    write_csv(out / "legacy_overlap.csv", rows)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit_slope(margins, human):
    scale = np.sqrt(np.mean(margins**2))
    if scale == 0:
        return 0.0
    x = margins / scale
    low, high = 0.0, 100.0
    for _ in range(60):
        mid = (low + high) / 2
        if np.mean(x * (sigmoid(mid * x) - human)) < 0:
            low = mid
        else:
            high = mid
    return float((low + high) / (2 * scale))


def brier_curve(development, holdout, out):
    margins, human, _ = development
    val_margins, val_human, _ = holdout
    slope = fit_slope(margins[:, 0], human)
    dev = sigmoid(slope * margins) - human[:, None]
    val = sigmoid(slope * val_margins) - val_human[:, None]
    pure = dev[:, 1:9]
    covariance = np.cov(pure, rowvar=False, bias=True)
    bias = pure.mean()
    variance = np.trace(covariance) / 8
    off_diag = ~np.eye(8, dtype=bool)
    rho = covariance[off_diag].mean() / variance if variance else 0.0
    # Keep the original view as a separate term in the anchored curve.
    gram = dev[:, :9].T @ dev[:, :9] / len(dev)
    indices = draw_indices(len(val))
    rows = []
    for anchored in (False, True):
        for k in BUDGETS:
            j = k - int(anchored)
            sets = list(combinations(range(1, 9), j))
            val_loss = np.mean([
                val[:, ((0, *subset) if anchored else subset)].mean(axis=1)**2 for subset in sets
            ], axis=0)
            lower, upper = ci(val_loss, indices)
            if anchored:
                predicted = (
                    gram[0, 0] + 2 * j * gram[0, 1:].mean() + j * np.diag(gram)[1:].mean()
                    + j * (j - 1) * gram[1:, 1:][off_diag].mean()
                ) / k**2
            else:
                predicted = bias**2 + variance * (rho + (1 - rho) / k)
            eff = 1 + (j - 1) * rho
            rows.append({
                "with_original": int(anchored), "budget_per_image": k,
                "predicted_development_mse": predicted, "measured_holdout_mse": val_loss.mean(),
                "holdout_ci_low": lower, "holdout_ci_high": upper,
                "effective_translation_views": j / eff if j and eff > 0 else "",
            })
    write_csv(out / "brier_budget.csv", rows)
    write_json(out / "brier_fit.json", {
        "probability_calibration_slope": slope, "translation_residual_bias": float(bias),
        "translation_residual_variance": float(variance), "translation_residual_mean_correlation": float(rho),
        "translation_equicorrelation_floor": float(bias**2 + variance * rho),
    })
    return slope


def two_view_outcomes(margins, human):
    signs = (2 * human - 1)[:, None]
    original = signs[:, 0] * margins[:, 0]
    augmented = signs * (margins[:, :1] + margins[:, 1:])
    return original, augmented[:, :8], augmented[:, 8:]


def policies(development, holdout, slope, out):
    dev, dev_human, _ = development
    val, human, _ = holdout
    threshold = np.quantile(np.abs(dev[:, 0]), 0.25)
    dev_trigger = np.abs(dev[:, 0]) <= threshold
    split = np.median(np.abs(dev[dev_trigger, 0]))
    dev_low = np.abs(dev[:, 0]) <= split
    p = sigmoid(slope * dev)
    error = p[:, 0] - dev_human
    move = (p[:, 1:] - p[:, :1]) / 2
    delta = 2 * error[:, None] * move + move**2
    dev_rows = []
    for segment, mask in (("low", dev_low), ("upper", dev_trigger & ~dev_low)):
        # An empty upper segment can result when both thresholds hit the same fp16 tie.
        groups = delta[mask] if mask.any() else delta[dev_trigger]
        t, b = groups[:, :8].mean(), groups[:, 8:].mean()
        dev_rows.append({
            "segment": segment, "pairs": int(mask.sum()), "mean_delta_brier_translation": t,
            "mean_delta_brier_brightness": b, "chosen_family": "B" if b < t else "T",
        })
    write_csv(out / "policy_fit.csv", dev_rows)

    d0, trans, bright = two_view_outcomes(val, human)
    trigger = np.abs(val[:, 0]) <= threshold
    low = np.abs(val[:, 0]) <= split
    brightness_selected = np.where(low, dev_rows[0]["chosen_family"] == "B",
                                   dev_rows[1]["chosen_family"] == "B")
    share = trigger.mean()
    base_ok = (d0 > 0).astype(float)
    base_tie = (d0 == 0).astype(float)
    fixed_t = (trans > 0).mean(axis=1)
    fixed_b = (bright > 0).mean(axis=1)
    tie_t = (trans == 0).mean(axis=1)
    tie_b = (bright == 0).mean(axis=1)
    oracle_select = np.flatnonzero((d0 <= 0) & np.any(np.column_stack((trans, bright)) > 0, axis=1))
    oracle_ok = base_ok.copy()
    oracle_tie = base_tie.copy()
    oracle_selected = oracle_select[:int(trigger.sum())]
    oracle_ok[oracle_selected] = 1
    oracle_tie[oracle_selected] = 0
    policies = {
        "fixed_1": (1, base_ok, base_tie),
        "fixed_2_translation": (2, fixed_t, tie_t),
        "fixed_2_brightness": (2, fixed_b, tie_b),
        "random_extra_translation": (1 + share, (1 - share)*base_ok + share*fixed_t,
                                     (1 - share)*base_tie + share*tie_t),
        "trigger_translation": (1 + share, np.where(trigger, fixed_t, base_ok),
                                np.where(trigger, tie_t, base_tie)),
        "trigger_brightness": (1 + share, np.where(trigger, fixed_b, base_ok),
                               np.where(trigger, tie_b, base_tie)),
        "trigger_learned_family": (1 + share, np.where(trigger, np.where(brightness_selected, fixed_b, fixed_t), base_ok),
                                   np.where(trigger, np.where(brightness_selected, tie_b, tie_t), base_tie)),
        "oracle_upper": (1 + share, oracle_ok, oracle_tie),
    }
    reference = policies["random_extra_translation"][1]
    indices = draw_indices(len(val))
    rows = []
    for name, (cost, correct, tie) in policies.items():
        delta_pp = 100 * (correct - reference)
        lo, hi = ci(delta_pp, indices) if name != "oracle_upper" else ("", "")
        rows.append({
            "policy": name, "mean_scores_per_image": cost, "accuracy_pct": 100 * correct.mean(),
            "tie_pct": 100 * tie.mean(), "delta_vs_same_cost_random_pp": delta_pp.mean(),
            "delta_ci_low_pp": lo, "delta_ci_high_pp": hi,
        })
    write_csv(out / "policy_summary.csv", rows)
    write_json(out / "policy.json", {
        "threshold": float(threshold), "threshold_internal_split": float(split),
        "holdout_pairs": len(val), "holdout_triggered": int(trigger.sum()),
        "holdout_oracle_improved": len(oracle_selected),
    })


def main(root, out, legacy_scores):
    pairs = REPLICATION / "pairs_frozen.csv"
    frozen_holdout = root / "holdout" / "pairs.csv"
    freeze = json.loads((root / "holdout" / "freeze.json").read_text())
    if digest(frozen_holdout) != freeze["pairs_sha256"]:
        raise ValueError("Holdout freeze checksum differs")
    dev = {pct: load_run(root, f"development-{pct:g}", pairs, pct, pct == 3) for pct in (1.5, 3, 6)}
    hps = load_run(root, "holdout-hps", frozen_holdout, 3, True)
    clip = load_run(root, "holdout-clip", frozen_holdout, 3, False, "clip")
    devices = [run[2]["gpu"] for run in (*dev.values(), hps, clip)]
    if len({(d["uuid"], d["name"], d["driver"]) for d in devices}) != 1:
        raise ValueError("The compared runs were not on the same GPU")
    out.mkdir(parents=True, exist_ok=True)
    compare_amplitudes(root, dev, out)
    slope = brier_curve(dev[3], hps, out)
    policies(dev[3], hps, slope, out)
    if legacy_scores:
        compare_legacy(root, pairs, legacy_scores, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-scores", type=Path)
    options = parser.parse_args()
    main(options.runs, options.out, options.legacy_scores)
