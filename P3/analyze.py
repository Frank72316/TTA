"""Compute exact direction-subset outcomes and paired bootstrap budget gains."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from budget import (
    BUDGETS, CARDINAL, FULL_DIRECTIONS, REPLICATION, ROOT, digest, load_pairs,
    load_scores, read_csv, write_csv, write_json,
)

SEED = 2026
RESAMPLES = 10_000


def analyze(pairs, scores, directions, budgets):
    views = ("O", *directions)
    margins = np.array([
        [float(scores[p["pair_id"], "A", view]["score"])
         - float(scores[p["pair_id"], "B", view]["score"]) for view in views]
        for p in pairs
    ])
    human_side = np.array([1 if p["human_winner_P0"] == "A" else -1 for p in pairs])
    original = np.sign(margins[:, 0]) * human_side
    subsets = {k: list(combinations(range(1, len(views)), k - 1)) for k in budgets}
    accuracy = {}
    pair_rows = []
    summary = []
    indices = np.random.default_rng(SEED).integers(0, len(pairs), (RESAMPLES, len(pairs)))
    for k, sets in subsets.items():
        augmented = np.array([margins[:, (0, *s)].sum(axis=1) for s in sets]).T
        choice = np.sign(augmented) * human_side[:, None]
        counts = {
            "correct": (choice > 0).sum(axis=1),
            "wrong": (choice < 0).sum(axis=1),
            "tie": (choice == 0).sum(axis=1),
            "corrected": ((original[:, None] < 0) & (choice > 0)).sum(axis=1),
            "harmed": ((original[:, None] > 0) & (choice < 0)).sum(axis=1),
            "correct_to_tie": ((original[:, None] > 0) & (choice == 0)).sum(axis=1),
            "tie_to_correct": ((original[:, None] == 0) & (choice > 0)).sum(axis=1),
        }
        assert np.array_equal(
            counts["correct"] - (original > 0) * len(sets),
            counts["corrected"] + counts["tie_to_correct"] - counts["harmed"] - counts["correct_to_tie"],
        )
        accuracy[k] = counts["correct"] / len(sets)
        accuracy_pct = 100 * accuracy[k].mean()
        interval = np.percentile(100 * accuracy[k][indices].mean(axis=1), [2.5, 97.5])
        summary.append({
            "budget_per_image": k,
            "combinations_per_pair": len(sets),
            "pairs": len(pairs),
            "accuracy_pct": accuracy_pct,
            "accuracy_ci_low_pct": interval[0],
            "accuracy_ci_high_pct": interval[1],
            **{f"expected_{name}_pairs": values.sum() / len(sets) for name, values in counts.items()},
        })
        for i, pair in enumerate(pairs):
            pair_rows.append({
                "pair_id": pair["pair_id"], "budget_per_image": k,
                "original_margin_a_minus_b": margins[i, 0], "combinations": len(sets),
                **{f"{name}_cells": values[i] for name, values in counts.items()},
            })

    marginal = []
    for k in budgets[1:]:
        previous = max(reference for reference in BUDGETS if reference < k)
        for reference in dict.fromkeys((previous, 1)):
            difference = 100 * (accuracy[k] - accuracy[reference])
            interval = np.percentile(difference[indices].mean(axis=1), [2.5, 97.5])
            marginal.append({
                "from_budget": reference, "to_budget": k,
                "gain_pp": difference.mean(),
                "ci_low_pp": interval[0], "ci_high_pp": interval[1],
                "gain_pp_per_additional_score_per_image": difference.mean() / (k - reference),
            })
    return summary, marginal, pair_rows


def summarize_timings(path, pairs):
    expected = {(p["pair_id"], k) for p in pairs for k in BUDGETS}
    values = {}
    for row in read_csv(path):
        key = row["pair_id"], int(row["budget_per_image"])
        elapsed = float(row["pair_wall_ms"])
        if key not in expected or key in values or not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Unexpected timing cell {key}")
        values[key] = elapsed
    if set(values) != expected:
        raise ValueError("Timing data does not cover every pair at every budget")
    for p in pairs:
        if not all(values[p["pair_id"], lo] < values[p["pair_id"], hi]
                   for lo, hi in zip(BUDGETS, BUDGETS[1:])):
            raise ValueError(f"Non-increasing cumulative timings for {p['pair_id']}")
    result = {}
    for k in BUDGETS:
        elapsed = [values[p["pair_id"], k] for p in pairs]
        result[k] = {
            "pair_wall_mean_ms": np.mean(elapsed),
            "pair_wall_p50_ms": np.median(elapsed),
            "pair_wall_p95_ms": np.percentile(elapsed, 95),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=REPLICATION / "pairs_frozen.csv")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--timings", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.timings and (not args.scores or not args.environment):
        parser.error("--timings requires fresh --scores and --environment")
    if args.environment and not args.scores:
        parser.error("--environment requires fresh --scores")
    pairs = load_pairs(args.pairs)
    score_path = args.scores or REPLICATION / "scores.csv"
    input_hashes = {
        "pairs_sha256": digest(args.pairs), "scores_sha256": digest(score_path),
        "timings_sha256": digest(args.timings) if args.timings else None,
    }
    if args.environment:
        environment = json.loads(args.environment.read_text())
        detected_gpu = environment["gpu"]["name"]
        if not detected_gpu or detected_gpu != environment.get("expected_gpu_name", detected_gpu):
            raise ValueError("The detected GPU differs from the planned GPU")
        for key, value in input_hashes.items():
            if value is not None and environment[key] != value:
                raise ValueError(f"Input does not match the GPU environment record: {key}")
    directions = FULL_DIRECTIONS if args.scores else CARDINAL
    budgets = BUDGETS if args.scores else (1, 2, 3, 4, 5)
    scores = load_scores(score_path, pairs, directions)
    summary, marginal, pair_rows = analyze(pairs, scores, directions, budgets)
    if args.timings:
        timing = summarize_timings(args.timings, pairs)
        for row in summary:
            row.update(timing[row["budget_per_image"]])
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "budget_summary.csv", summary)
    write_csv(args.out / "marginal_gains.csv", marginal)
    write_csv(args.out / "pair_results.csv", pair_rows)
    write_json(args.out / "analysis.json", {
        "mode": "eight_direction_fresh_scores" if args.scores else "p2_cache_pilot",
        **input_hashes,
        "environment_sha256": digest(args.environment) if args.environment else None,
        "bootstrap_seed": SEED, "bootstrap_resamples": RESAMPLES,
        "views": ["O", *directions], "budgets_per_image": budgets,
    })
    for row in summary:
        print(f"K={row['budget_per_image']}: {row['accuracy_pct']:.4f}%")


if __name__ == "__main__":
    main()
