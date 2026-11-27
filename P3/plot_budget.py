"""Plot paired preference gains, L4 latency, and the metric-depth budget curve."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from budget import read_csv

BUDGETS = [1, 2, 4, 8]


def main(runs, depth, out):
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 10, "axes.titlesize": 12, "figure.dpi": 160,
        "font.family": "DejaVu Sans",
    })
    fig, (quality, time, metric) = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for directory, label, color, dx in (
        ("development-3", "Development", "#186459", -0.055),
        ("holdout-hps", "Held-out prompts", "#92502c", 0.055),
    ):
        rows = [r for r in read_csv(runs / directory / "marginal_gains.csv") if r["from_budget"] == "1"]
        gain = np.array([0, *(float(r["gain_pp"]) for r in rows)])
        low = np.array([0, *(float(r["ci_low_pp"]) for r in rows)])
        high = np.array([0, *(float(r["ci_high_pp"]) for r in rows)])
        x = np.arange(4) + dx
        quality.plot(x, gain, "o-", lw=1.4, color=color, label=label)
        quality.vlines(x, low, high, lw=1.2, color=color, alpha=0.75)
    quality.axhline(0, color="#6d7476", lw=0.8, linestyle="--")
    quality.set(title="Preference accuracy gain", ylabel="vs original only (percentage points)")
    quality.legend(loc="lower left", frameon=False, fontsize=8)

    timings = read_csv(runs / "development-3" / "budget_summary.csv")
    for key, label, color in (
        ("pair_wall_p50_ms", "Median", "#186459"), ("pair_wall_p95_ms", "95th percentile", "#92502c")
    ):
        time.plot(range(4), [float(r[key]) for r in timings], "o-", color=color, label=label)
    time.set(title="End-to-end latency", ylabel="Milliseconds per pair")
    time.legend(frameon=False, fontsize=8)

    rows = [r for r in read_csv(depth / "depth_budget.csv") if r["region"] == "all" and r["with_original"] == "1"]
    observed = [float(r["heldout_mse_m2"]) for r in rows]
    predicted = [float(r["predicted_mse_m2"]) for r in rows]
    metric.plot(range(4), observed, "o-", color="#186459", label="Held-out frames")
    metric.plot(range(4), predicted, "s--", color="#92502c", label="Development prediction")
    metric.set(title="Metric depth error", ylabel="Mean squared error (m²)")
    metric.legend(frameon=False, fontsize=8)
    for axes in (quality, time, metric):
        axes.set_xticks(range(4), BUDGETS)
        axes.set_xlabel("Scores per image")
        axes.grid(axis="y", linewidth=0.6, color="#dbe3e2")
        axes.set_axisbelow(True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.runs, args.depth, args.out)
