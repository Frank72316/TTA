"""Rebuild the published contrasts and their cluster intervals from the per-bin cells."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results"


def rows(name):
    with (ROOT/name).open(encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    bins = rows("per_bin.csv")
    scenes = rows("per_scene.csv")
    published = rows("summary.csv")
    assert len({(r["year"],r["id"],r["scale"],r["axis"],r["bin"]) for r in bins}) == len(bins) == 990
    assert len({(r["year"],r["id"],r["scale"]) for r in scenes}) == len(scenes) == 100
    checked = {}
    for row in scenes:
        key = tuple(row[k] for k in ("year","id","scale"))
        part = [r for r in bins if tuple(r[k] for k in ("year","id","scale")) == key]
        counts = np.array([int(r["edges"]) for r in part])
        assert counts.sum() == int(row["edges"])
        def average(k):
            return sum(n*float(r[k]) for n,r in zip(counts,part))/counts.sum()
        hh, ll, lh, hl, mean = [average(k) for k in ("HH","LL","LH","HL","mean_only")]
        d = hh-ll
        s = (hh-hl+lh-ll)/2
        metrics = dict(HH=hh, LL=ll, LH=lh, HL=hl, mean_only=mean, D=d, S=s, V=d-s,
                       interaction=(lh-ll)-(hh-hl), mean_gain=hh-mean,
                       rank_with_H_values=hh-hl, rank_with_L_values=lh-ll, value_on_H_order=hh-lh,
                       value_on_L_order=hl-ll, shape_gain_on_H_order=mean-lh, mean_to_LL=mean-ll)
        for k,x in metrics.items():
            assert abs(float(row[k])-x) < 1e-8
        checked[key] = metrics
    rng = np.random.default_rng(2026)
    for batch,years in (("2005-2006",("2005","2006")),("2014",("2014",))):
        cohort = [r for r in scenes if r["year"] in years]
        labels = sorted({r["cluster"] for r in cohort})
        assert len(labels) == (17 if batch == "2005-2006" else 23)
        draws = np.array([rng.choice(len(labels),size=len(labels),replace=True) for _ in range(10000)])
        frequencies = np.array([np.bincount(i,minlength=len(labels)) for i in draws]) / len(labels)
        for row in [r for r in published if r["batch"] == batch]:
            k,mode = row["metric"],row["scale"]
            means = np.array([np.mean([checked[r["year"],r["id"],mode][k] for r in cohort
                                       if r["cluster"] == label and r["scale"] == mode]) for label in labels])
            lo,hi = np.percentile(frequencies @ means,[2.5,97.5])
            np.testing.assert_allclose([means.mean(),lo,hi],[float(row[f]) for f in ("mean","lo","hi")],
                                       atol=1e-8, rtol=0)
    print(f"Validated {len(bins)} stratum rows, {len(scenes)} paired scene rows, "
          f"and {len(published)} paired estimates with both interval endpoints.")


if __name__ == "__main__":
    main()
