"""Attribute changes in jump MAE to signed correction values and their placement."""
import argparse
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DETAIL = HERE.parent / "depth_jump_detail_intervention"
sys.path.insert(0, str(DETAIL))
from edge_data import intervals, jump, pinned_fields, predict_map
from shared import cache, get, ident, included, load_freeze, local_cache, read, save, sha, write

OUT = HERE / "results"
MODES = ("native_R", "low_core")
CELLS = ("HH", "LH", "HL", "LL", "mean_only")
TOL_MM = 1e-8


def read_scene(record):
    key = record["year"], record["id"]
    def prior_rows(name):
        return [r for r in read(DETAIL / "results" / name) if (r["year"], r["id"]) == key]

    b, native, info, near, far, gt, jb, old_u, groups = pinned_fields(record)
    index, = prior_rows("cache_index.csv")
    assert sha(cache(record) / "inference.json") == index["historical_inference_sha256"]
    folder = local_cache(record)
    assert sha(folder / "low.npy") == index["low_prediction_sha256"] == get(folder / "score.json")["low_prediction_sha256"]
    low = np.load(folder / "low.npy", allow_pickle=False)
    assert low.shape == native.shape and low.dtype == native.dtype == np.float32
    corners = sorted(prior_rows("input_windows.csv"), key=lambda v: int(v["corner"]))
    assert [int(w["corner"]) for w in corners] == list(range(4))
    scales = {m: [float(w[k]) for w in corners] for m, k in zip(MODES, ("k_H", "k_L"))}
    np.testing.assert_array_equal(scales["native_R"], info["k"]["R"])
    boxes = [layout["box"] for layout in info["layouts"]]
    need = (gt - jb) * 1000
    baselines = {r["scale"]: r for r in prior_rows("per_scene.csv")}
    assert len(baselines) == 2 and np.isfinite(need).all()
    updates, replay = {}, []
    for mode in MODES:
        h, l = [(jump(predict_map(b, views, boxes, scales[mode]), near, far) - jb) * 1000
                for views in (native, low)]
        assert np.isfinite(h).all() and np.isfinite(l).all()
        if mode == "native_R":
            np.testing.assert_array_equal(h, old_u)
        updates[mode] = h, l
        hh, ll = (float(np.abs(need - u).mean()) for u in (h, l))
        old = baselines[mode]
        assert record["cluster"] == old["cluster"] and len(need) == int(old["edges"]) and len(groups) == int(old["bins"])
        error = max(abs(hh - float(old["H_actual_mm"])), abs(ll - float(old["L_actual_mm"])),
                    abs((hh - ll) + float(old["G_mm"])))
        assert error < TOL_MM
        replay.append(dict(**ident(record), scale=mode, edges=len(need), bins=len(groups),
                           HH=hh, LL=ll, D=hh-ll, replay_error_mm=error))
    return ident(record), need, groups, updates, replay


def assigned_mae(required, values, ranks):
    """Average absolute loss of tied receivers over every admissible donor value."""
    order = np.argsort(ranks, kind="stable")
    sorted_ranks, donors, target = ranks[order], np.sort(values), required[order]
    losses = np.abs(target - donors)
    boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_ranks)), len(order)]
    ties = np.flatnonzero(np.diff(boundaries) > 1)
    for block in ties:
        lo, hi = boundaries[block:block+2]
        candidates = donors[lo:hi]
        for i in range(lo, hi, 128):
            j = min(i + 128, hi)
            losses[i:j] = np.abs(target[i:j, None] - candidates).mean(axis=1)
    return float(losses.mean()), len(ties), int(np.diff(boundaries)[ties].sum())


def check_assignment(required, values, ranks):
    """Independent prefix-sum check; use rank intervals for each edge, without tie ordering."""
    sorted_ranks, donors = np.sort(ranks), np.sort(values)
    lo = sorted_ranks.searchsorted(ranks, side="left")
    hi = sorted_ranks.searchsorted(ranks, side="right")
    prefix = np.r_[0., np.cumsum(donors, dtype=np.float64)]
    k = np.clip(donors.searchsorted(required), lo, hi)
    abs_sum = required * ((k-lo)-(hi-k)) + prefix[lo] + prefix[hi] - 2*prefix[k]
    return float(np.mean(abs_sum / (hi-lo)))


def contrasts(c):
    # First letter indexes the donor pool, second letter indexes receiver order.
    hh, lh, hl, ll, shifted = (c[k] for k in CELLS)
    v_h, s_l = hh-lh, lh-ll
    s_h, v_l = hh-hl, hl-ll
    return dict(D=hh-ll, V=(v_h+v_l)/2, S=(s_h+s_l)/2,
                value_on_H_order=v_h, rank_with_L_values=s_l,
                rank_with_H_values=s_h, value_on_L_order=v_l,
                interaction=v_l-v_h, mean_gain=hh-shifted,
                shape_gain_on_H_order=shifted-lh, mean_to_LL=shifted-ll)


def evaluate_bin(need, h, l):
    lh, h_ties, h_tied_edges = assigned_mae(need, l, h)
    hl, l_ties, l_tied_edges = assigned_mae(need, h, l)
    c = dict(HH=np.abs(need-h).mean(), LH=lh, HL=hl, LL=np.abs(need-l).mean(),
             mean_only=np.abs(need-(h-h.mean()+l.mean())).mean())
    check = max(abs(c[a+b]-check_assignment(need, u, v))
                for a, u in (("H", h), ("L", l)) for b, v in (("H", h), ("L", l)))
    d = contrasts(c)
    assert max(check, abs(d["D"]-d["V"]-d["S"])) < TOL_MM
    return dict(**c, **d, H_mean_mm=h.mean(), L_mean_mm=l.mean(), H_tie_blocks=h_ties,
                L_tie_blocks=l_ties, H_tied_edges=h_tied_edges, L_tied_edges=l_tied_edges,
                assignment_check_mm=check)


def main(workers):
    start = time.perf_counter()
    load_freeze()
    for line in (HERE/"sources.sha256").read_text().splitlines():
        digest, name = line.split()
        assert sha(HERE/name) == digest, name
    with ProcessPoolExecutor(workers) as pool:
        data = list(pool.map(read_scene, included()))
    prior = [r for *_, replay in data for r in replay]
    assert len(data) == 50 and sum(len(s[1]) for s in data) == 243818 and sum(len(s[2]) for s in data) == 495
    history = {(x["batch"], x["scale"]): x for x in read(DETAIL/"results/summary.csv") if x["metric"] == "G_mm"}
    replay_stats = intervals(prior, ["D"])
    for row in replay_stats:
        old = history[row["batch"], row["scale"]]
        assert (row["clusters"], row["scenes"]) == (int(old["clusters"]), int(old["scenes"]))
        np.testing.assert_allclose([row["mean"], row["lo"], row["hi"]],
                                   [-float(old[k]) for k in ("mean","hi","lo")], atol=TOL_MM, rtol=0)
        print("Baseline replay:", row["batch"], row["scale"], row["mean"], flush=True)
    write(OUT/"native_replay.csv", replay_stats)
    bins, scenes = [], []
    for key, need, groups, updates, replay in data:
        for mode, old in zip(MODES, replay):
            h, l = updates[mode]
            part = [dict(**key, scale=mode, axis=a, bin=b, edges=len(ix), **evaluate_bin(need[ix], h[ix], l[ix]))
                    for a, b, ix in groups]
            weights = [r["edges"] for r in part]
            cells = {k: float(np.average([r[k] for r in part], weights=weights)) for k in CELLS}
            d = contrasts(cells)
            error = max(abs(cells[k]-old[k]) for k in ("HH","LL"))
            assert max(error, abs(d["D"]-sum([d["V"],d["S"]]))) < TOL_MM
            scene = dict(**key, scale=mode, edges=len(need), bins=len(part), **cells, **d)
            for k in ("H_tie_blocks","L_tie_blocks","H_tied_edges","L_tied_edges"):
                scene[k] = sum(r[k] for r in part)
            scenes.append(scene)
            bins.extend(part)
    fields = list(CELLS) + list(contrasts(dict.fromkeys(CELLS, 0.)))
    summary = intervals(scenes, fields)
    four = []
    for batch in ("2005-2006","2014"):
        for mode in MODES:
            group = {r["metric"]: r for r in summary if r["scale"] == mode and r["batch"] == batch}
            for val in "HL":
                for rank in "HL":
                    r = group[val+rank]
                    four.append(dict(batch=batch, scale=mode, value_pool=val, order=rank,
                                     MAE_mm=r["mean"], lo_mm=r["lo"], hi_mm=r["hi"]))
    for name, rows in (("four_cells", four), ("summary", summary), ("per_scene", scenes), ("per_bin", bins)):
        write(OUT/(name+".csv"), rows)
    save(OUT/"checks.json", dict(host=platform.node(), python=platform.python_version(), numpy=np.__version__,
         scene_count=len(data), edge_count=sum(len(s[1]) for s in data), bins_per_scale=len(bins)//2, model_forwards=0,
         max_previous_MAE_error_mm=max(s["replay_error_mm"] for s in prior),
         max_independent_assignment_error_mm=max(r["assignment_check_mm"] for r in bins),
         max_scene_D_minus_V_S_mm=max(abs(r["D"]-r["V"]-r["S"]) for r in scenes),
         bootstrap_seed=2026, paired_cluster_draws=10000, CPU_wall_seconds=time.perf_counter()-start))
    for r in summary:
        if r["metric"] in ("D","V","S","mean_gain","value_on_H_order","interaction"):
            print(r["batch"],r["scale"],r["metric"],"%.5f [%.5f, %.5f]" % (r["mean"],r["lo"],r["hi"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args().workers)
