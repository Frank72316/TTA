"""Full Middlebury cached-map audit for Frank72316/TTA at 9467465 (CPU only).

Use the Python environment in P3/depth_occlusion_context/README.md.
python edge_cache.py --context-dir /path/to/TTA/P3/depth_occlusion_context --out /path/to/new/results
python edge_cache.py --self-test
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np


METHODS = ("B", "T", "R", "UP", "DOWN", "SHIFT", "ROLL")
TRANSITIONS = (
    "no_change", "started_at_truth", "away_same_side", "toward_no_cross",
    "cross_gain", "cross_tie", "cross_loss",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def derived_maps(base, tiled):
    # Operate on the entire prediction raster. No GT, edge mask or depth-range mask here.
    plus = np.maximum(tiled - base, 0)
    h, w = base.shape
    yield "UP", np.maximum(tiled, base)
    yield "DOWN", np.minimum(tiled, base)
    yield "SHIFT", base + np.mean(plus, dtype=np.float64)
    yield "ROLL", base + np.roll(plus, shift=(h // 2, w // 2), axis=(0, 1))


def score(pred, gt, valid, near_pts, far_pts, near_unique, far_unique, jump):
    if pred.shape != gt.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid prediction raster")
    pf = pred.ravel()
    zf = gt.ravel()
    err_near = (pf[near_unique].astype(np.float64) - zf[near_unique]) * 1000
    err_far = (pf[far_unique].astype(np.float64) - zf[far_unique]) * 1000
    estimate = np.median(pf[far_pts], axis=1).astype(np.float64) - np.median(pf[near_pts], axis=1)
    residual = (estimate - jump) * 1000
    near_mae = np.abs(err_near).mean()
    far_mae = np.abs(err_far).mean()
    metrics = dict(near_mae_mm=near_mae, far_mae_mm=far_mae, equal_mae_mm=(near_mae + far_mae) / 2,
                   global_mae_mm=np.abs(pred[valid].astype(np.float64) - gt[valid]).mean() * 1000,
                   jump_mae_mm=np.abs(residual).mean(), jump_signed_mm=residual.mean(),
                   near_signed_mm=err_near.mean(), far_signed_mm=err_far.mean())
    return metrics, residual


def transition_stats(before, after):
    """Classes partition baseline-to-new transitions; contribution weights use ALL edges."""
    step = after - before
    gain = np.abs(before) - np.abs(after)
    crossed = ((before < 0) & (after > 0)) | ((before > 0) & (after < 0))
    noncross = (before != 0) & (step != 0) & ~crossed
    groups = (step == 0, (before == 0) & (step != 0), noncross & (gain < 0),
              noncross & (gain > 0), crossed & (gain > 0), crossed & (gain == 0),
              crossed & (gain < 0))
    if not np.all(np.sum(groups, axis=0) == 1):
        raise ValueError("Transition classes overlap or fail to cover edges")
    rows = [dict(transition=label, n_edges=len(before), count=int(mask.sum()),
                 fraction=mask.mean(), contribution_mm=gain[mask].sum() / len(before))
            for label, mask in zip(TRANSITIONS, groups)]
    np.testing.assert_allclose(sum(r["contribution_mm"] for r in rows), gain.mean(), atol=1e-8)
    return rows


def displacement_audit(before, after):
    """Exact expected L1 error after uniformly permuting updates among this scene's edges."""
    update = after - before
    n = len(update)
    order = np.sort(update)
    prefix = np.r_[0., np.cumsum(order, dtype=np.float64)]
    cut = np.searchsorted(order, -before, side="left")
    # Of the n sums e+u, cut are negative. Negate their subtotal.
    expectation_at_edge = ((n - 2 * cut) * before + prefix[-1] - 2 * prefix[cut]) / n
    perm = expectation_at_edge.mean()
    constant = np.abs(before + update.mean()).mean()
    actual = np.abs(after).mean()
    base = np.abs(before).mean()
    common_gain, spread_cost, place_gain = base - constant, perm - constant, perm - actual
    bound = base - abs(update.mean()) - actual
    if spread_cost < -1e-7 or place_gain < bound - 1e-7:
        raise ValueError("Jensen or triangle bound violated")
    np.testing.assert_allclose(common_gain - spread_cost + place_gain, base - actual, atol=1e-7)
    return dict(baseline_mae_mm=base, actual_mae_mm=actual,
                mean_update_mm=update.mean(), constant_mae_mm=constant, permuted_mae_mm=perm,
                permuted_minus_actual_mm=perm - actual, constant_minus_actual_mm=constant - actual,
                permuted_minus_constant_mm=spread_cost, common_gain_mm=common_gain,
                location_gain_lower_bound_mm=bound, actual_gain_mm=base - actual)


def self_test():
    before = np.array([0., 0., 3., -3., -3., 3., -3., -3.])
    after = np.array([0., 2., 5., -1., 2., -3., 5., 0.])
    rows = transition_stats(before, after)
    assert [r["count"] for r in rows] == [1, 1, 1, 2, 1, 1, 1]
    b = np.array([[1., 2., 3.], [2., 3., 4.]])
    t = b + np.array([[1., -1., 0.], [-2., 3., 0.]])
    views = dict(derived_maps(b, t))
    np.testing.assert_allclose(views["UP"] + views["DOWN"], b + t)
    updates = [np.mean(views[v] - b) for v in ("UP", "ROLL", "SHIFT")]
    np.testing.assert_allclose(updates, updates[0])
    metrics, residual = score(b, b, b > 0, np.array([[0, 1, 2]]), np.array([[3, 4, 5]]),
                              np.arange(3), np.arange(3, 6), np.array([1.]))
    assert all(value == 0 for value in metrics.values()) and not residual.any()
    audit = displacement_audit(before, after)
    brute = np.abs(before[:, None] + (after - before)[None, :]).mean()
    np.testing.assert_allclose(audit["permuted_mae_mm"], brute, atol=1e-10)
    np.testing.assert_allclose(displacement_audit(before, before + 3)["permuted_minus_actual_mm"], 0, atol=1e-10)
    print("Synthetic geometry, controls and exhaustive transition tests passed.")


def summarize(scenes, moves, audits, reps, seed):
    lookup = {(r["year"], r["id"], r["method"]): r for r in scenes}
    contrasts = [("B", m) for m in METHODS if m != "B"]
    contrasts.extend([("T", "UP"), ("SHIFT", "UP"), ("ROLL", "UP"), ("DOWN", "UP")])
    metrics = [k for k in scenes[0] if k.endswith("_mm")]
    groups = {}
    for r in scenes:
        if r["method"] == "B":
            batch = "2014" if r["year"] == "2014" else "2005-2006"
            groups.setdefault(batch, {}).setdefault(r["cluster"], []).append((r["year"], r["id"]))
    paired = []
    attrib = []
    audit_summary = []
    transitions = {(r["year"], r["id"], r["method"], r["transition"]): r for r in moves}
    audit_lookup = {(r["year"], r["id"], r["method"]): r for r in audits}
    rng = np.random.default_rng(seed)
    for batch, clusters in sorted(groups.items()):
        members = [clusters[name] for name in sorted(clusters)]
        indices = rng.integers(len(members), size=(reps, len(members)))
        def estimate(vector):
            a = np.asarray(vector)
            lo, hi = np.quantile(a[indices].mean(axis=1), [.025, .975])
            return dict(mean=a.mean(), lo=lo, hi=hi)
        info = dict(batch=batch, scenes=sum(map(len, members)), clusters=len(members))
        for method in ("T", "R"):
            for metric in [k for k in audits[0] if k.endswith("_mm")]:
                vals = [mean(audit_lookup[y, i, method][metric] for y, i in group) for group in members]
                audit_summary.append(dict(**info, method=method, metric=metric, **estimate(vals)))
        for ref, method in contrasts:
            for metric in metrics:
                diff = [mean(lookup[y, i, ref][metric] - lookup[y, i, method][metric]
                             for y, i in group) for group in members]
                paired.append(dict(**info, reference=ref, method=method, metric=metric, **estimate(diff)))
        for method in ("T", "R", "UP"):
            for c in TRANSITIONS:
                for metric in ("fraction", "contribution_mm"):
                    vals = [mean(transitions[y, i, method, c][metric] for y, i in group)
                            for group in members]
                    attrib.append(dict(**info, method=method, transition=c, metric=metric, **estimate(vals)))
    return paired, attrib, audit_summary


def run(context, out, reps, seed):
    context = context.resolve()
    out = out.resolve()
    if out == context or (out.exists() and any(out.iterdir())):
        raise ValueError("Choose a separate empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(context))
    from analysis import maps
    from coverage import edge_path
    from data import cache, included, legacy_row, metric, old

    records = included()
    if Counter(r["year"] for r in records) != {"2014": 23, "2005": 6, "2006": 21}:
        raise ValueError("The frozen 50-scene manifest was not found")
    reference = {(r["year"], r["id"], r["method"]): r for r in csv_rows(context / "per_scene.csv")}
    index = {(r["year"], r["id"]): r for r in csv_rows(context / "cache_index.csv")}
    old_index = {r["id"]: r for r in csv_rows(context.parent / "depth_occlusion_tta" / "cache_index.csv")}
    fields = dict(near_mae_mm="all_near_mae", far_mae_mm="all_far_mae", equal_mae_mm="all_mae",
                  jump_mae_mm="all_jump_mae", jump_signed_mm="all_jump_signed", global_mae_mm="global_mae",
                  near_signed_mm="all_near_signed", far_signed_mm="all_far_signed")
    scenes, moves, audits = [], [], []
    all_edges = 0
    max_ref_diff = 0.
    for rec in records:
        ident = {k: rec[k] for k in ("year", "id", "cluster")}
        key = rec["year"], rec["id"]
        hashes = index[key]
        if (digest(cache(rec) / "predictions.npz") != hashes["prediction_sha256"] or
                digest(edge_path(rec)) != hashes["edge_sha256"]):
            raise ValueError(f"Cache hash mismatch: {key}")
        if rec["year"] == "2014":
            prior = old_index[rec["id"]]
            if digest(old.work(legacy_row(rec)) / "predictions.npz") != prior["raw_prediction_sha"]:
                raise ValueError(f"Shared 2014 map hash mismatch: {key}")
        z = metric(rec)
        valid = np.isfinite(z) & (z > 0) & (z <= 20)
        with np.load(edge_path(rec)) as e:
            near, far, jump = e["near"], e["far"], e["jump"]
        assert len(jump) > 0 and near.shape == far.shape == (len(jump), 3)
        all_edges += len(jump)
        points = (z, valid, near, far, np.unique(near), np.unique(far), jump)
        original = dict(maps(rec, ("B", "T", "R")))
        baseline = None
        def predictions():
            yield from original.items()
            yield from derived_maps(original["B"], original["T"])
        for name, pred in predictions():
            scores, residual = score(pred, *points)
            scenes.append(dict(**ident, method=name, n_edges=len(jump), **scores))
            if name in original:
                row = reference[key + (name,)]
                delta = max(abs(scores[m] - 1000 * float(row[k])) for m, k in fields.items())
                max_ref_diff = max(max_ref_diff, delta)
                if delta > 0.02:
                    raise ValueError(f"Reference differs by {delta:.4f} mm: {key}, {name}")
            if name == "B":
                baseline = residual
            elif name in ("T", "R", "UP"):
                for entry in transition_stats(baseline, residual):
                    moves.append(dict(**ident, method=name, **entry))
                if name in ("T", "R"):
                    audits.append(dict(**ident, method=name, n_edges=len(jump),
                                       **displacement_audit(baseline, residual)))
        print(f"{ident['year']}/{ident['id']}: {len(jump)} edges", flush=True)
    paired, attribution, audit_summary = summarize(scenes, moves, audits, reps, seed)
    write_rows(out / "scene_metrics.csv", scenes)
    write_rows(out / "scene_transitions.csv", moves)
    write_rows(out / "paired_contrasts.csv", paired)
    write_rows(out / "transition_summary.csv", attribution)
    write_rows(out / "edge_assignment_by_scene.csv", audits)
    write_rows(out / "edge_assignment_summary.csv", audit_summary)
    details = dict(scenes=len(records), edges=all_edges, reference_scene_method_metrics_checked=len(records)*3*8,
                   max_reference_difference_mm=max_ref_diff, bootstrap_replicates=reps, seed=seed,
                   signed_metrics="reference signed bias minus method signed bias; sign is not an accuracy gain",
                   contrast_sign="reference minus method", script_sha256=digest(__file__),
                   reference_csv_sha256=digest(context / "per_scene.csv"), methods=list(METHODS))
    (out / "run.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--context-dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.context_dir or not args.out or args.replicates <= 0:
            p.error("Supply --context-dir, --out and positive --replicates")
        run(args.context_dir, args.out, args.replicates, args.seed)
