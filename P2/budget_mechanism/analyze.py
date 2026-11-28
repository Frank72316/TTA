"""Exact finite-pool replay for brightness and contrast; no model or image I/O."""
import csv
from itertools import combinations, product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "translation"
CELLS = [(2, 3), (2, 5), (4, 3), (4, 5)]  # (number of views, reciprocal of original weight)
FAMILIES = ("B", "C", "BC")
FOCUS = ("N078", "N137")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write(name, rows):
    with (ROOT / name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    pairs = sorted(read(SOURCE / "pairs_frozen.csv"), key=lambda p: p["pair_id"])
    assert len(pairs) == len({p["pair_id"] for p in pairs}) == len({p["normalized_prompt"] for p in pairs}) == 256
    assert len({p["prompt_id"] for p in pairs}) == 256
    assert len({p["image_" + s] for s in "AB" for p in pairs}) == 512
    records = read(SOURCE / "scores.csv")
    relevant = [r for r in records if r["view"] == "O" or r["view"][0] in "BC"]
    bank = {(r["pair_id"], r["side"], r["view"]): r for r in relevant}
    assert len(bank) == len(relevant) == 4608
    params = {r["view"]: float(r["param"]) for r in relevant if r["view"] != "O"}
    pools = {f: tuple(sorted((v for v in params if v.startswith(f)), key=params.get)) for f in "BC"}
    assert all(len(p) == len({params[v] for v in p}) == 4 for p in pools.values())
    views = ["O", *pools["B"], *pools["C"]]
    margins = {v: np.empty(len(pairs), dtype=np.float64) for v in views}
    detail = []
    for i, p in enumerate(pairs):
        a, b = int(p["rank_A"]), int(p["rank_B"])
        assert a != b and p["human_winner_P0"] == ("A" if a < b else "B")
        tokens = set()
        for v in views:
            scores = []
            for side in "AB":
                r = bank[p["pair_id"], side, v]
                assert (r["prompt_id"], r["P0"], r["image_id"]) == (p["prompt_id"], p["P0"], p["image_" + side])
                assert r["native_dtype"] == "float16" and (v == "O" or float(r["param"]) == params[v])
                tokens.add(r["token_sha256"])
                value = np.float64(r["score"])
                assert np.isfinite(value) and value == np.float64(np.float16(value))
                scores.append(value)
            m = (scores[0] - scores[1]) * (1 if a < b else -1)
            margins[v][i] = m
            detail.append(dict(**p, view=v, factor=params.get(v, ""), score_A=scores[0], score_B=scores[1],
                               m0=margins["O"][i], margin=m, shift=m - margins["O"][i]))
        assert len(tokens) == 1
    assert (sum(margins["O"] > 0), sum(margins["O"] < 0)) == (175, 81)
    write("parameter_margins.csv", detail)
    return pairs, pools, margins


def choices(pools, family, n):
    if family in ("B", "C"):
        return list(combinations(pools[family], n))
    return [b + c for b, c in product(combinations(pools["B"], n // 2), combinations(pools["C"], n // 2))]


def probabilities(pairs, pools, m):
    rows, summary, coverage = [], [], []
    matrices, counts = {}, {}
    ids = np.array([p["pair_id"] for p in pairs])
    orig = m["O"]
    for n, q in CELLS:
        for family in FAMILIES:
            specs = choices(pools, family, n)
            sums = np.stack([sum(m[v] for v in spec) for spec in specs], axis=1)
            # All inputs are dyadic. An integer-weight numerator preserves exact ties.
            if q == n + 1:
                numer = orig[:, None] + sums
                divisor = q
            else:
                numer = n * orig[:, None] + (q - 1) * sums
                divisor = n * q
            matrices[n,q,family] = numer / divisor
            good, tie, bad = [(test.sum(axis=1)) for test in (numer > 0, numer == 0, numer < 0)]
            assert np.all(good + tie + bad == len(specs))
            counts[n,q,family] = good
            prefix = dict(views=n, weight=f"1/{q}", design="equal" if q == n+1 else "diagnostic", strategy=family)
            for idx, p in enumerate(pairs):
                rows.append(dict(pair_id=p["pair_id"], **prefix, m0=orig[idx], combinations=len(specs),
                                 correct=good[idx], tie=tie[idx], wrong=bad[idx], p_correct=good[idx]/len(specs),
                                 p_tie=tie[idx]/len(specs), p_wrong=bad[idx]/len(specs)))
            pgood, ptie, pbad = [t / len(specs) for t in (good, tie, bad)]
            summary.append(dict(**prefix, combinations=len(specs), accuracy_pct=100*pgood.mean(),
                                correct=pgood.sum(), tie=ptie.sum(), wrong=pbad.sum(), fixes=pgood[orig<0].sum(),
                                harms=pbad[orig>0].sum(), correct_to_tie=ptie[orig>0].sum(),
                                wrong_to_tie=ptie[orig<0].sum()))
            assert np.isclose(pgood.sum(), 175 + pgood[orig<0].sum() - pbad[orig>0].sum() - ptie[orig>0].sum())
        for cutoff in ("any", "quarter"):
            sets = {}
            for f in FAMILIES:
                c = counts[n,q,f]
                hits = c > 0 if cutoff == "any" else 4*c >= matrices[n,q,f].shape[1]
                sets[f] = set(ids[(orig < 0) & hits])
            b, c, mix = [sets[f] for f in FAMILIES]
            record = dict(views=n, weight=f"1/{q}", criterion=cutoff)
            for k, group in dict(B=b, C=c, BC=mix, both=b&c, B_only=b-c, C_only=c-b, mix_only=mix-b-c).items():
                record[k] = len(group)
                record[k+"_ids"] = " ".join(sorted(group))
            coverage.append(record)
    write("pair_probabilities.csv", rows)
    write("factorial_summary.csv", summary)
    write("coverage.csv", coverage)
    return matrices, rows, summary, coverage


def audit_reference(summary, coverage):
    snapshot = read(ROOT / "reference.csv")
    original = ROOT.parent / "method_relations/cached_relations.csv"
    if original.is_file():
        local = [r for r in read(original) if (r["n"],r["keep"],r["replace"]) == ("256","B","C")]
        assert snapshot == [{k:r[k] for k in snapshot[0]} for r in local]
    for ref in snapshot:
        k = int(ref["K"])
        cell = next(r for r in coverage if (r["views"], r["weight"], r["criterion"]) == (k-1, f"1/{k}", "any"))
        for key, old in dict(B="keep_fix_pairs", C="replace_fix_pairs", both="both_fix", B_only_ids="keep_only_ids",
                             C_only_ids="replace_only_ids", mix_only_ids="mix_only_ids").items():
            assert str(cell[key]) == ref[old], (k, key, cell[key], ref[old])
        for f, old in zip(FAMILIES, ("keep","replace","mix")):
            r = next(r for r in summary if (r["views"],r["weight"],r["strategy"]) == (k-1,f"1/{k}",f))
            assert r["combinations"] == int(ref["mixed_combos" if f=="BC" else "pure_combos"])
            for outcome in ("correct","tie","wrong"):
                assert np.isclose(r[outcome], float(ref[f"{old}_{outcome}_expected"]), rtol=0, atol=1e-12)


def case_history(pairs, pools, m, rows, coverage):
    equal = {r["views"]: r for r in coverage if r["criterion"]=="any" and r["weight"]==f"1/{r['views']+1}"}
    reached = {i for r in equal.values() for f in FAMILIES for i in r[f+"_ids"].split()}
    data = {(r["pair_id"],r["views"],r["strategy"]):r for r in rows if r["design"]=="equal"}
    migration, chains = [], []
    for idx, p in enumerate(pairs):
        i = p["pair_id"]
        if i not in reached:
            continue
        record = dict(**p, m0=m["O"][idx])
        for k in (3,5):
            hit = []
            for f in FAMILIES:
                r = data[i,k-1,f]
                for col in ("correct","tie","wrong","p_correct","p_tie","p_wrong"):
                    record[f"K{k}_{f}_{col}"] = r[col]
                if r["correct"]:
                    hit.append(f)
            record[f"K{k}_reachable"] = " ".join(hit) or "none"
        migration.append(record)
        if i not in FOCUS:
            continue
        for f in "BC":
            for a,b in combinations(pools[f],2):
                start = m["O"][idx]+m[a][idx]+m[b][idx]
                m3 = start/3
                u,v = [j for j in pools[f] if j not in (a,b)]
                mu,mv = m[u][idx],m[v][idx]
                m5 = (start+mu+mv)/5
                chains.append(dict(pair_id=i, strategy=f, selected=f"{a} {b}", sum3=start, M3=m3,
                                   extra1=u, extra1_margin=mu, extra1_change_if_alone=(mu-m3)/4,
                                   extra2=v, extra2_margin=mv, extra2_change_if_alone=(mv-m3)/4,
                                   sum5=start+mu+mv, M5=m5))
    write("case_migration.csv", migration)
    write("parameter_chains.csv", chains)
    return migration


def complements(pairs, pools, m, matrices, coverage):
    specs = choices(pools,"BC",4)
    complement = [tuple(v for v in pools["B"]+pools["C"] if v not in s) for s in specs]
    halves = [(i,specs.index(s)) for i,s in enumerate(complement) if i < specs.index(s)]
    assert sorted(k for pair in halves for k in pair) == list(range(36))
    mix = matrices[4,5,"BC"]
    b,c = [matrices[4,5,f][:,0] for f in "BC"]
    target = (b+c)/2
    residual = abs(mix.mean(1)-target)
    items, pair_errors, excess = [], [], []
    relevant = (b<=0) & (c<=0)
    for j,k in halves:
        avg = (mix[:,j]+mix[:,k])/2
        err = abs(avg-target)
        wins = (mix[:,j]>0).astype(int) + (mix[:,k]>0).astype(int)
        pair_errors.extend(err)
        excess.extend(np.maximum(0,wins[relevant]-1))
        for i,p in enumerate(pairs):
            items.append(dict(pair_id=p["pair_id"], choice=" ".join(specs[j]), complement=" ".join(specs[k]),
                              M=mix[i,j], M_complement=mix[i,k], pure_B=b[i], pure_C=c[i], target=target[i],
                              mean_two=avg[i], abs_error=err[i], applicable=int(relevant[i]), strict_wins=wins[i]))
    checks = []
    for name, errors, tolerance in [("all36_mean",residual,1e-15),("complement_mean",pair_errors,1e-15),
                                    ("conditional_at_most_one",excess,0)]:
        details = np.asarray(errors)
        checks.append(dict(check=name, comparisons=details.size, max_residual=details.max(), tolerance=tolerance,
                           violations=np.count_nonzero(details>tolerance)))
    assert all(r["violations"] == 0 for r in checks)
    assert ((mix[relevant]>0).sum(1)<=18).all()
    write("complement_checks.csv",items)
    write("identity_summary.csv",checks)
    ids = next(r["mix_only_ids"].split() for r in coverage if (r["views"],r["weight"],r["criterion"])==(4,"1/5","any"))
    examples = []
    for i,p in enumerate(pairs):
        if p["pair_id"] not in ids:
            continue
        values = mix[i]
        j = int(np.flatnonzero(values>0)[0])
        k = specs.index(complement[j])
        examples.append(dict(pair_id=p["pair_id"], pure_B=b[i], pure_C=c[i], minimum=values.min(), mean=values.mean(),
                             maximum=values.max(), correct=sum(values>0), tie=sum(values==0), wrong=sum(values<0),
                             p_correct=(values>0).mean(), choice=" ".join(specs[j]), M=values[j],
                             complement=" ".join(specs[k]), M_complement=values[k]))
    write("mixed_cases.csv",examples)


def main():
    pairs, pools, margins = load_inputs()
    matrices, rows, summary, coverage = probabilities(pairs,pools,margins)
    audit_reference(summary,coverage)
    case_history(pairs,pools,margins,rows,coverage)
    complements(pairs,pools,margins,matrices,coverage)
    print("256 prompts, 4,608 cached O/B/C scores, 12 strategy cells, 4,608 complement pairs verified.")


if __name__ == "__main__":
    main()
