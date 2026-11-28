"""Independent checks against source scores and the original P2 per-pair outcomes."""
import csv
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "translation"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    for file, digest in {
        "scores.csv":"fba976a8f06ece78da21371a71c6ea9619ca40074e752acfdde1be2e57668275",
        "pairs_frozen.csv":"7b06fc4d4c766fe70b41bfe434f90ab92d25b16e2b734a3f5af9e2d6da0095fe",
    }.items():
        assert sha256((OLD/file).read_bytes()).hexdigest() == digest
    values = defaultdict(dict)
    for r in read(OLD/"scores.csv"):
        if r["view"][0] in "OBC":
            values[r["pair_id"],r["side"]][r["view"]] = Fraction(r["score"])
    # A common exact integer scale for the two sides' original decimal CSV values.
    scale = max(v.denominator for d in values.values() for v in d.values())
    for d in values.values():
        for k,x in d.items():
            assert scale % x.denominator == 0
            d[k] = int(x*scale)
    results = read(ROOT/"pair_probabilities.csv")
    observed = {(r["pair_id"],int(r["views"]),r["weight"],r["strategy"]):r for r in results}
    assert len(observed) == len(results) == 3072
    strategies = {"B":"BB","C":"CC","BC":"BC"}
    old = {(r["pair_id"],r["strategy"]):r for r in read(OLD/"pair_strategy_results.csv")}
    decisions = 0
    old_checks = 0
    for pair in read(OLD/"pairs_frozen.csv"):
        i = pair["pair_id"]
        winner = pair["human_winner_P0"]
        loser = "B" if winner == "A" else "A"
        x,y = values[i,winner],values[i,loser]
        for n,q,f in product((2,4),(3,5),strategies):
            bs,cs = [sorted(k for k in x if k.startswith(fam)) for fam in "BC"]
            if f == "BC":
                specs = (b+c for b,c in product(combinations(bs,n//2),combinations(cs,n//2)))
            else:
                specs = combinations(bs if f=="B" else cs,n)
            counts = Counter()
            for spec in specs:
                # Evaluate both candidates' scores independently before comparing.
                left = n*x["O"]+(q-1)*sum(x[v] for v in spec)
                right = n*y["O"]+(q-1)*sum(y[v] for v in spec)
                counts["correct" if left>right else "wrong" if left<right else "tie"] += 1
                decisions += 1
            row = observed[i,n,f"1/{q}",f]
            for col in ("correct","wrong","tie"):
                assert int(row[col]) == counts[col], (i,n,q,f,col)
                if (n,q)==(2,3):
                    assert counts[col] == int(old[i,strategies[f]][col])
            if (n,q)==(2,3):
                old_checks += 1
    assert decisions==33792 and old_checks==768
    checks = read(ROOT/"complement_checks.csv")
    grouped = defaultdict(list)
    for r in checks:
        grouped[r["pair_id"]].append(r)
    assert len(grouped)==256
    for group in grouped.values():
        labels = [r[k] for r in group for k in ("choice","complement")]
        assert len(group)==18 and len(set(labels))==36
        for r in group:
            assert not (set(r["choice"].split()) & set(r["complement"].split()))
    assert all(int(row["violations"])==0 for row in read(ROOT/"identity_summary.csv"))
    cases = read(ROOT/"case_migration.csv")
    assert len(cases)==7
    for r in cases:
        group = [old[r["pair_id"],f] for f in ("BB","CC")]
        both_old = all(int(s["correct"])>0 for s in group)
        assert both_old == (r["pair_id"] in {"N015","N078","N100","N137","N156"})
    print(f"PASS {decisions} integer decisions, {old_checks} historical pair-strategy outcomes, "
          "256 complete complementary partitions, seven case histories.")


if __name__ == "__main__":
    main()
