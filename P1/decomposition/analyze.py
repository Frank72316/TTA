"""Offline decomposition of a frozen 2 x 2 x 2 scoring design. Python -B analyze.py."""
import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "prompts"
import sys
sys.path.append(str(ROOT.parent / "baseline"))
from layout_provenance import matches_digest

Q = ("Q0", "Q1")
SIDES = ("A", "B")
LIGHTS = ("L0", "L1")
CLASSES = ("same_direction", "toward_short", "exact_tie", "flip", "initial_tie", "zero_delta")
SCOPES = ("all", "no_conflict")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def keyed(rows, key):
    indexed = {row[key]: row for row in rows}
    assert len(indexed) == len(rows)
    return indexed


def write_csv(name, rows):
    path = ROOT / name
    with path.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(d):
    return "A" if d > 0 else "B" if d < 0 else "tie"


def classify(d, change):
    if d == 0:
        return "initial_tie"
    if change == 0:
        return "zero_delta"
    if (change > 0) == (d > 0):
        return "same_direction"
    if d + change == 0:
        return "exact_tie"
    return "toward_short" if choose(d + change) == choose(d) else "flip"


def input_tables():
    prompts = keyed(read_csv(SOURCE / "prompts.csv"), "pair_id")
    prior = keyed(read_csv(SOURCE / "pair_results.csv"), "pair_id")
    summary = keyed(read_csv(SOURCE / "summary.csv"), "group")
    env = json.loads((SOURCE / "environment.json").read_text())
    assert env["prompts_sha"] == sha(SOURCE / "prompts.csv")
    assert matches_digest(env["script_sha"], sha(SOURCE / "score_new.py"))
    assert env["hps_version"] == "v2.1" and env["brightness"] == 1.1
    assert len(prompts) == 64 and prompts.keys() == prior.keys()
    cells = {}
    for r in read_csv(SOURCE / "scores.csv"):
        i, s, q, l = key = tuple(r[k] for k in ("pair_id", "side", "prompt_condition", "light"))
        p = prompts[i]
        assert all(v not in (None, "") for v in r.values())
        assert key not in cells and r["image_id"] == p["image_"+s] == prior[i]["image_"+s]
        assert p["image_A"] != p["image_B"] and p["Q1"] == p["Q0"]+" The scene is at night, with dim lighting."
        assert p["Q0_token_sha"] != p["Q1_token_sha"] and r["text_token_sha"] == p[q+"_token_sha"]
        assert float(r["factor"]) == (1.0 if l == "L0" else 1.1) and r["source"] == "round2_fresh"
        assert r["native_dtype"] == "float16" and p["semantic_conflict"] == prior[i]["semantic_conflict"]
        value = float(r["score"])
        assert math.isfinite(value) and float(np.float16(value)) == value
        assert Decimal.from_float(value) == Decimal(r["score"])
        cells[key] = value
    assert set(cells) == set(product(prompts, SIDES, Q, LIGHTS)) and len(cells) == 512
    return prompts, prior, summary, cells


def decompose(cells, prompts, prior):
    result = []
    for i,p in sorted(prompts.items()):
        row = dict(pair_id=i, image_A=p["image_A"], image_B=p["image_B"], semantic_conflict=int(p["semantic_conflict"]))
        for s in SIDES:
            vals = []
            for j,q in enumerate(Q):
                a,b = (cells[i,s,q,l] for l in LIGHTS)
                row[f"{s}_{q}_L0"],row[f"{s}_{q}_L1"] = a,b
                row[f"b_{s}{j}"] = b-a
                vals.extend((a,b))
            row["h_"+s] = row[f"b_{s}1"]-row[f"b_{s}0"]
            row["max_ulp_"+s] = max(float(np.spacing(np.float16(x))) for x in vals)
        ha,hb = row["h_A"],row["h_B"]
        for j,q in enumerate(Q):
            d0 = row[f"A_{q}_L0"]-row[f"B_{q}_L0"]
            d1 = row[f"A_{q}_L1"]-row[f"B_{q}_L1"]
            delta = row[f"b_A{j}"]-row[f"b_B{j}"]
            tta = d0+delta/2
            assert d1-d0 == delta and tta == (d0+d1)/2
            assert tta == (row[f"A_{q}_L0"]+row[f"A_{q}_L1"]-row[f"B_{q}_L0"]-row[f"B_{q}_L1"])/2
            row[f"delta{j}"] = delta
            toward = -np.sign(d0)*delta if d0 else None
            fields = dict(d0=d0, d1=d1, d_tta=tta, lead=abs(d0), toward=toward,
                          pick0=choose(d0), pick1=choose(d1), pick_tta=choose(tta),
                          class_L1=classify(d0,delta), class_TTA=classify(d0,delta/2))
            row.update({q+"_"+k:v for k,v in fields.items()})
            for field in ("d0","d1","d_tta"):
                assert row[q+"_"+field] == float(prior[i][q+"_"+field])
            for field in ("pick0","pick1","pick_tta"):
                assert row[q+"_"+field] == prior[i][q+"_"+field]
            assert delta == float(prior[i][q+"_delta"])
        C,D = (ha+hb)/2,(ha-hb)/2
        I,G = row["delta1"]-row["delta0"], abs(row["delta1"])-abs(row["delta0"])
        energy = C*C+D*D
        assert I == ha-hb == 2*D and ha == C+D and hb == C-D
        assert energy == (ha*ha+hb*hb)/2
        assert I == float(prior[i]["I"]) and G == float(prior[i]["G"])
        row.update(C=C,D=D,I=I,G=G,C_sq=C*C,D_sq=D*D,total_sq=energy,total_amplitude=math.sqrt(energy))
        result.append(row)
    return result


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def describe(rows, scope):
    s = {"pairs":len(rows), "images":len(rows)*2}
    def quant(name, values):
        q1,med,q3 = np.quantile(values,[.25,.5,.75])
        s.update({name+"_q1":q1, name+"_median":med, name+"_q3":q3})
    def tally(name, values):
        v=np.asarray(values)
        for k,op in [("pos",v>0),("neg",v<0),("zero",v==0)]:
            s[name+"_"+k]=int(op.sum())
    h = np.array([[r["h_A"],r["h_B"]] for r in rows])
    quant("h",h)
    quant("abs_h",np.abs(h))
    tally("h",h)
    s["h_within_local_max_ulp"] = sum(abs(r["h_"+side])<=r["max_ulp_"+side] for r in rows for side in SIDES)
    pair_sign = Counter("zero" if 0 in (a,b) else "opposite" if a*b<0 else "both_pos" if a>0 else "both_neg" for a,b in h)
    for k in ("both_pos","both_neg","opposite","zero"):
        s["h_pair_"+k]=pair_sign[k]
    xy=Counter(map(tuple,h))
    s.update(origin=xy[0,0],distinct_xy=len(xy),repeated_locations=sum(n>1 for n in xy.values()),
             pairs_at_repeated_locations=sum(n for n in xy.values() if n>1),max_multiplicity=max(xy.values()))
    for name in ("C","D","I"):
        quant("abs_"+name, [abs(r[name]) for r in rows])
    s["I_zero"]=sum(r["I"]==0 for r in rows)
    gs=sorted(r["G"] for r in rows)
    tally("G",gs)
    quant("G",gs)
    s["G_lower_middle"],s["G_upper_middle"]=gs[(len(gs)-1)//2],gs[len(gs)//2]
    tally("absC_minus_absD",[abs(r["C"])-abs(r["D"]) for r in rows])
    c2=sum(r["C_sq"] for r in rows)
    d2=sum(r["D_sq"] for r in rows)
    total=c2+d2
    s.update(sum_C_sq=c2,sum_D_sq=d2,sum_total_sq=total,R=ratio(c2,total))
    leave=[]
    for r in rows:
        x=ratio(c2-r["C_sq"],total-r["total_sq"])
        r["R_without_pair_"+scope] = x if x is not None else "NA"
        if x is not None:
            leave.append((x,r["pair_id"]))
    for label,fn in [("min",min),("max",max)]:
        endpoint=fn((x for x,_ in leave),default=None)
        s["loo_"+label]=endpoint
        s["loo_"+label+"_ids"]=",".join(i for x,i in leave if x==endpoint)
    top=sorted(rows,key=lambda r:(-r["total_sq"],r["pair_id"]))[:3]
    s["largest_energy_id"]=top[0]["pair_id"]
    s["largest_energy_share"]=ratio(top[0]["total_sq"],total)
    s["top3_energy_share"]=ratio(sum(r["total_sq"] for r in top),total)
    s["R_without_top3"]=ratio(c2-sum(r["C_sq"] for r in top),total-sum(r["total_sq"] for r in top))
    for j,q in enumerate(Q):
        tally(q+"_brightness",[r[f"b_{side}{j}"] for r in rows for side in SIDES])
        quant(q+"_lead",[r[q+"_lead"] for r in rows])
        quant(q+"_abs_delta",[abs(r[f"delta{j}"]) for r in rows])
        for mode in ("L1","TTA"):
            counts=Counter(r[q+"_class_"+mode] for r in rows)
            assert sum(counts[c] for c in CLASSES)==len(rows)
            for c in CLASSES:
                s[f"{q}_{mode}_{c}"]=counts[c]
        towards=[r for r in rows if r[q+"_toward"] is not None and r[q+"_toward"]>0]
        s[q+"_n_toward"]=len(towards)
        quant(q+"_toward_lead",[r[q+"_lead"] for r in towards])
        quant(q+"_toward_move",[r[q+"_toward"] for r in towards])
        quant(q+"_toward_fraction",[r[q+"_toward"]/r[q+"_lead"] for r in towards])
    return s


def precision(cells):
    vals=np.array(list(cells.values()),dtype=np.float64)
    unique=np.unique(vals)
    gaps=Counter(np.diff(unique))
    spacing=Counter(np.spacing(vals.astype(np.float16)).astype(np.float64))
    assert sum(gaps.values())==len(unique)-1 and sum(spacing.values())==len(vals)
    stats=dict(n=len(vals),minimum=min(vals),maximum=max(vals),unique_values=len(unique),
               gap_min=min(gaps),gap_max=max(gaps),distinct_gap_sizes=len(gaps))
    step=min(gaps)
    for gap,n in sorted(gaps.items()):
        k=gap/step
        assert k.is_integer()
        stats[f"adjacent_gap_{int(k)}_steps"]=n
    for gap,n in sorted(spacing.items()):
        stats[f"binary16_ulp_{gap:.13f}".rstrip("0")]=n
    return stats


def examples(rows):
    amplitudes=sorted(r["total_amplitude"] for r in rows)
    lower,upper=amplitudes[(len(rows)-1)//2],amplitudes[len(rows)//2]
    target=(lower+upper)/2
    # Both middle order statistics are equidistant by definition, even if their
    # computed midpoint rounds toward one side. Include all ties at both endpoints.
    central=[r for r in rows if r["total_amplitude"] in (lower,upper)]
    return target,[("共同分量绝对值最大",min(rows,key=lambda r:(-abs(r["C"]),r["pair_id"]))),
                   ("差异分量绝对值最大",min(rows,key=lambda r:(-abs(r["D"]),r["pair_id"]))),
                   ("总变化幅度最接近中位数",min(central,key=lambda r:r["pair_id"]))]


def main():
    prompts,prior,previous,cells=input_tables()
    rows=decompose(cells,prompts,prior)
    for r in rows:
        r["R_without_pair_all"]="NA"
        r["R_without_pair_no_conflict"]="NA"
    groups={"all":rows,"no_conflict":[r for r in rows if not r["semantic_conflict"]]}
    sums={key:describe(group,key) for key,group in groups.items()}
    for key,prior_key in [("all","new_all"),("no_conflict","no_conflict")]:
        s,p=sums[key],previous[prior_key]
        assert s["pairs"]==int(p["n"]) and s["G_median"]==float(p["G_median"])
        assert [s["G_"+k] for k in ("pos","neg","zero")]==[int(p["G_"+k]) for k in ("positive","negative","zero")]
    p=sums["all"]
    assert [p["G_pos"],p["G_neg"],p["G_zero"],p["G_median"]]==[23,34,7,-0.0001220703125]
    assert len(groups["no_conflict"])==58
    prec=precision(cells)
    write_csv("pair_decomposition.csv",rows)
    write_csv("summary.csv",[dict(scope=k,metric=m,value=v if v is not None else "NA")
                              for k,stats in [*sums.items(),("raw_scores",prec)] for m,v in stats.items()])
    from present import render
    render(rows,sums,prec,prompts,examples(rows))
    print("Verified 64 pairs, 512 scores, 58 non-conflicts, all pair identities and both G summaries.")
    print({k:(s["h_median"],s["abs_h_median"],s["R"],s["loo_min"],s["loo_max"]) for k,s in sums.items()})

if __name__=="__main__":
    main()

