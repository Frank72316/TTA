"""CPU-only random matching of frozen image-level interactions."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "decomposition"
PROMPTS = ROOT.parent / "prompts" / "prompts.csv"
DRAWS = 1_000_000
SEED = 42
BATCH = 4096


def read(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (ROOT / name).open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inputs():
    path = SOURCE / "pair_decomposition.csv"
    prompts = read(PROMPTS)
    out = sorted(read(path), key=lambda r:r["pair_id"])
    lookup = {r["pair_id"]:r for r in prompts}
    assert len(lookup) == len(out) == len(prompts) == 64
    assert {r["pair_id"] for r in out} == lookup.keys()
    # A permutation with no original partner therefore excludes every same-prompt match.
    for col in ("original_prompt","prompt_id","Q0","Q0_token_sha","Q1_token_sha"):
        assert len({r[col] for r in prompts}) == len(out), col
    assert len({r["image_"+s] for r in out for s in ("A","B")}) == 128
    for r in out:
        p = lookup[r["pair_id"]]
        assert p["semantic_conflict"] == r["semantic_conflict"]
        for side in ("A","B"):
            assert p["image_"+side] == r["image_"+side]
            b0,b1 = (float(r[f"{side}_{q}_L1"])-float(r[f"{side}_{q}_L0"]) for q in ("Q0","Q1"))
            assert float(r["h_"+side]) == b1-b0
    return out


def exact_units(*vectors):
    # All saved h values are binary rationals. Use their common denominator
    # so that Monte Carlo upper tails include exact ties without a tolerance.
    den = max(float(v).as_integer_ratio()[1] for vec in vectors for v in vec)
    result = [np.rint(np.asarray(vec)*den).astype(np.int64) for vec in vectors]
    assert all(np.array_equal(ints/den,orig) for ints,orig in zip(result,vectors))
    return result


def simulate(a,b,restricted,rng):
    n = len(a)
    dot = np.empty(DRAWS,dtype=np.int64)
    concordant = np.empty(DRAWS,dtype=np.int64)
    indices = np.arange(n)
    filled = 0
    while filled < DRAWS:
        # permuted(...,axis=1) shuffles each row independently. Conditioning
        # a uniform permutation on having no fixed point gives a uniform derangement.
        permutations = rng.permuted(np.broadcast_to(indices,(BATCH,n)),axis=1)
        if restricted:
            permutations = permutations[(permutations!=indices).all(axis=1)]
        accepted = permutations[:DRAWS-filled]
        count = len(accepted)
        products = a*b[accepted]
        sl = slice(filled, filled+count)
        dot[sl] = products.sum(axis=1)
        concordant[sl] = (products>0).sum(axis=1)
        filled += count
    return dot,concordant


def describe(scope,design,metric,draws,obs,expected,offset,scale):
    low,high = np.quantile(draws,[.025,.975],method="inverted_cdf")
    above=int((draws>=obs).sum())
    eq=int((draws==obs).sum())
    mean=draws.mean()
    sd=draws.std(ddof=1)
    assert abs(mean-expected)<8*sd/math.sqrt(DRAWS)
    return dict(scope=scope,design=design,metric=metric,draws=DRAWS,observed=offset+obs*scale,
                exact_null_mean=offset+expected*scale,sim_mean=offset+mean*scale,
                excess_over_exact_mean=(obs-expected)*scale,null_q025=offset+low*scale,
                null_q975=offset+high*scale,n_ge=above,n_equal=eq,fraction_ge=above/DRAWS,
                fraction_strictly_below=(DRAWS-above)/DRAWS,
                permutation_p=(above+1)/(DRAWS+1) if design=="unrestricted" else "")


def main():
    source=inputs()
    rows=[]; hist=[]; info=[]
    streams=np.random.SeedSequence(SEED).spawn(4)
    for group_index,(scope,group) in enumerate([("all",source),("no_conflict",[r for r in source if r["semantic_conflict"]=="0"])]):
        a0,b0=[np.array([float(r["h_"+s]) for r in group]) for s in ("A","B")]
        a,b=exact_units(a0,b0)
        n=len(a)
        assert n==(64 if scope=="all" else 58)
        S=int(a@a+b@b)
        assert S and int(max(abs(a).max(),abs(b).max()))**2*2*n < 2**53
        orig=int(a@b)
        same=int((a*b>0).sum())
        R=.5+orig/S
        published=sum(float(r["C_sq"]) for r in group)/sum(float(r["total_sq"]) for r in group)
        assert R == published or abs(R-published)<1e-15
        if scope=="all": assert same==42 and round(100*R,2)==83.24
        all_dot=int(a.sum())*int(b.sum())
        all_same=sum(int((a*sgn>0).sum())*int((b*sgn>0).sum()) for sgn in (-1,1))
        t=dict(scope=scope,n=n,R=R,mean_h_A=a0.mean(),mean_h_B=b0.mean(),same=same,
               marginal_dot_contribution=all_dot/(n*S),centered_pair_contribution=(orig-all_dot/n)/S,
               pearson_r=np.corrcoef(a0,b0)[0,1])
        assert abs(.5+t["marginal_dot_contribution"]+t["centered_pair_contribution"]-R)<1e-15
        info.append(t)
        for j,(design,restricted) in enumerate((("derangement",True),("unrestricted",False))):
            rng=np.random.default_rng(streams[2*group_index+j])
            dot,counts=simulate(a,b,restricted,rng)
            divisor=n-1 if restricted else n
            exp_dot=(all_dot-orig if restricted else all_dot)/divisor
            exp_same=(all_same-same if restricted else all_same)/divisor
            for metric,v,obs,exp,offset,scale in [("R",dot,orig,exp_dot,.5,1/S),("same_nonzero",counts,same,exp_same,0,1)]:
                rec=describe(scope,design,metric,v,obs,exp,offset,scale)
                rec["n_pairs"]=n
                rows.append(rec)
                vals,frequencies=np.unique(v,return_counts=True)
                assert frequencies.sum()==DRAWS
                hist.extend(dict(scope=scope,design=design,metric=metric,value=offset+int(k)*scale,
                                 frequency=int(f)) for k,f in zip(vals,frequencies))
                print(scope,design,metric,rec["observed"],rec["exact_null_mean"],
                      rec["null_q025"],rec["null_q975"],rec["n_ge"],flush=True)
    save("summary.csv",rows)
    save("null_distribution.csv",hist)
    save("mean_decomposition.csv",info)
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/"pair_decomposition.csv",PROMPTS)}
    (ROOT/"inputs.json").write_text(json.dumps(dict(sha256=hashes,seed=SEED,draws_per_scope_design=DRAWS,
                              numpy=np.__version__,bit_generator="PCG64",batch=BATCH),indent=2)+"\n")

if __name__=="__main__":
    main()

