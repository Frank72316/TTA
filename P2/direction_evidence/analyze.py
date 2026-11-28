"""Offline analysis of the frozen spatial-relation experiment. Standard library only."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from math import comb
from pathlib import Path
from statistics import mean, quantiles
import struct
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "prompt_aware_flip"
sys.path.insert(0, str(SOURCE))
from common import choose, rows, save_json, save_rows, sha

DIRECTIONS = ("left", "right")
STRATEGIES = ("original", "image_only", "synchronized")


def box_truth(pairs, images, annotations):
    """Recover X relative to Y from box coordinates before reading any scores."""
    raw = None
    if annotations:
        frozen = json.loads((SOURCE / "freeze.json").read_text())
        assert sha(annotations.read_bytes()) == frozen["annotation_sha256"]
        raw = {a["id"]: a for a in json.loads(annotations.read_bytes())["annotations"]}
    truth = {}
    for r in images:
        pid, side = r["pair_id"], r["side"]
        assert pairs[pid]["image_" + side] == r["image_id"]
        w = int(r["width"])
        intervals = []
        for category in ("x", "y"):
            box = json.loads(r[category + "_original_xywh"])
            if raw is not None:
                a = raw[int(r[category + "_ann"])]
                assert a["bbox"] == box and a["iscrowd"] == 0
                assert (a["image_id"], a["category_id"]) == (int(r["image_id"]), int(pairs[pid][category + "_cat"]))
            x, _, width, _ = box
            intervals.append((max(0, x), min(w, x + width)))
        (xl, xr), (yl, yr) = intervals
        assert xr <= yl or yr <= xl
        assert abs(xl + xr - yl - yr) / (2*w) >= .2
        direction = "left" if xr <= yl else "right"
        ax, _, bx, _ = json.loads(r["x_input_xyxy"])
        ay, _, by, _ = json.loads(r["y_input_xyxy"])
        assert (bx <= ay if direction == "left" else by <= ax)
        assert (pid, direction) not in truth
        truth[pid, direction] = side
    for pid,p in pairs.items():
        assert {truth[pid,q] for q in DIRECTIONS} == {"A","B"}
        assert all(truth[pid,q] == p["winner_"+q] for q in DIRECTIONS)
    return truth


def load_scores(pairs):
    data = rows(SOURCE / "scores.csv")
    bank = {}
    for r in data:
        pid,side,v,q = (r[k] for k in ("pair_id","side","view","direction"))
        assert (r["image_id"],r["prompt"]) == (pairs[pid]["image_"+side], pairs[pid]["prompt_"+q])
        key = pid,side,v,q
        value = float(r["score"])
        # First verify the exact saved decimal is a native binary16 value.
        assert Fraction(r["score"]) == Fraction(value) and struct.unpack("e",struct.pack("e",value))[0] == value
        assert key not in bank
        bank[key] = value
    assert set(bank) == set(product(pairs, "AB", ("O","F"), DIRECTIONS))
    return bank


def reproduce(pairs, truth, bank):
    previous = {r["pair_id"]:r for r in rows(SOURCE/"pair_results.csv")}
    assert set(previous) == set(pairs)
    old = {}
    for pid in pairs:
        choices = {}
        for st in STRATEGIES:
            for q, opposite in (("left","right"),("right","left")):
                def candidate(side):
                    base = bank[pid,side,"O",q]
                    if st == "original":
                        return base
                    second = q if st == "image_only" else opposite
                    return (base+bank[pid,side,"F",second])/2
                choice = choose(candidate("A") - candidate("B"))
                assert choice == previous[pid][st+"_"+q]
                choices[st+"_"+q] = choice
            choices[st+"_both"] = int(all(choices[st+"_"+q] == truth[pid,q] for q in DIRECTIONS))
            choices[st+"_same"] = int(choices[st+"_left"] == choices[st+"_right"] != "=")
        old[pid] = choices
    totals = {st:sum(r[st+"_both"] for r in old.values()) for st in STRATEGIES}
    assert list(totals.values()) == [1,2,0]
    return old, totals


def mirror_photo(pid,side,p,truth,bank):
    result = dict(pair_id=pid,side=side,image_id=p["image_"+side])
    correct = ties = 0
    for q in DIRECTIONS:
        o,f = bank[pid,side,"O",q],bank[pid,side,"F",q]
        expected = "O" if truth[pid,q] == side else "F"
        picked = "O" if o>f else "F" if f>o else "="
        result.update({q+"_O":o, q+"_F":f, q+"_expected":expected, q+"_choice":picked,
                       q+"_truth_margin":(o-f) if expected=="O" else (f-o)})
        correct += picked == expected
        ties += picked == "="
    result.update(correct_questions=correct,tie_questions=ties,
                  outcome="has_tie" if ties else ("both_wrong","one_correct","both_correct")[correct])
    return result


def group_row(group, data):
    return dict(group=group,n=len(data),**{k:" ".join(r["pair_id"] for r in data if r["assignment"]==k)
                                           for k in ("correct","reversed","tie")},
                **{k+"_n":sum(r["assignment"]==k for r in data) for k in ("correct","reversed","tie")})


def analyze(out, annotations=None):
    out.mkdir(parents=True,exist_ok=True)
    frozen = json.loads((SOURCE/"freeze.json").read_text())
    for name in ("pairs.csv","images.csv"):
        assert sha((SOURCE/name).read_bytes()) == frozen["hashes"][name]
    pair_rows = rows(SOURCE/"pairs.csv")
    pairs = {p["pair_id"]:p for p in pair_rows}
    images = rows(SOURCE/"images.csv")
    assert len(pairs) == len(pair_rows) == 64 and len(images) == len({r["image_id"] for r in images}) == 128
    truth = box_truth(pairs,images,annotations)
    bank = load_scores(pairs)
    old, totals = reproduce(pairs,truth,bank)
    detail, mirrors = [], []
    for pid,p in pairs.items():
        left,right = (truth[pid,q] for q in DIRECTIONS)
        r = dict(pair_id=pid,x=p["x"],y=p["y"],prompt_left=p["prompt_left"],prompt_right=p["prompt_right"],
                 image_A=p["image_A"],image_B=p["image_B"],true_left=left,true_right=right)
        for side in "AB":
            l,h = (bank[pid,side,"O",q] for q in DIRECTIONS)
            r.update({side+"_left":l,side+"_right":h,"e_"+side:l-h})
            mirrors.append(mirror_photo(pid,side,p,truth,bank))
        m = r["e_"+left]-r["e_"+right]
        exact = sum(Fraction(bank[pid,s,"O",q])*factor for s,q,factor in
                    ((left,"left",1),(left,"right",-1),(right,"left",-1),(right,"right",1)))
        assert Fraction(m) == exact
        inferred_left = choose(r["e_A"]-r["e_B"])
        r.update(M=m,assignment="correct" if m>0 else "reversed" if m<0 else "tie",
                 inferred_left=inferred_left,inferred_right={"A":"B","B":"A","=":"="}[inferred_left],**old[pid])
        assert (inferred_left==left)==(m>0)
        r["each_strategy_same"] = int(all(r[st+"_same"] for st in STRATEGIES))
        r["all_six_same"] = int(r["each_strategy_same"] and len({r[st+"_left"] for st in STRATEGIES})==1)
        detail.append(r)
    groups = [group_row(key,[r for r in detail if r[key]]) for key in
              ("original_same","image_only_same","synchronized_same","each_strategy_same","all_six_same")]
    save_rows(out/"pair_results.csv",detail)
    save_rows(out/"mirror_results.csv",mirrors)
    save_rows(out/"same_choice_groups.csv",groups)
    values = [r["M"] for r in detail]
    counts = Counter(r["assignment"] for r in detail)
    n = counts["correct"]+counts["reversed"]
    tail = min(counts["correct"],counts["reversed"])
    sign_p = float(min(Fraction(1), Fraction(2*sum(comb(n,j) for j in range(tail+1)),2**n))) if n else None
    q1,q2,q3 = quantiles(values,n=4,method="inclusive")
    checked_mirror = Counter(r["outcome"] for r in mirrors)
    assert sum(checked_mirror.values())==128
    summary = dict(pairs=len(detail),photos=len(images),cached_scores=len(bank),old_both=totals,
        assignments={k:counts[k] for k in ("correct","reversed","tie")},sign_test_n=n,sign_test_two_sided_p=sign_p,
        M=dict(min=min(values),q1=q1,median=q2,q3=q3,max=max(values),mean=mean(values)),
        max_cases=[r["pair_id"] for r in detail if r["M"]==max(values)],
        min_cases=[r["pair_id"] for r in detail if r["M"]==min(values)],
        mirror={k:checked_mirror[k] for k in ("both_correct","one_correct","both_wrong","has_tie")},
        mirror_question_correct=sum(r["correct_questions"] for r in mirrors),
        mirror_question_ties=sum(r["tie_questions"] for r in mirrors),
        same_choice={r["group"]:{k:r[k+"_n"] for k in ("correct","reversed","tie")} for r in groups},
        input_hashes={n:sha((SOURCE/n).read_bytes()) for n in ("images.csv","pairs.csv","scores.csv","pair_results.csv")},
        annotation_sha256=frozen["annotation_sha256"])
    save_json(out/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE)
    parser.add_argument("--annotations",type=Path,help="Recheck the exported COCO boxes against a local instances_val2017.json")
    args=parser.parse_args()
    print(json.dumps(analyze(args.out,args.annotations),ensure_ascii=False,indent=2))
