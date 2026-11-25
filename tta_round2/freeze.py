"""Audit the actual old token stream and validate the Q manifest before scoring."""
import csv
from common import *

def main():
    with (ROOT/"rewrites.tsv").open(encoding="utf-8", newline="") as f:
        drafts = list(csv.DictReader(f,delimiter="\t"))
    assert [r["pair_id"] for r in drafts]==list(PAIRS)
    tagged = {r["pair_id"]:r for r in read_csv(OLD/"lighting.csv")}
    old_scores = read_csv(OLD/"scores.csv")
    index_scores(old_scores, PAIRS, ("P0","P1"))
    prompts, audits = [], []
    for draft in drafts:
        i=draft["pair_id"]
        p=PAIRS[i]
        q0=draft["Q0"]
        q1=q0+" "+SUFFIX
        f0,t0,n=token_record(p["P0"])
        f1,t1,_=token_record(p["P1"])
        g0,u0,_=token_record(q0)
        g1,u1,_=token_record(q1)
        suffix_ids=token_record(SUFFIX)[0][1:-1]
        assert u0==g0 and u1==g1 and u0!=u1 and u1[1:-1]==u0[1:-1]+suffix_ids
        assert p["P1"]==p["P0"]+" "+SUFFIX
        assert all(int(r["same_P0_P1_tokens"])==(t0==t1) and
                   int(r["tokens_before_truncation"])==len(f0 if r["prompt_condition"]=="P0" else f1)
                   for r in old_scores if r["pair_id"]==i)
        audits.append(dict(pair_id=i,P0=p["P0"],P1=p["P1"],P0_full_tokens=len(f0),P1_full_tokens=len(f1),
                           context=n,P0_sent=len(t0),P1_sent=len(t1),P1_complete=int(f1==t1),
                           identical=int(t0==t1),light=int(i in tagged),light_evidence=tagged.get(i,{}).get("evidence","—"),
                           P0_ids=json.dumps(t0),P1_ids=json.dumps(t1)))
        prompts.append(dict(pair_id=i,prompt_id=p["prompt_id"],image_A=p["image_A"],image_B=p["image_B"],
                            original_prompt=p["P0"],Q0=q0,Q1=q1,removed_light_or_tone=draft["removed_light_or_tone"],
                            semantic_conflict=int(draft["conflict_reason"]!="—"),conflict_reason=draft["conflict_reason"],
                            Q0_tokens=len(g0),Q1_tokens=len(g1),context=n,Q0_token_sha=token_hash(u0),Q1_token_sha=token_hash(u1)))
    path=ROOT/"prompts_round2.csv"
    if path.exists():
        assert read_csv(path)==[{k:str(v) for k,v in r.items()} for r in prompts],"Frozen prompts differ"
    else:
        assert not (ROOT/"freeze.json").exists()
        write_csv(path,prompts)
    manifest=dict(prompts_sha256=digest(path), pairs_sha256=digest(OLD/"pairs.csv"), old_scores_sha256=digest(OLD/"scores.csv"),
                  draft_sha256=digest(ROOT/"rewrites.tsv"),old_lighting_sha256=digest(OLD/"lighting.csv"),
                  primary="median of within-pair G", bootstrap="percentile; numpy default_rng(42); 10000 complete pairs resamples",
                  cases="sort abs(I) descending, ID ascending; random.Random(42).sample(sorted remaining IDs,3)",
                  blind="random.Random(42): sample 16 sorted IDs before seeing scores, shuffle 64 conditions, randomize left/right")
    if (ROOT/"freeze.json").exists():
        assert jsonfile(ROOT/"freeze.json")==manifest
    else:
        write_json(ROOT/"freeze.json",manifest)
    write_csv(ROOT/"old_prompt_audit.csv",audits)
    print(f"{len(frozen())} prompts frozen, Q1 max {max(p['Q1_tokens'] for p in prompts)} tokens", manifest["prompts_sha256"])

if __name__=="__main__":
    main()

