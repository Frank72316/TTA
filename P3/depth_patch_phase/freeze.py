"""Draw before inference using only the old manifests, without pixels or scores."""
from collections import defaultdict
import random
from study import HERE, PRIOR, SEED, PAIRS, read,write,save,sha,phase

def main():
    source=PRIOR/"frames.csv"
    mapping=PRIOR/"scenes.csv"
    scenes={r["frame"]:r["scene"] for r in read(mapping)}
    groups=defaultdict(list)
    for r in read(source):
        groups[scenes[r["frame"]]].append(r)
    assert len(groups)==95 and sum(map(len,groups.values()))==128
    rng=random.Random(SEED)
    chosen=[]
    for s in rng.sample(sorted(groups),32):
        r=rng.choice(sorted(groups[s],key=lambda r:r["frame_id"]))
        chosen.append(dict(**r,scene=s))
    dest=HERE/"frames.csv"
    if dest.exists():
        assert read(dest)==chosen
    else:
        write(dest,chosen)
    save(HERE/"freeze.json",dict(seed=SEED,frame_sha256=sha(dest),protocol_sha256=sha(HERE/"protocol.md"),
        sources={p.name:sha(p) for p in (source,mapping)},scenes=len(groups),selected=len(chosen),
        prediction_table=[dict(a=a,b=b,phase_a=phase(a),phase_b=phase(b),
                          grid_prior="opposite" if phase(a)==phase(b) else "same",position_prior="opposite") for a,b in PAIRS]))

if __name__=="__main__":
    main()
