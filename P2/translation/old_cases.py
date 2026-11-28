"""Read-only P2 audit and rendering through HPS's actual validation transform."""
import re
from collections import Counter
from zipfile import ZipFile
import numpy as np
from PIL import Image
from common import *

def main():
    from hpsv2.src.open_clip.transform import image_transform
    env=file_json(P2/"environment.json")
    assert digest(P2/"views.py")==env["source_hashes"]["views.py"]
    tf=image_transform(224,False,resize_longest_max=True)
    assert re.sub(r" at 0x[0-9a-fA-F]+","",repr(tf))==env["preprocess"]
    scores=read_csv(P2/"scores.csv")
    ps=read_csv(PILOT/"pairs.csv")
    bank={(r["pair_id"],r["side"],r["view"]):r for r in scores}
    assert len(bank)==len(scores)==2176
    found={}
    for p in ps:
        i=p["pair_id"]; human=p["human_winner_P0"]
        d={v:float(bank[i,"A",v]["score"])-float(bank[i,"B",v]["score"]) for v in ["O",*sum(FAMILIES.values(),[])]}
        events={}
        for s in ["BB","CC","JJ","TT","BC","BJ","BT","CJ","CT","JT"]:
            events[s]=[(x,y,(d["O"]+d[x]+d[y])/3) for x,y in combinations(s)]
        if any(choice(z)!=choice(d["O"]) for vals in events.values() for _,__,z in vals):
            found[i]=(p,d,events)
    assert set(found)=={"003","005","008","031","053"}
    for i,s,state,n in [("003","TT","correct",4),("003","BT","correct",4),("008","BT","wrong",16)]:
        p,d,ev=found[i]
        assert sum(outcome(choice(z),p["human_winner_P0"])==state for _,__,z in ev[s])==n
    folder=ROOT/"old_cases";assets=folder/"assets";assets.mkdir(exist_ok=True,parents=True)
    order=["003","008","053","005","031"]
    details=[]
    for i in order:
        p,d,ev=found[i];human=p["human_winner_P0"]
        for s in ("A","B"):
            src=Path(p["image_"+s]);fname=f"{i}-{s}.webp"
            with ZipFile(PILOT/"data"/(src.parent.name+".zip")) as z:(assets/fname).write_bytes(z.read(src.name))
        rgbs={s:original(p,s) for s in ("A","B")}
        for v in ["O",*FAMILIES["T"]]:
            for s,rgb in rgbs.items():
                im,_=make_view(rgb,v)
                assert hashlib.sha256(im.tobytes()).hexdigest()==bank[i,s,v]["pixel_sha256"]
                final=final_rgb(tf,im)
                base=final_rgb(tf,rgb)
                assert final.shape==(224,224,3)
                assert np.isclose(np.abs(final-base).mean(dtype=np.float64)*255,float(bank[i,s,v]["input_mean_abs_RGB255"]),atol=1e-7)
                file=f"{i}-{s}-{v}.png";Image.fromarray(np.uint8(np.rint(255*np.clip(final,0,1)))).save(assets/file)
        for b,t,x in ev["BT"]:
            details.append(dict(pair_id=i,method="BT",brightness=b,direction=t,d0=d["O"],margin=x,pick=choice(x),human=human))
        for t in FAMILIES["T"]:
            x=(d["O"]+d[t])/2
            details.append(dict(pair_id=i,method="O+T",brightness="—",direction=t,d0=d["O"],margin=x,pick=choice(x),human=human))
    write_csv(ROOT/"old_case_checks.csv",details)
    print("Old 64: five changed pairs:",order,"003 TT 4/6 BT 4/16 corrections; 008 BT 16/16 harms; 50 model inputs verified.")
if __name__=="__main__": main()

