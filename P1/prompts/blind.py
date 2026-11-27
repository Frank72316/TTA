"""Generate only from the frozen manifest, with no access to model scores or labels."""
import json
import random
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED
from common import ROOT, frozen, images, write_csv, read_csv

def main():
    folder=ROOT/"human_eval"
    assets=folder/"assets"
    assets.mkdir(parents=True,exist_ok=True)
    prompts={p["pair_id"]:p for p in frozen()}
    rng=random.Random(42)
    ids=rng.sample(sorted(prompts),16)
    order=[(i,c,l) for i in ids for c in ("Q0","Q1") for l in ("L0","L1")]
    rng.shuffle(order)
    mapping=[]
    page=[]
    files={}
    for i in ids:
        for (s,l),img in zip([("A","L0"),("A","L1"),("B","L0"),("B","L1")],images(i)):
            name=hashlib.sha256(img.tobytes()).hexdigest()[:24]+".png"
            files[i,s,l]="assets/"+name
            img.save(assets/name)
    for k,(i,c,l) in enumerate(order,1):
        left,right=("A","B") if rng.getrandbits(1) else ("B","A")
        qid=f"T{k:02}"
        mapping.append(dict(question_id=qid,pair_id=i,condition=c,light=l,left_side=left,right_side=right))
        page.append(dict(id=qid,prompt=prompts[i][c],left=files[i,left,l],right=files[i,right,l]))
    assert len({(m["pair_id"],m["condition"],m["light"]) for m in mapping})==64
    write_csv(folder/"mapping_private.csv",mapping)
    blank=folder/"annotations_blank.csv"
    if not blank.exists():
        write_csv(blank,[dict(question_id=m["question_id"],choice="") for m in mapping])
    markup=(ROOT/"blind_template.html").read_text(encoding="utf-8")
    payload=json.dumps(page,ensure_ascii=False).replace("<","\\u003c")
    key="tta-blind-"+hashlib.sha256(payload.encode()).hexdigest()[:14]
    (folder/"index.html").write_text(markup.replace("__STORAGE__",key).replace("__DATA__",payload),encoding="utf-8")
    with ZipFile(folder/"participant.zip","w",compression=ZIP_DEFLATED,compresslevel=1) as archive:
        archive.write(folder/"index.html","index.html")
        archive.writestr("annotations_blank.csv","question_id,choice\r\n"+"".join(m["question_id"]+",\r\n" for m in mapping))
        for name in sorted(set(files.values())):
            archive.write(folder/name,name)
    print(len(ids),"pairs,",len(page),"questions,",len(set(files.values())),"lossless images")

if __name__=="__main__":
    main()

