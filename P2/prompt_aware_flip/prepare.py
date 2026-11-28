"""Select on COCO annotations, acquire originals, export the pre-score box review."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
from random import Random
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image, ImageDraw
from common import ANNOTATION_SHA, BUCKET, HERE, SEED, save_json, save_rows, sha, valid_rect


def get(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "TTA-COCO-research/1.0"}), timeout=45) as r:
                return r.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def target_box(ann, w, h, frame):
    ox, oy, nw, nh = frame
    x, y, bw, bh = ann["bbox"]
    if min(bw,bh) <= 0:
        return None
    x1,y1,x2,y2 = max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
    visible = max(0,x2-x1)*max(0,y2-y1)/(bw*bh)
    if visible < .8:
        return None
    return dict(ann=ann["id"], original=ann["bbox"], visible=visible,
                box=[ox+x1*nw/w, oy+y1*nh/h, ox+x2*nw/w, oy+y2*nh/h])


def select(doc):
    rng=Random(SEED)
    by_image=defaultdict(list)
    for a in doc["annotations"]:
        by_image[a["image_id"]].append(a)
    names={c["id"]:c["name"] for c in doc["categories"]}
    photos={v["id"]:v for v in doc["images"]}
    pools=defaultdict(lambda: [[],[]])
    for image_id in sorted(photos):
        p=photos[image_id]
        w,h=p["width"],p["height"]
        frame=valid_rect(w,h)
        if not frame:
            continue
        anns=by_image[image_id]
        counts=Counter(a["category_id"] for a in anns)
        targets={a["category_id"]:target_box(a,w,h,frame) for a in anns if counts[a["category_id"]]==1 and not a["iscrowd"]}
        for x,y in combinations(sorted(k for k in targets if targets[k]),2):
            a,b=targets[x],targets[y]
            ax,_,ax2,_=a["box"]
            bx,_,bx2,_=b["box"]
            sep=abs((ax+ax2-bx-bx2)/2)/frame[2]
            if sep<.2 or (ax2>bx and bx2>ax):
                continue
            direction=int(ax>bx)
            pools[x,y][direction].append((image_id,a,b,sep))
    keys=[k for k in sorted(pools) if all(pools[k])]
    rng.shuffle(keys)
    for k in keys:
        for bucket in pools[k]:
            rng.shuffle(bucket)
    frozen=[]; sources=[]; used=set()
    while len(frozen)<64:
        start=len(frozen)
        for key in keys:
            buckets=pools[key]
            for bucket in buckets:
                while bucket and bucket[-1][0] in used:
                    bucket.pop()
            if not all(buckets):
                continue
            # Randomise which noun is the subject and which photo is candidate A.
            swap_subject=bool(rng.getrandbits(1))
            x,y=key[::-1] if swap_subject else key
            a_left=bool(rng.getrandbits(1))
            picks=[b.pop() for b in buckets]
            if swap_subject!= (not a_left):
                picks.reverse()
            pid=f"F{len(frozen)+1:03}"
            verb="are" if names[x] in {"scissors","skis"} else "is"
            q=f"The {names[x]} {verb} to the LEFT of the {names[y]}."
            pair=dict(pair_id=pid,x=names[x],y=names[y],x_cat=x,y_cat=y,
                      prompt_left=q.replace("LEFT","left"),prompt_right=q.replace("LEFT","right"),
                      winner_left="A" if a_left else "B",winner_right="B" if a_left else "A")
            for side,(iid,first,second,separation) in zip("AB",picks):
                assert iid not in used
                used.add(iid)
                bx,by=(second,first) if swap_subject else (first,second)
                assert (bx["box"][2]<=by["box"][0])==(pair["winner_left"]==side)
                p=photos[iid]
                pair["image_"+side]=iid
                sources.append(dict(pair_id=pid,side=side,image_id=iid,file=p["file_name"],width=p["width"],height=p["height"],
                    valid_xywh=json.dumps(valid_rect(p["width"],p["height"])),x_ann=bx["ann"],y_ann=by["ann"],
                    x_original_xywh=json.dumps(bx["original"]),y_original_xywh=json.dumps(by["original"]),
                    x_input_xyxy=json.dumps(bx["box"]),y_input_xyxy=json.dumps(by["box"]),x_visible=bx["visible"],
                    y_visible=by["visible"],center_gap_fraction=separation,license_id=p["license"],
                    source=p["flickr_url"],url=BUCKET+"/val2017/"+p["file_name"]))
            frozen.append(pair)
            if len(frozen)==64:
                break
        if len(frozen)==start:
            break
    assert frozen
    return frozen,sources,len(keys)


def prepare(data,out):
    data.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    annot=data/"instances_val2017.json"
    if not annot.exists():
        with ZipFile(BytesIO(get(BUCKET+"/annotations/annotations_trainval2017.zip"))) as z:
            annot.write_bytes(z.read("annotations/instances_val2017.json"))
    assert sha(annot.read_bytes())==ANNOTATION_SHA
    doc=json.loads(annot.read_bytes())
    pairs,sources,buckets=select(doc)
    folder=data/"val2017"
    folder.mkdir(exist_ok=True)
    def acquire(src):
        path=folder/src["file"]
        if not path.is_file():
            path.write_bytes(get(src["url"]))
        with Image.open(path) as im:
            assert im.size==(src["width"],src["height"])
            im.load()
        src["jpeg_sha256"]=sha(path.read_bytes())
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(acquire,sources))
    save_rows(out/"pairs.csv",pairs)
    save_rows(out/"images.csv",sources)
    local=data/"box_previews"; local.mkdir(exist_ok=True)
    for i,p in enumerate(pairs):
        sheet=Image.new("RGB",(448,224))
        for j,s in enumerate(sources[2*i:2*i+2]):
            ox,oy,w,h=json.loads(s["valid_xywh"])
            with Image.open(folder/s["file"]) as im:
                tile=Image.new("RGB",(224,224))
                tile.paste(im.convert("RGB").resize((w,h),Image.Resampling.BICUBIC),(ox,oy))
            draw=ImageDraw.Draw(tile)
            for key,color in (("x","#26baff"),("y","#ff9822")):
                x1,y1,x2,y2=json.loads(s[key+"_input_xyxy"])
                draw.rectangle((x1,y1,x2,y2),outline=color,width=2)
            sheet.paste(tile,(224*j,0))
        sheet.save(local/(p["pair_id"]+".jpg"),quality=94)
    save_json(out/"freeze.json",dict(seed=SEED,source="COCO 2017 validation instances",annotation_sha256=ANNOTATION_SHA,
              bidirectional_category_pairs=buckets,selected=len(pairs),photos=len(sources),
              category_pairs=len({tuple(sorted((p["x_cat"],p["y_cat"]))) for p in pairs}),
              licenses=doc["licenses"],hashes={name:sha((out/name).read_bytes()) for name in ("pairs.csv","images.csv")}))
    print("Frozen",len(pairs),"pairs,",len(sources),"originals",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=HERE)
    a=ap.parse_args(); prepare(a.data,a.out)
