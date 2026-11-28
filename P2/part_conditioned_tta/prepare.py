"""Verify and unpack only the frozen photographs and necessary CUB annotations."""
import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
MD5 = "97eceeb196236b17998738112f37df78"

def digest(path, alg="sha256"):
    h = hashlib.new(alg)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write(path, records):
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0])
        w.writeheader()
        w.writerows(records)

def as_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf8")

def check_packet():
    for line in (ROOT/"SHA256SUMS").read_text(encoding="utf8").splitlines():
        h, n = line.split("  ")
        assert digest(ROOT/n) == h, n

def extract(archive, data, rows):
    assert digest(archive, "md5") == MD5, "Caltech archive MD5 mismatch"
    files = {"images/"+r["path"] for r in rows}
    ids = {r["image_id"].encode() for r in rows}
    needed = {"images.txt", "image_class_labels.txt", "train_test_split.txt", "attributes.txt",
              "parts.txt", "part_locs.txt", "image_attribute_labels.txt"}
    data.mkdir(exist_ok=True, parents=True)
    meta = data/"meta"
    meta.mkdir(exist_ok=True)
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            if not item.isfile(): continue
            name = item.name.removeprefix("./")
            if name.startswith("CUB_200_2011/"): name = name[len("CUB_200_2011/"):]
            short = name.rsplit("/",1)[-1]
            if name in files:
                dest = data/name
                dest.parent.mkdir(exist_ok=True, parents=True)
                with dest.open("wb") as f: shutil.copyfileobj(tar.extractfile(item),f)
            elif short in needed:
                src = tar.extractfile(item)
                with (meta/short).open("wb") as f:
                    if short == "image_attribute_labels.txt":
                        for line in src:
                            if line.split(None,1)[0] in ids: f.write(line)
                    else: shutil.copyfileobj(src,f)
    return {"md5":MD5,"sha256":digest(archive),"bytes":archive.stat().st_size}

def tokens(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]

def verify(data, rows, archive_info):
    from PIL import Image
    m = data/"meta"
    names = dict(tokens(m/"images.txt"))
    species = dict(tokens(m/"image_class_labels.txt"))
    split = dict(tokens(m/"train_test_split.txt"))
    parts = {s[0]:" ".join(s[1:]) for s in tokens(m/"parts.txt")}
    assert [parts[k] for k in ("5","9","13")] == ["crown","left wing","right wing"]
    loc = {(i,k):(float(x),float(y)) for i,k,x,y,v in tokens(m/"part_locs.txt") if v=="1"}
    attr = {label:i for i,label in tokens(m/"attributes.txt")}
    facts = {(s[0],s[1]):(s[2],s[3]) for s in tokens(m/"image_attribute_labels.txt")}
    records = []
    pairs = {r["pair_id"] for r in rows}
    assert len(pairs)==96 and len(rows)==len({r["image_id"] for r in rows})==len({r["path"] for r in rows})==192
    assert len({r["species"] for r in rows})==76 and len({r["color"] for r in rows})==9
    definite = 0
    for pid in pairs:
        a,b = sorted((r for r in rows if r["pair_id"]==pid),key=lambda r:r["side"])
        assert (a["side"],b["side"])==("A","B") and a["species"]==b["species"] and a["color"]==b["color"]
        assert a["crown_present"]==b["wing_present"]!=a["wing_present"]==b["crown_present"]
        definite += all(r[q+"_certainty"]=="4" for r in (a,b) for q in ("crown","wing"))
    assert definite==29
    for r in rows:
        i = r["image_id"]
        assert (r["path"],r["species"],"0")== (names[i],species[i],split[i])
        k = "9" if (i,"9") in loc else "13"
        assert k == r["wing_id"]
        for q,p in [("crown","5"),("wing",k)]:
            assert loc[i,p]==(float(r[q+"_x"]),float(r[q+"_y"]))
            assert facts[i,attr["has_"+q+"_color::"+r["color"]]]==(r[q+"_present"],r[q+"_certainty"])
        photo = data/"images"/r["path"]
        with Image.open(photo) as im:
            assert im.format=="JPEG"; w,h=im.size; im.verify()
        records.append(dict(pair_id=r["pair_id"],side=r["side"],image_id=i,path=r["path"],width=w,height=h,
                            jpeg_sha256=digest(photo),color=r["color"],crown_x=r["crown_x"],crown_y=r["crown_y"],
                            wing_id=k,wing_x=r["wing_x"],wing_y=r["wing_y"]))
    write(ROOT/"inputs.csv", records)
    as_json(ROOT/"data_audit.json",dict(archive=archive_info,source="https://data.caltech.edu/records/65de6-vp158",
            packet_sha256=digest(ROOT/"SHA256SUMS"),pairs_sha256=digest(ROOT/"pairs.csv"),inputs_sha256=digest(ROOT/"inputs.csv"),
            pairs=len(pairs),photos=len(records),species=76,colors=9,definitely_pairs=definite,
            wing_id_counts=dict(Counter(r["wing_id"] for r in records)),official_labels_verified=2*len(records),
            metadata_sha256={p.name:digest(p) for p in sorted(m.glob("*.txt"))}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    a=p.parse_args();check_packet();rows=read(ROOT/"pairs.csv")
    verify(a.data,rows,extract(a.archive,a.data,rows))
