"""Extract only official HDF5 scene identifiers; the default uses HTTP byte ranges."""
import argparse
import csv
import tempfile
import sys
from pathlib import Path
from zipfile import ZipFile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"depth_adaptive_budget"))
from scene_metadata import URL,RangeFile,read_scenes

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--mat",type=Path)
    a=p.parse_args()
    with ZipFile(a.archive) as z: names=sorted(n for n in z.namelist() if "/images/val/" in n and n.endswith(".jpg"))
    assert len(names)==654
    with tempfile.TemporaryDirectory() as t:
        dest=Path(t);manifest=dest/"manifest.csv"
        with manifest.open("w",encoding="utf8",newline="") as f:
            w=csv.writer(f);w.writerow(["frame"]);w.writerows([[n] for n in names])
        read_scenes(a.mat or RangeFile(URL),manifest,dest/"scenes.csv")
        out=HERE/"official_test_scenes.csv";data=(dest/"scenes.csv").read_bytes()
        if out.exists():assert out.read_bytes()==data
        else:out.write_bytes(data)
