"""Fetch official single-exposure FullSize two-view archives and audit RGB identity, without predictions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import subprocess
import sys
import zipfile
from PIL import ImageDraw, ImageOps
from scipy.fft import dctn
from data import *

def fetch(r):
    if r["year"]=="2014":
        return
    d=sources(r)[0].parent
    d.mkdir(parents=True,exist_ok=True)
    url=f'https://vision.middlebury.edu/stereo/data/scenes{r["year"]}/FullSize/zip-2views/{r["id"]}-2views.zip'
    p=d/"official.zip"
    if not p.exists():
        tmp=d/"official.part"
        subprocess.run(["curl","-fsSL","--retry","4","--connect-timeout","15","--max-time","480","-o",str(tmp),url],check=True)
        tmp.replace(p)
    with zipfile.ZipFile(p) as archive:
        for name in ("view1.png","disp1.png","dmin.txt","view5.png","disp5.png"):
            match=[n for n in archive.namelist() if n.split("/")[-1]==name]
            assert len(match)==1,(p,name,match)
            dst=d/name
            if not dst.is_file():
                dst.write_bytes(archive.read(match[0]))

def audit():
    rows=all_rows()
    pic=Image.new("RGB",(900,((len(rows)+5)//6)*110),"white")
    pen=ImageDraw.Draw(pic)
    features={}
    for i,r in enumerate(sorted(rows,key=key)):
        p=sources(r)[0]; im=Image.open(p).convert("RGB")
        thumb=ImageOps.contain(im,(145,85))
        xx=(i%6)*150; yy=(i//6)*110
        pic.paste(thumb,(xx,yy));pen.text((xx,yy+85),key(r),fill="black")
        patch=np.asarray(im.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=float)
        co=dctn(patch,type=2,norm="ortho")[:8,:8].ravel()[1:]
        features[key(r)]=(co>np.median(co),patch)
    pic.save(ROOT/"scene_audit.jpg",quality=88)
    pairs=[]
    for a,b in itertools.combinations(sorted(features),2):
        if a[:4]==b[:4]:continue
        fa,pa=features[a];fb,pb=features[b]
        pairs.append(dict(scene1=a,scene2=b,phash_hamming=int(np.sum(fa!=fb)),
                    rgb_luma_correlation=float(np.corrcoef(pa.ravel(),pb.ravel())[0,1])))
    write(ROOT/"cross_year_audit.csv",sorted(pairs,key=lambda x:x["phash_hamming"]))

def freeze():
    assert not (ROOT/"freeze.json").exists()
    rows=all_rows()
    for r in rows:
        im,z,c=sources(r)
        r.update(include=1,exclusion="",rgb_sha256=sha(im),depth_sha256=sha(z),calib_sha256=sha(c))
    assert len({r["rgb_sha256"] for r in rows})==len(rows)==50
    write(ROOT/"scenes.csv",rows)
    save(ROOT/"freeze.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),base_commit="aba9d08bc57f5424c73b499d616ff33ccee40844",
        protocol_sha256=sha(ROOT/"protocol.md"),scenes_sha256=sha(ROOT/"scenes.csv"),duplicate_check=sha(ROOT/"cross_year_audit.csv"),
        registered_plan={"2014":23,"2005":6,"2006":21},seed=2026,draws=10000,
        legacy_files={f:sha(LEGACY/f) for f in ("io_data.py","predict.py","edges.py","evaluate.py","per_image.csv","cache_index.csv","calibration.json")}))

if __name__=="__main__":
    if sys.argv[1]=="fetch":
        with ThreadPoolExecutor(8) as pool:list(pool.map(fetch,all_rows()))
        audit()
    elif sys.argv[1]=="freeze":freeze()
