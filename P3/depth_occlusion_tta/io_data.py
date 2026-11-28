"""Data and coordinates shared by the independent GPU and CPU pipelines."""
import csv,hashlib,json,os,re
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
BASE=Path(os.environ.get("OCC_DATA","/home/data/users/shenc/tta_occlusion_data"))
MODEL=Path(os.environ.get("OCC_MODEL","/home/shenc/tta_depth_adaptive/data/model"))
SCALE=(1,1.25,1.5,1.75,2,2.25,2.5)
def sha(p,alg="sha256"):
    h=hashlib.new(alg)
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(".tmp");t.write_bytes((json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode());t.replace(p)
def get(p):return json.loads(Path(p).read_text(encoding="utf8"))
def read(p):
    with open(p,encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in rows for k in r))
    t=p.with_suffix(".tmp")
    with t.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,cols,lineterminator="\n");w.writeheader();w.writerows(rows)
    t.replace(p)
def paths(r):
    if r["dataset"]=="middlebury":
        d=BASE/"midd"/r["id"];return d/"im0.png",d/"disp0.pfm",d/"calib.txt"
    d=BASE/"ibims_hd"/r["id"];return d/"rgb.png",d/"depth_m.npy",d/"metadata.json"
def rgb(r):
    with Image.open(paths(r)[0]) as im:return im.convert("RGB")
def pfm(path):
    with open(path,"rb") as f:
        assert f.readline().strip()==b"Pf"
        w,h=map(int,f.readline().split());scale=float(f.readline())
        assert scale!=0
        arr=np.fromfile(f,dtype="<f4" if scale<0 else ">f4")
    assert arr.size==w*h
    return np.ascontiguousarray(arr.reshape(h,w)[::-1],dtype="float32")
def depth(r):
    _,p,c=paths(r)
    if r["dataset"]=="middlebury":
        v=dict(line.split("=",1) for line in c.read_text().splitlines() if "=" in line)
        focal=float(v["cam0"].lstrip("[").split()[0]);offset=float(v["doffs"]);base=float(v["baseline"])
        a=pfm(p);den=a+offset
        assert a.shape==(int(v["height"]),int(v["width"])) and min(base,focal)>0
        z=np.full(a.shape,np.nan,np.float32)
        use=np.isfinite(a)&(den>0);z[use]=.001*base*focal/den[use]
    else:
        meta=get(c);assert meta["units"]=="m" and meta["coordinates"]=="optical_axis" and meta["verified_registration"]
        z=np.load(p).astype(np.float32);assert z.shape==(1000,1500)
        for name in ("mask_transp","mask_invalid"):
            mask=p.parent/(name+".png")
            if mask.exists():z[np.asarray(Image.open(mask))==0]=np.nan
    return z
def rounded(x):return int(np.floor(x+.5))
def boxes(w,h):
    cw,ch=rounded(.6*w),rounded(.6*h)
    return [(x,y,x+cw,y+ch) for y in (0,h-ch) for x in (0,w-cw)]
def degrade(rgb,hw):
    import torch
    import torch.nn.functional as F
    im=np.array(rgb);t=torch.from_numpy(im).permute(2,0,1)[None].float()
    small=F.interpolate(t,size=hw,mode="area")
    big=F.interpolate(small,size=im.shape[:2],mode="bilinear",align_corners=False)
    return Image.fromarray(big[0].permute(1,2,0).clamp(0,255).add(.5).floor().byte().numpy())
def work(r):return BASE/"cache"/r["dataset"]/r["id"]
def named(r):return r["dataset"]+"/"+r["id"]
def official_inputs():
    ib=[s.strip() for s in (BASE/"ibims/imagelist.txt").read_text().splitlines() if s.strip()]
    mb=[s.strip().split("/")[-1].removesuffix("-perfect") for s in (BASE/"midd/list.txt").read_text().splitlines() if s.strip()]
    assert len(set(ib))==100 and len(set(mb))==23
    rows=[dict(dataset=ds,id=s) for ds,ls in (("ibims",ib),("middlebury",mb)) for s in ls]
    return sorted(rows,key=lambda r:hashlib.sha256(named(r).encode()).digest())
if __name__=="__main__":
    rows=official_inputs();dst=HERE/"expected.csv"
    if dst.exists():assert rows==read(dst)
    else:write(dst,rows)
    save(HERE/"freeze.json",dict(frozen_local_time="2026-11-28T21:32:15+08:00",protocol_sha256=sha(HERE/"protocol.md"),
        full_spec_sha256=sha(HERE/"design_spec.md"),expected_sha256=sha(dst),counts={"ibims":100,"middlebury":23},
        seed=2026,base_commit="d777ac372cefc35e4ea9a57f5bd22f478d9ec6d8"))
