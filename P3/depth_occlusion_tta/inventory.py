"""Audit the released files without reading model output. Record every expected image and source checksum."""
import io
from collections import Counter
from zipfile import ZipFile
from scipy.io import loadmat
from io_data import *

def main():
    d=BASE/"ibims"
    checks={f:h for h,f in (line.split() for line in (d/"Ibims_Dataset.sha512").read_text().splitlines())}
    for f,h in checks.items():assert sha(d/f,"sha512")==h
    counts={}
    rows=[];disps=[]
    with ZipFile(d/"ibims1_core_raw.zip") as raw,ZipFile(d/"ibims1_core_mat.zip") as mat:
        for r in read(HERE/"expected.csv"):
            a=dict(**r)
            if r["dataset"]=="ibims":
                ident=r["id"]
                key="ibims1_core_raw/{}/"+ident+".png"
                vals=loadmat(io.BytesIO(mat.read("ibims1_core_mat/"+ident+".mat")),simplify_cells=True)["data"]
                for k in ("rgb","depth","mask_invalid","mask_transp","mask_floor","mask_wall","mask_table","edges"):
                    name=key.format(k)
                    if name in raw.namelist():
                        with raw.open(name) as f:im=Image.open(f);a["raw_"+k]="x".join(map(str,im.size))
                    ar=np.asarray(vals.get(k,[]))
                    if ar.ndim>=2 and ar.size:a["mat_"+k]="x".join(map(str,ar.shape[1::-1]))
                a["mat_unit"]="m";a["ready"]=int(all(p.exists() for p in paths(r)))
                a["reason"]="" if a["ready"] else "official core RGB and laser GT at VGA; registered HD pair unavailable"
            else:
                im,gt,cal=paths(r)
                with gt.open("rb") as ff:header=[ff.readline().decode().strip() for _ in range(3)]
                arr=pfm(gt);vals=arr[np.isfinite(arr)];z=depth(r);use=np.isfinite(z)&(z>0)
                v=dict(line.split("=",1) for line in cal.read_text().splitlines() if "=" in line)
                with Image.open(im) as pic:shape=pic.size
                assert shape==arr.shape[::-1]
                a.update(ready=1,raw_rgb="x".join(map(str,shape)),raw_depth="x".join(map(str,shape)),
                         rgb_sha256=sha(im),gt_sha256=sha(gt),calib_sha256=sha(cal),reason="")
                disps.append(dict(id=r["id"],header=header[-1],endian="little" if float(header[-1])<0 else "big",d_min=float(vals.min()),
                          d_max=float(vals.max()),calib_vmin=v["vmin"],calib_vmax=v["vmax"],median_Z_m=float(np.median(z[use])),
                          baseline_mm=v["baseline"],f_px=v["cam0"].split()[0][1:],doffs_px=v["doffs"]))
            rows.append(a)
    for archive in ("ibims1_aux_augmented.zip","ibims1_aux_mvs.zip","ibims1_aux_patterns.zip"):
        with ZipFile(d/archive) as zz:
            shapes=Counter()
            for n in zz.namelist():
                if n.lower().endswith((".png",".jpg",".jpeg")):
                    with zz.open(n) as b:im=Image.open(b);shapes[(Path(n).suffix.lower(),im.size)]+=1
            counts[archive]={str(k):v for k,v in shapes.items()}
    mirror=d/"mirror.zip"
    if mirror.exists():
        with ZipFile(mirror) as zz:
            shapes=Counter()
            for n in zz.namelist():
                if n.endswith(("image.jpg","depth.png")):
                    with zz.open(n) as b:shapes[(Path(n).name,Image.open(b).size)]+=1
            counts["Ruicheng/monocular-geometry-evaluation iBims-1.zip"]={str(k):v for k,v in shapes.items()}
    write(HERE/"inventory.csv",rows);write(HERE/"disparity_units.csv",disps)
    save(HERE/"data_audit.json",dict(official_sha512=checks,verified=True,auxiliary_and_mirror_shapes=counts,
          official_hd_spec="https://www.asg.ed.tum.de/en/lmf/ibims1/",public_deposit="https://mediatum.ub.tum.de/1455541",
          external="https://vision.middlebury.edu/stereo/data/scenes2014/",middlebury_unit="optical-axis metres from float pixel disparity",
          ibims_ready=sum(r["ready"] for r in rows if r["dataset"]=="ibims")))
if __name__=="__main__":main()
