"""Plot the two preselected extreme frames on identical signed color scales."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from metrics import HERE,OLD,VIEWS,sha,read,fields,step,align,valid_truth,evaluate

def plot(out,archive):
    assert sha(archive)==json.loads((OLD/"environment.json").read_text())["archive_sha"]
    cases=json.loads((out/"validation.json").read_text())["extreme_frames"]
    rows={r["frame_id"]:r for r in read(out/"frame_results.csv")};panels=[]
    with ZipFile(archive) as z:
        for name in ("max","min"):
            fid=cases[name];r=rows[fid]
            with Image.open(BytesIO(z.read(r["frame"]))) as f:rgb=np.asarray(f.convert("RGB"))
            with Image.open(BytesIO(z.read(r["truth"]))) as f:g,valid=valid_truth(np.asarray(f))
            with np.load(out/"predictions"/(fid+".npz"),allow_pickle=False) as f:
                assert str(f["signature"])==sha(out/"environment.json")
                p={v:f[v] for v in VIEWS}
            verify=evaluate(p,g,valid);assert verify["G"]==float(r["G"])
            maps=[np.ma.masked_where(~valid,x) for x in fields(p)]
            panels.append((r,[align(rgb,0,step(rgb.shape[1],4)),*maps]))
    # Identical normalization within physical units across both rows; only display saturation.
    m=float(np.quantile(np.concatenate([abs(tile.compressed()) for _,pics in panels for tile in pics[1:3]]),.99))
    m2=float(np.quantile(np.concatenate([abs(pics[3].compressed()) for _,pics in panels]),.99))
    cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#777777")
    fig=plt.figure(figsize=(13,6.6),layout="constrained")
    grid=fig.add_gridspec(3,4,height_ratios=[1,1,.08])
    images={}
    for i,(r,pics) in enumerate(panels):
        for j,arr in enumerate(pics):
            ax=fig.add_subplot(grid[i,j])
            if j==0:ax.imshow(arr)
            else:
                limit=m if j<=2 else m2
                images[j]=ax.imshow(arr,cmap=cmap,vmin=-limit,vmax=limit,interpolation="nearest")
            ax.set(xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
            if i==0:ax.set_title(["RGB • common field","D2 = L2 − R2","D4 = L4 − R4","Contribution = D2 × D4 / 9"][j],fontsize=12,pad=10)
            if j==0:
                label="Highest G" if i==0 else "Lowest G"
                ax.set_xlabel(f"{label}  ·  {r['frame_id']}  ·  {r['scene']}\nG = {float(r['G']):+.6f} m²",fontsize=10)
    for j,span,cap in [(1,grid[2,1:3],"Aligned depth difference (m)"),(3,grid[2,3],"Pixel contribution (m²)")]:
        bar=fig.colorbar(images[j],cax=fig.add_subplot(span),orientation="horizontal",extend="both")
        bar.set_label(cap);bar.locator=plt.MaxNLocator(nbins=5);bar.update_ticks()
    fig.savefig(out/"cases.png",dpi=155,bbox_inches="tight",facecolor="white");plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE);p.add_argument("--archive",type=Path,required=True)
    a=p.parse_args();plot(a.out,a.archive)
