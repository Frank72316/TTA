"""Two sign-selected cases, using the same metric depth cache and boundary mask."""
import argparse,csv,json
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from measure import HERE,OLD,sha,zones
def plot(archive,root):
    stats=json.loads((HERE/"summary.json").read_text())
    entries={r["frame_id"]:r for r in csv.DictReader((OLD/"frame_results.csv").open())}
    ours={r["frame_id"]:r for r in csv.DictReader((HERE/"frame_results.csv").open())}
    cache=root/sha(OLD/"environment.json")
    fig,axes=plt.subplots(2,3,figsize=(9.2,4.9),layout="constrained")
    with ZipFile(archive) as z:
        for i,kind in enumerate(("positive","negative")):
            fid=stats["cases"][kind]["frame_id"];row=entries[fid]
            with Image.open(BytesIO(z.read(row["truth"]))) as im: gt=np.asarray(im,dtype=np.float32)/1000
            s=(gt.shape[1]*3+50)//100;gt=gt[:,s:-s];valid=(gt>0)&(gt<=10)
            pred=[]
            for v in "OFLR":
                with np.load(cache/(fid+"-"+v+".npz")) as f:pred.append(f["p"].astype(float))
            p2=np.mean(pred[:2],axis=0);p4=np.mean(pred,axis=0);ps=p2+float(ours[fid]["shift_m"])
            gain=(ps-gt)**2-(p4-gt)**2
            cols=[(gt,"Depth (m)","viridis",0,8),(zones(gt.astype(float),valid)["boundary"],"Depth boundary band","Greys",0,1),
                  (gain,"Spatial MSE gain (m²)","RdBu",-.1,.1)]
            for j,(img,title,cmap,lo,hi) in enumerate(cols):
                ax=axes[i,j];im=ax.imshow(img,cmap=cmap,vmin=lo,vmax=hi,interpolation="nearest")
                if i==0:ax.set_title(title)
                ax.set_xticks([]);ax.set_yticks([])
                if j==0:ax.set_ylabel(fid)
                if i==1 and j!=1:fig.colorbar(im,ax=axes[:,j],orientation="horizontal",shrink=.78,extend="both")
    fig.savefig(HERE/"case_regions.png",dpi=170);plt.close(fig)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--zip",type=Path,required=True);ap.add_argument("--cache",type=Path,required=True)
    a=ap.parse_args();plot(a.zip,a.cache)
