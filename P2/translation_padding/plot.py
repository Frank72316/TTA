import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
HERE=Path(__file__).resolve().parent
if __name__=="__main__":
    selected=json.loads((HERE/"summary.json").read_text())["examples"]
    fig,axes=plt.subplots(2,2,figsize=(5,4.7),layout="constrained")
    with np.load(HERE/"input_masks.npz") as bank:
        for j,(key,r) in enumerate(selected.items()):
            direction=("left","right","up","down")[r["combination"]]
            for i,side in enumerate("AB"):
                a=np.unpackbits(bank[f"{r['pair_id']}_{side}_{direction}"]).reshape(224,224)
                ax=axes[i,j];ax.imshow(a,cmap=ListedColormap(["#ededeb","#007569"]),vmin=0,vmax=1,interpolation="nearest")
                ax.set_title(f"{r['pair_id']} {direction} · {side}  ({100*a.mean():.2f}%)",fontsize=10);ax.axis("off")
    fig.suptitle("Pixels changed by replacing reflection with edge padding")
    fig.savefig(HERE/"case_masks.png",dpi=165);plt.close(fig)
