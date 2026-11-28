import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
if __name__=="__main__":
    with (HERE/"pair_results.csv").open() as f:rows=[r for r in csv.DictReader(f) if 0<int(r["J_correct"])<4]
    fig,ax=plt.subplots(figsize=(7,3.4),layout="constrained")
    for j,(p,c) in enumerate(zip(("00","40","04","44"),("#087e71","#437bb1","#d77a23","#784795"))):
        ax.scatter([float(r["aggregate_J"+p]) for r in rows],np.arange(len(rows))+.085*(j-1.5),
                   label=f"({p[0]}, {p[1]})",color=c,s=37)
    ax.axvline(0,color="black",lw=.85);ax.set_yticks(range(len(rows)),[r["pair_id"] for r in rows]);ax.invert_yaxis()
    ax.set_xlabel("Human-winner margin after averaging original and JPEG scores")
    ax.legend(title="Source-pixel phase",ncol=4,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.23))
    fig.savefig(HERE/"phase_crossings.png",dpi=180);plt.close(fig)
