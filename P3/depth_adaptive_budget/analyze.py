"""Scene-cluster bootstrap with the scheduling decision recomputed in each batch."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from freeze import HERE, csv_bytes, sha


def read(name):
    with (HERE/name).open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def weights_half(values, ids):
    order=np.lexsort((ids,-values))
    weight=np.zeros(len(order))
    weight[order[:len(order)//2]]=1
    # Odd cluster bootstrap batches: fair coin at the cutoff in expectation.
    if len(order)%2:
        weight[order[len(order)//2]]=.5
    return weight


def run():
    rows=read("measurements.csv"); scene_rows=read("scenes.csv"); freeze_rows=read("frames.csv")
    scenes={r["frame"]:r["scene"] for r in scene_rows}
    assert len(scenes)==len(scene_rows)==len(rows)==128
    assert [r["frame"] for r in rows]==[r["frame"] for r in freeze_rows]
    env=json.loads((HERE/"environment.json").read_text())
    assert sha(HERE/"frames.csv")==env["frames_sha"]
    log=json.loads((HERE/"runs.json").read_text())
    assert log[-1]["schedule_sha"]==sha(HERE/"schedule.csv")
    ids=np.array([r["frame_id"] for r in rows]); scene=np.array([scenes[r["frame"]] for r in rows])
    arr=lambda k:np.array([float(r[k]) for r in rows])
    D,g,m2,m4=map(arr,("D","G","MSE2","MSE4"))
    assert np.array_equal(m2-m4,g) and np.isfinite([D,g,m2,m4]).all()
    schedule=read("schedule.csv"); mask=weights_half(D,ids)
    assert len(set(ids))==128 and mask.sum()==64 and np.array_equal(mask,arr("selected"))
    assert len(schedule)==128 and all((x["frame_id"],x["D"],x["rank_D"],x["selected"])==
                  (y["frame_id"],y["D"],y["rank_D"],y["selected"]) for x,y in zip(schedule,rows))
    a=(arr("MSE3L")+arr("MSE3R"))/2; b=(m2+m4)/2; c=np.where(mask,m4,m2)
    oracle_mask=weights_half(g,ids)
    oracle=np.where(oracle_mask,m4,m2)
    for j,r in enumerate(rows):
        r.update(scene=scene[j],oracle_selected=int(oracle_mask[j]),MSE_A=a[j],MSE_B=b[j],
                 MSE_C=c[j],MSE_oracle=oracle[j])
    (HERE/"frame_results.csv").write_bytes(csv_bytes(rows))
    strategies=dict(two=m2,A=a,B=b,C=c,four=m4,oracle=oracle)
    clusters=[np.flatnonzero(scene==s) for s in sorted(set(scene))]
    rng=np.random.default_rng(20261128); draws=[]; odd=0
    for _ in range(10000):
        idx=np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])
        odd+=len(idx)%2
        allocation=weights_half(D[idx],ids[idx])
        best=weights_half(g[idx],ids[idx])
        scheduled=m2[idx]-allocation*g[idx]
        upper=m2[idx]-best*g[idx]
        draws.append([(b[idx]-scheduled).mean(),(a[idx]-scheduled).mean(),(scheduled-upper).mean()])
    intervals=np.quantile(draws,[.025,.975],axis=0).T
    comps=dict(zip(("B_minus_C","A_minus_C","C_minus_oracle"),
                  (float((b-c).mean()),float((a-c).mean()),float((c-oracle).mean()))))
    timing=read("timings.csv"); warm=[r for r in timing if r["warmup"]=="0"]
    assert len(timing)==len({(r["frame_id"],r["view"]) for r in timing})==512 and len(warm)==508
    assert all({r["view"] for r in timing if r["frame_id"]==fid}==set("OFLR") for fid in ids)
    stats=dict(mse={k:float(v.mean()) for k,v in strategies.items()},comparison=comps,
        ci95=dict(zip(comps,intervals.tolist())),scene_clusters=len(clusters),
        odd_boot_batches=odd,bootstrap=10000,seed=20261128,
        groups={},schedule_oracle_overlap=int((mask*oracle_mask).sum()),
        native_units="metres squared, equal frame weights",geometric_pixels=int(arr("geometry_pixels").sum()),
        valid_pixels=int(arr("valid_pixels").sum()),D_cutpoints=[max(D[mask==0]),min(D[mask==1])],
        warmup_excluded=4,post_warmup_views=508,warm_ms={})
    for k in ("view_ms","processor_h2d_ms","forward_ms","resize_d2h_ms","align_ms"):
        values=[float(r[k]) for r in warm]
        stats["warm_ms"][k]=dict(mean=np.mean(values),median=np.median(values))
    for k,m in (("high",mask==1),("low",mask==0)):
        vals=g[m]
        stats["groups"][k]=dict(frames=len(vals),mean_D=D[m].mean(),mean_G=vals.mean(),
                       improved=int((vals>0).sum()),worsened=int((vals<0).sum()),tied=int((vals==0).sum()))
    assert np.isclose(comps["B_minus_C"],(stats["groups"]["high"]["mean_G"]-stats["groups"]["low"]["mean_G"])/4)
    (HERE/"summary.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    fig,ax=plt.subplots(figsize=(7.1,4.25),layout="constrained")
    for flag,color,label in [(0,"#758396","Lower D · two views"),(1,"#007569","Higher D · four views")]:
        ix=mask==flag
        ax.scatter(D[ix],g[ix],s=31,alpha=.82,c=color,edgecolor="white",linewidth=.4,label=f"{label} (64)")
    ax.axhline(0,color="#4a4a4a",lw=.8)
    ax.axvline(np.mean(stats["D_cutpoints"]),color="#626262",linestyle=":",lw=1)
    ax.set(xlabel="Original–mirror prediction disagreement D (m²)",
           ylabel="Benefit of two extra views G (m²)",title="NYU Depth V2 · 128 held-out frames")
    ax.legend(frameon=False); fig.savefig(HERE/"disagreement_gain.png",dpi=180); plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    run()
