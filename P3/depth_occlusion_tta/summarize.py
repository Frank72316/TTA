"""All scene differences are paired; summarize on metric-native or explicitly converted units."""
from collections import defaultdict
import numpy as np
from io_data import *

def estimate(vals,clusters):
    groups=defaultdict(list)
    for v,c in zip(vals,clusters):groups[c].append(v)
    unit=np.array([np.mean(g) for g in groups.values()])
    rng=np.random.default_rng(2026);boot=unit[rng.integers(len(unit),size=(10000,len(unit)))].mean(axis=1)
    return dict(mean=unit.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),images=len(vals),clusters=len(unit),
                positive=sum(v>0 for v in vals),zero=sum(v==0 for v in vals),negative=sum(v<0 for v in vals))
def main():
    metrics=read(HERE/"per_image.csv");cov=read(HERE/"coverage.csv")
    metadata={(r["dataset"],r["id"]):r for r in cov}
    summary=[];diffs=[];cased=[];lt=[];files=[]
    for ds in ("ibims","middlebury"):
        rs=[r for r in read(HERE/"expected.csv") if r["dataset"]==ds and (ds,r["id"]) in metadata]
        if not rs:continue
        by={(r["id"],r["condition"],r["strategy"]):r for r in metrics if r["dataset"]==ds}
        groups={r["id"]:r["id"] for r in rs}
        if ds=="ibims":
            rooms={r["id"]:get(paths(r)[2]).get("room_id") for r in rs}
            if all(rooms.values()):groups=rooms
        for i,r in enumerate(rs):
            ident=r["id"];d=work(r);info=get(d/"inference.json");cv=metadata[ds,ident]
            record=dict(**r,cluster_id=groups[ident],both_width=int(cv["both_eligible"]))
            def val(c,s,m="all_mae"):
                x=by.get((ident,c,s),{}).get(m)
                return float(x) if x not in ("",None) else np.nan
            def gain(c,m="all_mae",s="T"):return (val(c,"B",m)-val(c,s,m))*1000
            if int(cv["all_edges"]):
                record.update(G_N=gain("N"),G_L=gain("L"),delta_G=gain("N")-gain("L"),
                        common_anchor=(val("counterfactual","BN_kN_Lwindows")-val("N","T"))*1000,
                        delta_G_unscaled=gain("N",s="fusion_raw")-gain("L",s="fusion_raw"))
                for c in ("N","L"):
                    record[f"scale_effect_{c}"]=(val(c,"fusion_raw")-val(c,"T"))*1000
                    for m in ("near_mae","far_mae","jump_mae"):
                        record[f"{c}_{m}_gain"]=gain(c,"all_"+m)
                if (ident,"N","H") in by:
                    record["H_minus_T"]=(val("N","H")-val("N","T"))*1000
                if int(cv["both_eligible"]):
                    record["narrow_gain"]=gain("N","narrow_mae");record["wide_gain"]=gain("N","wide_mae")
                    record["narrow_minus_wide"]=record["narrow_gain"]-record["wide_gain"]
            for c in ("N","L"):
                record[c+"_global_mae_gain"]=gain(c,"global_mae")
                record[c+"_absrel_gain"]=val(c,"B","absrel")-val(c,"T","absrel")
            diffs.append(record)
            cond=info["conditions"]
            glob=dict(**r,w=info["size"][0],h=info["size"][1],model_h=cond["N"]["base_hw"][0],model_w=cond["N"]["base_hw"][1],
                 uuid=info["meta"]["uuid"],B_NL_MAD_m=info["B_prediction_MAD_m"],
                 N_grid_sha=cond["N"]["grid_sha"],L_grid_sha=cond["L"]["grid_sha"],grid_MAD_255=cond["L"]["grid_mean_abs"],
                 grid_max_255=cond["L"]["grid_max_abs"],B_ms=cond["N"]["B_ms"],T_ms=cond["N"]["T_ms"],
                 B_peak_MiB=cond["N"]["B_peak_MiB"],T_peak_MiB=cond["N"]["T_peak_MiB"],
                 L_B_ms=cond["L"]["B_ms"],L_T_ms=cond["L"]["T_ms"],L_B_peak_MiB=cond["L"]["B_peak_MiB"],L_T_peak_MiB=cond["L"]["T_peak_MiB"],
                 image_forwards=info["forwards"],H_factor=info["H_actual_factor"])
            if "H" in cond:
                hh,ww=cond["H"]["hw"];glob.update(H_ms=cond["H"]["ms"],H_peak_MiB=cond["H"]["peak_MiB"],H_net_h=hh,H_net_w=ww)
            lt.append(glob)
            files.append(dict(**r,rgb_sha=info["signature"]["rgb_sha256"],depth_sha=cv["gt_sha256"],
                raw_prediction_sha=info["prediction_sha256"],edge_cache_sha=cv["edges_sha256"],cache=str(d)))
        selected=[r for r in diffs if r["dataset"]==ds]
        for metric in dict.fromkeys(k for r in selected for k in r if k not in ("dataset","id","cluster_id","both_width")):
            usable=[r for r in selected if metric in r and np.isfinite(r[metric])]
            if usable:
                summary.append(dict(dataset=ds,metric=metric,units="unitless" if "absrel" in metric else "mm",
                       **estimate([r[metric] for r in usable],[r["cluster_id"] for r in usable])))
        elig=[r for r in selected if "G_N" in r]
        if elig:
            median=np.median([r["G_N"] for r in elig])
            for name,case in (("best",max(elig,key=lambda r:r["G_N"])),("median",min(elig,key=lambda r:abs(r["G_N"]-median))),("worst",min(elig,key=lambda r:r["G_N"]))):
                cased.append(dict(dataset=ds,case=name,id=case["id"],G_mm=case["G_N"]))
        for c,s in dict.fromkeys((r["condition"],r["strategy"]) for r in metrics if r["dataset"]==ds):
            for m in ("all_mae","all_near_signed","all_far_signed","all_jump_signed","all_jump_mae","global_mae","absrel"):
                rr=[r for r in metrics if r["dataset"]==ds and (r["condition"],r["strategy"])==(c,s) and r.get(m)]
                if not rr:continue
                fac=1 if m=="absrel" else 1000
                summary.append(dict(dataset=ds,metric=c+"_"+s+"_"+m,units="unitless" if fac==1 else "mm",
                    **estimate([fac*float(r[m]) for r in rr],[groups[r["id"]] for r in rr])))
    write(HERE/"paired_results.csv",diffs);write(HERE/"summary.csv",summary);write(HERE/"cases.csv",cased)
    write(HERE/"latencies.csv",lt);write(HERE/"cache_index.csv",files)
    ls=[]
    for ds in sorted({r["dataset"] for r in lt}):
        for k in ("B","T","H","L_B","L_T"):
            rr=[r for r in lt if r["dataset"]==ds and k+"_ms" in r]
            if rr:ls.append(dict(dataset=ds,strategy=k,n=len(rr),median_ms=np.median([r[k+"_ms"] for r in rr]),
                p90_ms=np.percentile([r[k+"_ms"] for r in rr],90),peak_allocated_MiB=max(r[k+"_peak_MiB"] for r in rr)))
    write(HERE/"latency_summary.csv",ls)
    windows=read(HERE/"per_window.csv");ws=[]
    for ds,c in dict.fromkeys((r["dataset"],r["condition"]) for r in windows):
        rr=[r for r in windows if (r["dataset"],r["condition"])==(ds,c)]
        a=np.array([float(r["k"]) for r in rr])
        ws.append(dict(dataset=ds,condition=c,windows=len(a),k_median=np.median(a),k_q25=np.quantile(a,.25),k_q75=np.quantile(a,.75),
                       lower=int(sum(a==.5)),upper=int(sum(a==2)),fallback=sum(int(r["fallback"]) for r in rr),
                       median_log_before=np.median([float(r["abs_log_before"]) for r in rr if r["abs_log_before"]]),
                       median_log_after=np.median([float(r["abs_log_after"]) for r in rr if r["abs_log_after"]])))
    write(HERE/"scale_summary.csv",ws)
if __name__=="__main__":main()
