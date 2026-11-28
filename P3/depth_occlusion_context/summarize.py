"""All reported differences give each numbered-series cluster identical weight."""
from collections import defaultdict
from data import *

SETS={"2014":{"2014"},"2005+2006":{"2005","2006"},"2005":{"2005"},"2006":{"2006"}}

def interval(rows,field,scale=1):
    groups=defaultdict(list)
    for r in rows:
        if r.get(field) not in ("",None):
            groups[r["cluster"]].append(float(r[field])*scale)
    values=np.array([np.mean(groups[k]) for k in sorted(groups)])
    if not len(values):return {}
    draws=np.random.default_rng(2026).integers(len(values),size=(10000,len(values)))
    lo,hi=np.percentile(values[draws].mean(axis=1),[2.5,97.5])
    return dict(mean=float(values.mean()),lo=float(lo),hi=float(hi),scenes=sum(map(len,groups.values())),clusters=len(values))

def main():
    per=read(ROOT/"per_scene.csv");diff=read(ROOT/"paired_differences.csv")
    cov=read(ROOT/"coverage.csv");mov=read(ROOT/"pixel_movements.csv")
    latency=read(ROOT/"latencies.csv");qs=[];motion=[];cover=[];clocks=[]
    for k,years in SETS.items():
        sub=lambda rows:[r for r in rows if r["year"] in years]
        for field in diff[0]:
            if field.endswith("_mm"):
                rows=sub(diff);val=interval(rows,field)
                if val:qs.append(dict(batch=k,metric=field,method="paired",unit="mm",**val,
                    positive=sum(float(r[field])>0 for r in rows if r[field]),negative=sum(float(r[field])<0 for r in rows if r[field])))
        for method in sorted({r["method"] for r in per}):
            group=[r for r in sub(per) if r["method"]==method]
            for field in ("all_near_mae","all_far_mae","all_mae","all_near_signed","all_far_signed","all_jump_mae","all_jump_signed","global_mae","absrel"):
                factor=1 if field=="absrel" else 1000
                qs.append(dict(batch=k,method=method,metric=field,unit="ratio" if factor==1 else "mm",**interval(group,field,factor)))
        for side in ("near","far"):
            for direction in ("nearer","farther","unchanged"):
                group=[r for r in sub(mov) if r["side"]==side and r["direction"]==direction]
                counts={v:sum(int(r[v]) for r in group) for v in ("total","count","improved","worsened","tied")}
                shift=interval(group,"err_change_mm")
                motion.append(dict(batch=k,side=side,direction=direction,**counts,
                    frac_cluster=interval(group,"fraction")["mean"],pooled_frac=counts["count"]/counts["total"],
                    mean_err_change_mm=shift.get("mean"),lo=shift.get("lo"),hi=shift.get("hi"),
                    conditional_clusters=shift.get("clusters",0)))
        group=sub(cov)
        cover.append(dict(batch=k,scenes=len(group),clusters=len({r["cluster"] for r in group}),
                     scenes_with_edges=sum(int(r["stable"])>0 for r in group),**{v:sum(int(r[v]) for r in group)
                     for v in ("adjacent","both_valid","coarse_jump","in_bounds","no_holes","consistent_jump","stable","near_unique","far_unique","narrow_edges","wide_edges","native_pixels","valid_pixels")}))
        for method in ("B","T","T25","P","R"):
            group=[r for r in sub(latency) if r["method"]==method]
            ms=[float(r["ms"]) for r in group];mem=[float(r["peak_MiB"]) for r in group]
            clocks.append(dict(source="real",batch=k,method=method,n=len(ms),median_ms=np.median(ms),p90_ms=np.percentile(ms,90),max_allocated_MiB=max(mem)))
    for shard in range(2):
        ss=get(ROOT/f"synthetic_{shard}.json")["rows"]+get(ROOT/f"blend_time_{shard}.json")["rows"]
        for method in sorted({r["method"] for r in ss}):
            group=[r for r in ss if not r["warmup"] and r["method"]==method]
            ms=[r["ms"] for r in group];q=group[0]
            clocks.append(dict(source="synthetic",batch=f'{q["source_w"]}x{q["source_h"]}',method=method,n=len(ms),median_ms=np.median(ms),p90_ms=np.percentile(ms,90),max_allocated_MiB=max(r["peak_MiB"] for r in group)))
    write(ROOT/"summary.csv",qs);write(ROOT/"movement_summary.csv",motion)
    write(ROOT/"coverage_summary.csv",cover);write(ROOT/"latency_summary.csv",clocks)

if __name__=="__main__":main()
