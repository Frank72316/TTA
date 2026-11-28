"""Aggregate frozen coverage, exclusions, hardware use and first-task timing without model imports."""
import numpy as np
from io_data import *
def main():
    inv=read(HERE/"inventory.csv");cov=read(HERE/"coverage.csv");tim=read(HERE/"latencies.csv")
    runs=[get(p) for p in sorted(HERE.glob("gpurun_*.json"))]
    cold={z["rows"][0]["id"] for z in runs if z["rows"]};rows=[];separate=[]
    for ds in ("ibims","middlebury"):
        rr=[r for r in cov if r["dataset"]==ds];ii=[r for r in inv if r["dataset"]==ds]
        a=dict(dataset=ds,expected=len(ii),input_ready=sum(int(r["ready"]) for r in ii),with_edges=sum(int(r["all_edges"])>0 for r in rr),
               with_narrow=sum(int(r["narrow_edges"])>0 for r in rr),with_wide=sum(int(r["wide_edges"])>0 for r in rr),
               narrow_at_least_20_each=sum(min(int(r["narrow_near_pixels"]),int(r["narrow_far_pixels"]))>=20 for r in rr),
               wide_at_least_20_each=sum(min(int(r["wide_near_pixels"]),int(r["wide_far_pixels"]))>=20 for r in rr),
               both_at_least_20_each=sum(int(r["both_eligible"]) for r in rr))
        for name in ("native_pixels","depth_positive_finite","out_of_model_range","valid_pixels","adjacent","both_valid","coarse_jump",
                     "in_bounds","no_holes","consistent_jump","stable","narrow_edges","wide_edges"):
            a[name]=sum(int(r[name]) for r in rr) if rr else ""
        fields=["adjacent","both_valid","coarse_jump","in_bounds","no_holes","consistent_jump","stable"]
        for first,second in zip(fields,fields[1:]):
            a["excluded_at_"+second]=a[first]-a[second] if rr else ""
        rows.append(a)
        for group,sel in [("first_per_worker",True),("remaining",False)]:
            t=[r for r in tim if r["dataset"]==ds and (named(r) in cold)==sel]
            for k in ("B","T","H"):
                samples=[float(r[k+"_ms"]) for r in t if r.get(k+"_ms")]
                if samples:separate.append(dict(dataset=ds,stratum=group,strategy=k,images=len(samples),median_ms=np.median(samples),p90_ms=np.percentile(samples,90)))
    write(HERE/"coverage_summary.csv",rows);write(HERE/"worker_timing.csv",separate)
    calibration=read(HERE/"calibration.csv")
    unique=sum(int(r["image_forwards"]) for r in tim)
    assert len({r["id"] for r in tim})==len(tim) and sum(r["fresh"] for r in runs)==unique
    assert len(set(r["meta"]["uuid"] for r in runs))==len(runs)
    save(HERE/"accounting.json",dict(planned_formal=1353,actual_formal=unique,formal_by_dataset={
        ds:sum(int(r["image_forwards"]) for r in tim if r["dataset"]==ds) for ds in ("ibims","middlebury")},
        measured_synthetic_and_warmup=sum(int(r["completed_image_forwards"]) for r in calibration),
        extra_formal_forwards_from_PFM_repair=0,OOM_calibration=sum(r["status"]=="OOM" for r in calibration),
        formal_images_first_on_worker=len(cold),workers=len(runs),worker_wall_seconds=[r["wall_s"] for r in runs],
        cache_hits=sum(i["hits"] for r in runs for i in r["rows"]),total_real_and_synthetic=unique+sum(int(r["completed_image_forwards"]) for r in calibration)))
if __name__=="__main__":main()
