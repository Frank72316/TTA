"""Paired MAE contrasts and direct all-pairs squared losses on each original bin."""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from edge_data import *
from predict import stitch

def exhaustive_random_mse(r,u):
    # Enumerate the n x n array in blocks; no covariance enters this computation.
    total=0.
    for start in range(0,len(r),96):
        err=r[start:start+96,None]-u[None,:]
        np.square(err,out=err)
        total+=err.sum(dtype=np.float64)
    return total/(len(r)*len(u))

def cov2(r,u):
    return 2*np.mean((r-r.mean())*(u-u.mean()))

def square_metrics(r,uh,ul,diff,groups,base):
    bins=[]
    for axis,label,ix in groups:
        target=r[ix]
        item=dict(**base,axis=axis,bin=label,edges=len(ix),independent_pairs=len(ix)**2)
        for name,u in (("H",uh[ix]),("L",ul[ix])):
            random=exhaustive_random_mse(target,u)
            moment=np.mean(target**2)+np.mean(u**2)-2*target.mean()*u.mean()
            actual=np.mean((target-u)**2)
            covariance=cov2(target,u)
            item.update({name+"_random_m2":random,name+"_actual_m2":actual,name+"_placement_m2":random-actual,
                name+"_2cov_m2":covariance,name+"_moment_residual_m2":abs(moment-random)})
            assert max(abs(moment-random),abs(random-actual-covariance))<1e-11
        item["delta_placement_m2"]=item["H_placement_m2"]-item["L_placement_m2"]
        item["delta_2cov_m2"]=cov2(target,diff[ix])
        item["identity_residual_m2"]=abs(item["delta_placement_m2"]-item["delta_2cov_m2"])
        assert item["identity_residual_m2"]<1e-11
        bins.append(item)
    metrics={key:sum(b["edges"]*b[key] for b in bins)/len(r) for key in bins[0] if key.endswith("_m2") and "residual" not in key}
    return metrics,bins

def load_low(rec):
    folder=local_cache(rec);mark=get(folder/"score.json")
    assert sha(folder/"low.npy")==mark["low_prediction_sha256"]
    assert mark["signature"]["source"]==rec["rgb_sha256"] and mark["signature"]["implementation"]==sha(HERE/"score.py")
    return np.load(folder/"low.npy",allow_pickle=False),mark

def each(rec):
    b,high,info,near,far,gt,jb,old_u,groups=pinned_fields(rec)
    low,mark=load_low(rec)
    assert low.shape==high.shape and low.dtype==high.dtype==np.float32
    boxes=[w["box"] for w in info["layouts"]]
    kh=info["k"]["R"]
    _,kl=stitch(b,low,boxes)
    windows=[dict(**ident(rec),**v,k_H=kh[i],k_L=kl[i]) for i,v in enumerate(mark["window_rows"])]
    scenes=[];bin_rows=[]
    target=gt-jb
    for mode,scales in (("native_R",kh),("low_core",kl)):
        hmap=predict_map(b,high,boxes,scales);jh=jump(hmap,near,far)
        lmap=predict_map(b,low,boxes,scales);jl=jump(lmap,near,far)
        uh,ul=jh-jb,jl-jb
        if mode=="native_R":np.testing.assert_array_equal(uh*1000,old_u)
        ha,hc,ph=mae_pair(target,uh,groups)
        la,lc,pl=mae_pair(target,ul,groups)
        base=np.mean(np.abs(target))*1000
        ident_scale=dict(**ident(rec),scale=mode)
        square,detail=square_metrics(target,uh,ul,jh-jl,groups,ident_scale);bin_rows.extend(detail)
        scenes.append(dict(**ident_scale,edges=len(gt),bins=len(groups),B_actual_mm=base,H_actual_mm=ha,L_actual_mm=la,
            H_cond_mm=hc,L_cond_mm=lc,H_P_mm=ph,L_P_mm=pl,delta_P_mm=ph-pl,G_mm=la-ha,
            B_minus_H_mm=base-ha,B_minus_L_mm=base-la,**square))
    cache_row=dict(**ident(rec),low_prediction_sha256=mark["low_prediction_sha256"],low_path=str(local_cache(rec)/"low.npy"),
        historical_inference_sha256=mark["signature"]["native_meta"],uuid=mark["uuid"],image_forwards=mark["unique_forwards"],
        prepare_predict_ms=mark["prepare_predict_ms"],network_and_remap_ms=mark["network_and_remap_ms"],
        peak_allocated_MiB=mark["peak_allocated_MiB"])
    return scenes,bin_rows,windows,cache_row

def main():
    load_freeze();before=time.perf_counter()
    native={(r["year"],r["id"]):r for r in read(RESULTS/"native_replay.csv")}
    assert len(native)==50
    with ProcessPoolExecutor(4) as pool:out=list(pool.map(each,included()))
    scenes=[r for item in out for r in item[0]];bins=[r for item in out for r in item[1]]
    windows=[r for item in out for r in item[2]];index=[item[3] for item in out]
    for r in scenes:
        if r["scale"]=="native_R":
            assert float(native[r["year"],r["id"]]["H_P_mm"])==r["H_P_mm"]
    fields=[k for k in scenes[0] if k.endswith(("_mm","_m2"))]
    summary=intervals(scenes,fields)
    coverage=[]
    for batch,years in COHORTS.items():
        ws=[w for w in windows if w["year"] in years]
        totals={k:sum(w[k] for w in ws) for k in ("core_pixels","changed_core","core_abs_rgb_sum","outer_pixels","outer_tensor_changed",
             "band_pixels","band_changed","band_abs_rgb_sum","interior_pixels","interior_changed","interior_abs_rgb_sum")}
        count=sum(r["edges"] for r in scenes if r["scale"]=="native_R" and r["year"] in years)
        coverage.append(dict(batch=batch,scenes=len(ws)//4,bins=sum(r["bins"] for r in scenes if r["year"] in years and r["scale"]=="native_R"),
            edges=count,windows=len(ws),core_changed_fraction=totals["changed_core"]/totals["core_pixels"],
            mean_core_abs_rgb=totals["core_abs_rgb_sum"]/(3*totals["core_pixels"]),max_core_abs_rgb=max(w["core_max_rgb_change"] for w in ws),
            min_changed_fraction=min(w["changed_core"]/w["core_pixels"] for w in ws),max_changed_fraction=max(w["changed_core"]/w["core_pixels"] for w in ws),
            **totals))
    for name,rows in (("per_scene",scenes),("per_bin_mse",bins),("input_windows",windows),("cache_index",index),
                      ("input_summary",coverage),("summary",summary)):
        write(RESULTS/(name+".csv"),rows)
    assert sum(c["edges"] for c in coverage)==243818 and len(bins)==990 and len(windows)==200
    shards=get(RESULTS/"worker_0.json")["shards"]
    counts=[get(RESULTS/f"worker_{i}.json") for i in range(shards)]
    for i,worker in enumerate(counts):
        assert worker["shard"]==i and worker["shards"]==shards
        assert worker["fresh"]+worker["hits"]==4*len(index[i::shards])
    assert sum(w["fresh"]+w["hits"] for w in counts)==sum(r["image_forwards"] for r in index)==200
    save(RESULTS/"checks.json",dict(scenes=len(index),scene_scale_rows=len(scenes),edges=243818,groups=495,all_native_input_hashes_verified=True,
        total_outer_changed=sum(w["outer_tensor_changed"] for w in windows),max_H_direct_vs_cov_m2=max(abs(r["H_placement_m2"]-r["H_2cov_m2"]) for r in bins),
        max_L_direct_vs_cov_m2=max(abs(r["L_placement_m2"]-r["L_2cov_m2"]) for r in bins),max_delta_identity_m2=max(r["identity_residual_m2"] for r in bins),
        max_moment_residual_m2=max(max(r["H_moment_residual_m2"],r["L_moment_residual_m2"]) for r in bins),
        evaluated_ordered_pairs_all_scales_conditions=2*sum(r["independent_pairs"] for r in bins),
        historical_B_reused=50,historical_H_windows_reused=200,new_L_forwards=sum(w["fresh"] for w in counts),
        resumed_L_hits=sum(w["hits"] for w in counts),gpu_batches=sum(w["fresh"] for w in counts)//4,CPU_analysis_s=time.perf_counter()-before))
    for r in summary:
        if r["metric"] in ("delta_P_mm","G_mm","L_P_mm","B_minus_H_mm","B_minus_L_mm","delta_placement_m2"):
            print(r["batch"],r["scale"],r["metric"],*[round(r[k],6) for k in ("mean","lo","hi")])

if __name__=="__main__":main()
