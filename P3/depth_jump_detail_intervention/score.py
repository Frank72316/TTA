"""Intervene in encoder RGB space and run only the four low-detail windows."""
import argparse
import time
import numpy as np
from PIL import Image
from shared import *
from views import canvas, ref, torch, F

def build_inputs(pp, image, info):
    base_hw=tuple(info["base_hw"])
    assert ref.raster(pp,image).shape[1:]==base_hw
    low_image=old.degrade(image,base_hw)
    native_views=[];low_views=[];rows=[]
    for i,layout in enumerate(info["layouts"]):
        src=layout["box"]
        native=ref.raster(pp,image.crop(src)).transpose(1,2,0)
        low=ref.raster(pp,low_image.crop(src)).transpose(1,2,0)
        h,w=native.shape[:2]
        x=np.minimum(np.arange(w),np.arange(w)[::-1])
        y=np.minimum(np.arange(h),np.arange(h)[::-1])
        weight=np.minimum(np.minimum(x[None,:],y[:,None]),8).astype(np.int32)
        # Integer numerator is exact for the 1/8 weights. Adding four implements half-up rounding.
        core=((native.astype(np.int32)*(8-weight[:,:,None])+low.astype(np.int32)*weight[:,:,None]+4)//8).astype("uint8")
        np.testing.assert_array_equal(core[weight==8],low[weight==8])
        np.testing.assert_array_equal(core[weight==0],native[weight==0])
        hi=canvas(image,native,layout,"R",pp.resample)
        lo=hi.copy()
        a,b,c,d=layout["net_box"]
        lo[b:d,a:c]=core
        native_views.append(hi);low_views.append(lo)
        diff=np.abs(native.astype("i2")-core)
        changed=diff.any(axis=2)
        row=dict(corner=i,source_w=image.width,source_h=image.height,base_h=base_hw[0],base_w=base_hw[1],
                 core_h=h,core_w=w,canvas_h=hi.shape[0],canvas_w=hi.shape[1],
                 native_box=" ".join(map(str,src)),encoder_core_box=" ".join(map(str,layout["net_box"])),
                 core_pixels=h*w,changed_core=int(changed.sum()),core_abs_rgb_sum=int(diff.sum()),
                 core_max_rgb_change=int(diff.max()),outer_pixels=hi.shape[0]*hi.shape[1]-h*w)
        for name,mask in (("band",weight<8),("interior",weight==8)):
            row[name+"_pixels"]=int(mask.sum())
            row[name+"_changed"]=int(changed[mask].sum())
            row[name+"_abs_rgb_sum"]=int(diff[mask].sum())
        rows.append(row)
    def normalize(items):
        return pp(images=[Image.fromarray(v) for v in items],do_resize=False,return_tensors="pt").pixel_values
    hi_tensor=normalize(native_views);lo_tensor=normalize(low_views)
    for i,(layout,row) in enumerate(zip(info["layouts"],rows)):
        hi,lo=hi_tensor[i],lo_tensor[i]
        hist=info["input_audit"][i]
        native_hash=ah(hi.numpy())
        assert native_hash==hist["R_sha256"]
        a,b,c,d=layout["net_box"]
        outside=np.ones(hi.shape[-2:],dtype=bool);outside[b:d,a:c]=False
        changed=torch.any(hi!=lo,dim=0).numpy()
        assert not changed[outside].any() and row["changed_core"]==int(changed.sum())
        assert row["changed_core"]>0 and tuple(hi.shape[1:])==tuple(layout["canvas_hw"])
        row.update(outer_tensor_changed=0,outer_tensor_max_abs=0.,native_sha256=native_hash,low_sha256=ah(lo.numpy()),
            core_tensor_abs_mean=float(torch.mean(torch.abs(hi[:,b:d,a:c]-lo[:,b:d,a:c]),dtype=torch.float64)))
    return lo_tensor,rows,native_views,low_views

def predict_windows(model,batch,layouts):
    with torch.inference_mode():
        out=model(pixel_values=batch.cuda()).predicted_depth
        assert out.dtype==torch.float32 and out.shape==(4,*batch.shape[-2:])
        cores=[]
        for i,layout in enumerate(layouts):
            a,b,c,d=layout["net_box"]
            cores.append(out[i,b:d,a:c])
        x,y,u,v=layouts[0]["box"]
        restored=F.interpolate(torch.stack(cores)[:,None],size=(v-y,u-x),mode="bicubic",align_corners=False)
        result=restored[:,0].cpu().numpy()
    assert np.isfinite(result).all()
    return result

def run(shard,shards):
    frozen=load_freeze()
    pp,model,meta=ref.setup()
    idx={(r["year"],r["id"]):r for r in read(CONTEXT/"cache_index.csv")}
    start=time.perf_counter();fresh=hits=0
    for rec in included()[shard::shards]:
        jpath=cache(rec)/"inference.json"
        entry=idx[rec["year"],rec["id"]]
        assert sha(jpath)==entry["inference_sha256"]
        info=get(jpath)
        assert all(meta[f]==info["model"][f] for f in ("model","revision","weights","torch","transformers","numpy","pillow","cuda"))
        assert sha(sources(rec)[0])==rec["rgb_sha256"]==entry["rgb_sha256"]
        dest=local_cache(rec);dest.mkdir(parents=True,exist_ok=True)
        path=dest/"low.npy";record=dest/"score.json"
        signature=dict(freeze=sha(HERE/"freeze.json"),source=rec["rgb_sha256"],native_meta=entry["inference_sha256"],
                       implementation=sha(HERE/"score.py"),shared=sha(HERE/"shared.py"),revision=meta["revision"],
                       weights=meta["weights"],protocol=frozen["protocol"])
        if record.exists():
            mark=get(record)
            assert signature==mark["signature"] and sha(path)==mark["low_prediction_sha256"]
            hits+=4;continue
        im=Image.open(sources(rec)[0]).convert("RGB")
        tick=time.perf_counter()
        batch,rows,_,_=build_inputs(pp,im,info)
        (low,ms,mem)=ref.timed(lambda:predict_windows(model,batch,info["layouts"]))
        total_ms=(time.perf_counter()-tick)*1000
        tmp=path.with_suffix(".part")
        with tmp.open("wb") as f:np.save(f,low,allow_pickle=False)
        tmp.replace(path)
        save(record,dict(signature=signature,window_rows=rows,low_prediction_sha256=sha(path),unique_forwards=4,
                         uuid=meta["uuid"],network_and_remap_ms=ms,prepare_predict_ms=total_ms,peak_allocated_MiB=mem))
        fresh+=4
        print(rec["year"],rec["id"],fresh,flush=True)
    save(RESULTS/f"worker_{shard}.json",dict(shard=shard,shards=shards,meta=meta,fresh=fresh,hits=hits,elapsed_s=time.perf_counter()-start))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,required=True);p.add_argument("--shards",type=int,required=True)
    args=p.parse_args()
    assert 0<=args.shard<args.shards
    run(args.shard,args.shards)
