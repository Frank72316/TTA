"""Two UUID-bound workers share only the frozen base predictions, never labels."""
import argparse,time
from data import *
from views import coordinate_test,extended,audit_input,ref,torch

def baseline_image(pp,model,im):
    return ref.forward(pp,model,[im])

def core_views(pp,model,im,b):
    boxes=old.boxes(*im.size)
    windows,wh=ref.forward(pp,model,[im.crop(box) for box in boxes])
    local,k=ref.stitch(b,windows,boxes)
    return windows,wh,k,(b+local)*.5

def extra(pp,model,im,b,method):
    windows,inputs,layouts=extended(pp,model,im,method)
    local,k=ref.stitch(b,windows,old.boxes(*im.size))
    return windows,inputs,layouts,k,(b+local)*.5

def synthetic(pp,model,shard):
    coords=coordinate_test(pp)
    dims=(2864,1924) if shard==0 else (1340,1110)
    im=Image.fromarray(np.random.default_rng(2026).integers(256,size=(*dims[::-1],3),dtype="u1"))
    rows=[]
    for rep in range(5):
        for method in (("B","T","P","R") if rep%2 else ("B","T","R","P")):
            def pipeline():
                (bb,_)=baseline_image(pp,model,im)
                if method=="B":return bb
                if method=="T":return core_views(pp,model,im,bb[0])[-1]
                return extra(pp,model,im,bb[0],method)[-1]
            _,ms,mem=ref.timed(pipeline)
            rows.append(dict(rep=rep,method=method,ms=ms,peak_MiB=mem,forwards=1 if method=="B" else 5,
                             warmup=rep==0,source_w=dims[0],source_h=dims[1]))
    save(ROOT/f"synthetic_{shard}.json",dict(coords=coords,rows=rows))

def legacy(r,idx):
    info_path=old.work(legacy_row(r));j=get(info_path/"inference.json");cov=get(info_path/"coverage.json")
    im,z,c=sources(r);sig=j["signature"]
    assert (sig["rgb_sha256"]==r["rgb_sha256"]==idx["rgb_sha"] and sig["common_impl"]==sha(LEGACY/"io_data.py")
            and sig["calibration"]==sha(LEGACY/"calibration.json"))
    assert sha(info_path/"predictions.npz")==idx["raw_prediction_sha"]==j["prediction_sha256"]
    assert sha(info_path/"edges.npz")==idx["edge_cache_sha"]==cov["edges_sha256"]
    assert cov["gt_sha256"]==r["depth_sha256"] and cov["calibration_sha256"]==r["calib_sha256"]
    with np.load(info_path/"predictions.npz") as ck:b,parts=ck["NB"],ck["NW"]
    return b,parts,j

def run(shard):
    fr=get(ROOT/"freeze.json")
    assert sha(ROOT/"scenes.csv")==fr["scenes_sha256"] and sha(ROOT/"protocol.md")==fr["protocol_sha256"]
    for f,s in fr["legacy_files"].items():assert sha(LEGACY/f)==s
    pp,model,meta=ref.setup()
    plan=included()
    idx={r["id"]:r for r in read(LEGACY/"cache_index.csv")}
    start=time.perf_counter();fresh=hits=prior=0
    if not (ROOT/f"synthetic_{shard}.json").exists():synthetic(pp,model,shard)
    for ix,r in enumerate(plan):
        if ix%2!=shard:continue
        d=cache(r);d.mkdir(parents=True,exist_ok=True)
        p=d/"predictions.npz";stamp=d/"inference.json"
        signature=dict(rgb=r["rgb_sha256"],frozen=sha(ROOT/"freeze.json"),code=[sha(ROOT/f) for f in ("views.py","score.py")],weights=meta["weights"])
        if stamp.exists():
            j=get(stamp);assert j["signature"]==signature and j["prediction_sha256"]==sha(p)
            hits+=j["new"];continue
        assert sha(sources(r)[0])==r["rgb_sha256"]
        im=Image.open(sources(r)[0]).convert("RGB");arr={};ks={};times={};mems={}
        if r["year"]=="2014":
            b,parts,j=legacy(r,idx[r["id"]])
            assert all(j["meta"][f]==meta[f] for f in ("model","revision","weights","torch","transformers","numpy","pillow","cuda"))
            n=j["conditions"]["N"];prior+=5
            tb,mb=n["B_ms"],n["B_peak_MiB"]
            times["T"],mems["T"]=n["T_ms"],n["T_peak_MiB"]
            hw,wh,ks["T"]=n["base_hw"],n["window_hw"],n["k"]
        else:
            (base,hw),tb,mb=ref.timed(lambda:baseline_image(pp,model,im));b=base[0]
            (parts,wh,ks["T"],_),dt,mt=ref.timed(lambda:core_views(pp,model,im,b))
            arr.update(B=b,T=parts);times["T"]=tb+dt;mems["T"]=max(mb,mt)
        times["B"]=tb;mems["B"]=mb
        ct=ref.stitch(b,parts,old.boxes(*im.size),ks["T"])[0]
        (_,blend_ms,_)=ref.timed(lambda:.75*b+.25*ct)
        # Time the complete local T25 path from cached window outputs via a separate component accounting.
        (_,half_ms,_)=ref.timed(lambda:(b+ct)*.5)
        times["T25"]=times["T"]-half_ms+blend_ms;mems["T25"]=mems["T"]
        del parts,ct
        hashes={}; order=("P","R") if ((ix//2+shard)%2) else ("R","P")
        for m in order:
            (arr[m],x,info,ks[m],_),dt,mt=ref.timed(lambda:extra(pp,model,im,b,m))
            times[m]=tb+dt;mems[m]=max(mb,mt);hashes[m]=(x,info)
        audit=audit_input(pp,im,hashes)
        assert list(wh)==info[0]["core_hw"] and min(v["changed_outer_pixels"] for v in audit)>0
        for v in arr.values():assert v.dtype==np.float32 and np.isfinite(v).all()
        with p.with_suffix(".part").open("wb") as f:np.savez(f,**arr)
        p.with_suffix(".part").replace(p)
        n=8 if r["year"]=="2014" else 13
        save(stamp,dict(signature=signature,new=n,prior=13-n,base_hw=hw,core_hw=wh,model=meta,
            layouts=info,input_audit=audit,k=ks,ms=times,peak_MiB=mems,order=list(order),prediction_sha256=sha(p)))
        fresh+=n
        print(key(r),n,flush=True)
    save(ROOT/f"worker_{shard}.json",dict(meta=meta,new=fresh,hits=hits,prior=prior,elapsed_s=time.perf_counter()-start))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--shard",type=int,required=True,choices=[0,1])
    run(parser.parse_args().shard)
