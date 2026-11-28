"""GT-only audit; all 2014 indices and validity rules remain read-only."""
from concurrent.futures import ProcessPoolExecutor
from data import *
from edges import geometry, synthetic
from transformers import AutoImageProcessor

def edge_path(r):
    if r["year"]=="2014":
        return old.work(legacy_row(r))/"edges.npz"
    return DATA/"edges"/key(r)/"edges.npz"

def one(r):
    pi,pd,pc=sources(r)
    assert all(sha(p)==r[k] for p,k in ((pi,"rgb_sha256"),(pd,"depth_sha256"),(pc,"calib_sha256")))
    im=Image.open(pi).convert("RGB")
    z=metric(r);valid=np.isfinite(z)&(z>0)&(z<=20)
    assert im.size==z.shape[::-1]
    path=edge_path(r)
    if r["year"]=="2014":
        c=get(path.parent/"coverage.json")
        assert c["edges_sha256"]==sha(path) and sha(LEGACY/"edges.py")==c["geometry_source"]
        qc=dict(source="2014 perfect im0/disp0",depth_unit="m")
    else:
        disp=np.array(Image.open(pd));off=int(pc.read_text());s=disp[disp>0].astype(float)+off
        step=.160*3740/(s*(s+1))
        # View-1 pixels sample the rectified right view at x - stored disparity.
        gray=np.asarray(im.convert("L"),dtype=float)
        right=np.asarray(Image.open(pi.parent/"view5.png").convert("L"),dtype=float)
        dr=np.asarray(Image.open(pi.parent/"disp5.png"))
        assert right.shape==dr.shape==z.shape and im.height==1110
        yy,xx=np.where((disp>0)&valid);yy=yy[::31];xx=xx[::31]
        shift=disp[yy,xx].astype(int)
        ok=(xx-shift>=0)&(xx+shift<im.width)
        yy,xx,shift=yy[ok],xx[ok],shift[ok]
        assert len(xx)>0
        a=gray[yy,xx]
        qc=dict(source=r["exposure"]+"/view1",depth_unit="m",dmin=off,dmin_input_sha256=sha(pc),
            disparity_min=int(disp[disp>0].min()),disparity_max=int(disp.max()),quant_step_mm_median=float(np.median(step)*1000),
            quant_step_mm_min=float(step.min()*1000),quant_step_mm_max=float(step.max()*1000),
            correct_warp_luma_MAE=float(np.abs(a-right[yy,xx-shift]).mean()),
            opposite_warp_luma_MAE=float(np.abs(a-right[yy,xx+shift]).mean()),
            stereo_consistent_frac=float((np.abs(dr[yy,xx-shift].astype(int)-shift)<=1).mean()),stereo_points=len(a))
        assert qc["correct_warp_luma_MAE"]<qc["opposite_warp_luma_MAE"]
        pp=AutoImageProcessor.from_pretrained(old.MODEL,local_files_only=True)
        hw=tuple(pp(images=im,return_tensors="np").pixel_values.shape[-2:])
        e,counts=geometry(z,hw);c=dict(**counts,base_h=hw[0],base_w=hw[1])
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix(".part").open("wb") as f:np.savez_compressed(f,**e)
        path.with_suffix(".part").replace(path)
    with np.load(path) as p:
        near,far=p["near"],p["far"]
        qc.update(narrow_edges=int((p["width"]==1).sum()),wide_edges=int((p["width"]==2).sum()))
    assert (z.ravel()[near].mean(axis=1)<z.ravel()[far].mean(axis=1)).all()
    return dict(year=r["year"],id=r["id"],cluster=r["cluster"],native_w=im.width,native_h=im.height,
        depth_p05_m=float(np.percentile(z[valid],5)),depth_p95_m=float(np.percentile(z[valid],95)),
        valid_pixels=int(valid.sum()),native_pixels=z.size,range_excluded=int((z>20).sum()),missing_pixels=int((~np.isfinite(z)).sum()),
        **{k:c[k] for k in ("base_h","base_w","adjacent","both_valid","coarse_jump","in_bounds","no_holes","consistent_jump","stable")},
        near_unique=len(np.unique(near)),far_unique=len(np.unique(far)),edge_sha256=sha(path),**qc)

if __name__=="__main__":
    assert synthetic()
    with ProcessPoolExecutor(5) as ex:rows=list(ex.map(one,included()))
    write(ROOT/"coverage.csv",rows)
