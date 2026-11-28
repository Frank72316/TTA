"""Complete synthetic end-to-end timing of the two offline blend controls."""
from score import *
from views import ref

def run_blends(shard):
    pp,model,meta=ref.setup()
    size=(2864,1924) if shard==0 else (1340,1110)
    im=Image.fromarray(np.random.default_rng(2026).integers(256,size=(*size[::-1],3),dtype="u1"))
    boxes=old.boxes(*size)
    rows=[]
    for rep in range(5):
        for method in ("T25","C"):
            def run_one():
                base,_=baseline_image(pp,model,im)
                wins,_=ref.forward(pp,model,[im.crop(b) for b in boxes])
                merged,_=ref.stitch(base[0],wins,boxes)
                return .75*base[0]+.25*merged if method=="T25" else merged
            _,ms,mem=ref.timed(run_one)
            rows.append(dict(rep=rep,method=method,ms=ms,peak_MiB=mem,forwards=5,warmup=rep==0,source_w=size[0],source_h=size[1]))
    save(ROOT/f"blend_time_{shard}.json",dict(uuid=meta["uuid"],rows=rows))

if __name__=="__main__":run_blends(int(sys.argv[1]))
