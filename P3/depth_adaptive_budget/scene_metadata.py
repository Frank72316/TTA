"""Read only sequence identifiers from the official NYU labeled HDF5 metadata."""
import argparse
import csv
import io
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

URL="https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
HERE=Path(__file__).resolve().parent


class RangeFile(io.RawIOBase):
    """File-like reader for h5py, fetching small metadata blocks on demand."""
    block=256*1024
    def __init__(self,url):
        self.url=url
        with urlopen(Request(url,headers={"Range":"bytes=0-0"}),timeout=30) as r:
            self.length=int(r.headers["Content-Range"].split("/")[-1])
        self.position=0

    def seek(self,offset,whence=0):
        self.position=(0,self.position,self.length)[whence]+offset
        return self.position

    def tell(self):
        return self.position

    @lru_cache(maxsize=96)
    def chunk(self,index):
        lo=index*self.block; hi=min(self.length,lo+self.block)-1
        for attempt in range(4):
            try:
                with urlopen(Request(self.url,headers={"Range":f"bytes={lo}-{hi}"}),timeout=35) as r:
                    assert r.headers["Content-Range"]==f"bytes {lo}-{hi}/{self.length}"
                    return r.read()
            except OSError:
                if attempt==3: raise

    def readinto(self,b):
        count=min(len(b),self.length-self.position)
        if count<=0:return 0
        n=0
        while n<count:
            index,offset=divmod(self.position,self.block)
            data=self.chunk(index)[offset:offset+count-n]
            b[n:n+len(data)]=data
            n+=len(data); self.position+=len(data)
        return count


def read_scenes(source,manifest,out):
    import h5py
    with manifest.open(encoding="utf8") as f: rows=list(csv.DictReader(f))
    with h5py.File(source,"r") as mat:
        refs=mat["scenes"][:].ravel()
        assert len(refs)==1449
        for row in rows:
            ix=int(Path(row["frame"]).stem.split("_")[-1])
            # Archive stems are zero-based; MATLAB's testNdxs is one-based.
            # measure.py independently verifies this mapping for all 654 IDs.
            name="".join(chr(int(v)) for v in mat[refs[ix]][:].ravel())
            assert name and "_" in name and name.isprintable(),name
            row["scene"]=name
    with out.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=("frame","scene"),extrasaction="ignore",lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(len(rows),"frames",len({r["scene"] for r in rows}),"scenes",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,default=HERE/"frames.csv"); ap.add_argument("--out",type=Path,default=HERE/"scenes.csv")
    ap.add_argument("--mat",type=Path)
    a=ap.parse_args(); read_scenes(a.mat if a.mat else RangeFile(URL),a.manifest,a.out)
