"""One-time exact provenance reconciliation for the GT-only PFM unit correction."""
import hashlib
from io_data import *
def main():
    src=HERE/"io_data.py"
    oldline=b'return np.ascontiguousarray(arr.reshape(h,w)[::-1],dtype="float32")'
    current=src.read_bytes();assert current.count(oldline)==1
    prev=hashlib.sha256(current.replace(oldline,oldline+b"*abs(scale)")).hexdigest()
    assert prev=="465779388d2b45a15cf772dd581af55c3c4cae916bad99921b61dfe440e0fc78"
    for r in read(HERE/"expected.csv"):
        p=work(r)/"inference.json"
        if not p.exists():continue
        x=get(p)
        if x["signature"]["common_impl"]==sha(src):continue
        assert x["signature"]["common_impl"]==prev and sha(paths(r)[0])==x["signature"]["rgb_sha256"]
        assert sha(work(r)/"predictions.npz")==x["prediction_sha256"]
        x["inference_common_original_sha256"]=prev
        x["signature"]["common_impl"]=sha(src)
        save(p,x)
if __name__=="__main__":main()
