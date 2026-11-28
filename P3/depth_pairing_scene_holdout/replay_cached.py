"""Independent split, window, recipe and bootstrap checks. CPU only; no images required."""
import argparse
from collections import Counter
import json
from pathlib import Path
from holdout_lib import HERE,ORDER,frozen_rows,read,save,sha
from evaluate_holdout import summarize_data
from verify import verify

def audit(out):
    frames=frozen_rows();historic=read(HERE/"history.csv")
    official={Path(r["frame"]).stem:r["scene"] for r in read(HERE/"official_test_scenes.csv")}
    old_ids={r["frame_id"] for r in historic};old_scenes={official[i] for i in old_ids}
    retained={r["frame_id"] for r in frames};new_scenes={r["scene"] for r in frames}
    drop=read(HERE/"excluded_scenes.csv")
    assert old_scenes=={r["scene"] for r in drop} and not old_scenes&new_scenes
    assert retained=={i for i,s in official.items() if s not in old_scenes}
    for r in frames:assert r["scene"]==official[r["frame_id"]] and int(r["matlab_index"])==int(r["frame_id"][4:])+1
    for r in drop:
        ids={i for i,s in official.items() if s==r["scene"]}
        assert ids==set(r["all_excluded_ids"].split()) and ids&old_ids==set(r["used_ids"].split())
    windows=read(out/"windows.csv");pairs=read(out/"region_pairs.csv")
    assert len(windows)==16*len(frames)==len({(r["frame_id"],r["window"]) for r in windows})
    assert len(pairs)==len({(r["frame_id"],r["A"],r["B"]) for r in pairs})
    scene={r["frame_id"]:r["scene"] for r in frames}
    allowed={(r["frame_id"],r["window"]) for r in windows if int(r["valid_pixels"])*10>=9*1024}
    count=Counter(i for i,w in allowed)
    assert all((r["frame_id"],r["A"]) in allowed and (r["frame_id"],r["B"]) in allowed and int(r["A"])<int(r["B"]) for r in pairs)
    assert all(r["scene"]==scene[r["frame_id"]] for r in pairs)
    assert sum(n*(n-1)//2 for n in count.values())==len(pairs)
    for r in pairs:assert abs(float(r["G"])-float(r["G_product"]))<1e-12
    summarize_data(frames,pairs,out)
    # Independent upstream path: aggregate candidate windows before differencing and resample actual scene member frames.
    assert sha(ORDER/"verify.py")==json.loads((HERE/"sources.json").read_text())["source_files"]["P3/depth_order_pairing/verify.py"]
    verify(out)
    save(out/"split_replay.json",dict(official_test_frames=len(official),official_scenes=len(set(official.values())),
        historical_frames=len(old_ids),excluded_scenes=len(old_scenes),excluded_test_frames=len(official)-len(retained),
        new_frames=len(retained),new_scenes=len(new_scenes),shared_scenes=sorted(new_scenes&old_scenes),
        main_and_secondary_share_identical_pair_list=True,window_count=len(windows),eligible_pairs=len(pairs)))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE);audit(p.parse_args().out)
