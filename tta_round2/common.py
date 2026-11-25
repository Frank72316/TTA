import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "tta_pilot"
sys.dont_write_bytecode = True
sys.path.append(str(OLD))
from pilot_io import SUFFIX, digest, load_pairs, read_csv, write_csv, write_json

PAIRS = {p["pair_id"]:p for p in load_pairs()}
CELLS = [(s,l) for s in ("A","B") for l in ("L0","L1")]


def jsonfile(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen():
    name = ROOT/"prompts_round2.csv"
    manifest = jsonfile(ROOT/"freeze.json")
    assert digest(name) == manifest["prompts_sha256"]
    assert digest(OLD/"pairs.csv") == manifest["pairs_sha256"]
    rows = read_csv(name)
    assert [r["pair_id"] for r in rows] == list(PAIRS)
    for r in rows:
        p = PAIRS[r["pair_id"]]
        assert (r["original_prompt"], r["image_A"], r["image_B"]) == (p["P0"],p["image_A"],p["image_B"])
        assert r["Q1"] == r["Q0"]+" "+SUFFIX and int(r["Q1_tokens"]) <= int(r["context"])
    return rows


def images(pair_id):
    from PIL import Image, ImageEnhance
    out = []
    for side in ("A","B"):
        path = Path(PAIRS[pair_id]["image_"+side])
        with zipfile.ZipFile(OLD/"data"/(path.parent.name+".zip")) as arc:
            with Image.open(io.BytesIO(arc.read(path.name))) as im:
                rgb = im.convert("RGB")
        out.extend([rgb, ImageEnhance.Brightness(rgb).enhance(1.1)])
    return out


def index_scores(rows, ids, conditions):
    cells = {}
    for r in rows:
        i,s,c,l = (r[k] for k in ("pair_id","side","prompt_condition","light"))
        assert r["image_id"] == PAIRS[i]["image_"+s] and all(v not in (None,"") for v in r.values())
        key=(i,s,c,l)
        assert key not in cells and math.isfinite(float(r["score"])) and r["native_dtype"]=="float16"
        cells[key]=float(r["score"])
    assert set(cells)=={(i,s,c,l) for i in ids for s,l in CELLS for c in conditions}
    return cells


def token_record(text):
    from hpsv2.src.open_clip import get_tokenizer
    from hpsv2.src.open_clip.tokenizer import _tokenizer, tokenize
    tok = get_tokenizer("ViT-H-14")
    assert tok is tokenize
    actual = tok([text])[0].tolist()
    full = [_tokenizer.encoder["<start_of_text>"], *_tokenizer.encode(text), _tokenizer.encoder["<end_of_text>"]]
    end = actual.index(full[-1])+1
    assert not any(actual[end:])
    sent = actual[:end]
    assert sent == full[:len(sent)-1] + [full[-1]]
    return full,sent,len(actual)


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()

